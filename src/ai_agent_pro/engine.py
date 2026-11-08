"""Bounded sequential draft/review orchestration with durable execution records."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
from ai_agent_pro.approvals import ApprovalPaused, Approvals
from ai_agent_pro.practice import NAME as PRACTICE_NAME, SCHEMA as PRACTICE_SCHEMA
import logging
from time import monotonic
from ai_agent_pro.config import AppError
from ai_agent_pro.safety import instructions_for, redact, streaming_redact, token_reservation, validate_text
from ai_agent_pro.rag.context import RAG_POLICY, NO_EVIDENCE, evidence_message, validate_citations
from ai_agent_pro.telemetry import Recorder

logger = logging.getLogger("ai_agent_pro")


class AgentEngine:
    def __init__(self, repository, provider, retrieval=None, observer=None, on_text=None, cancel_requested=None, tool_access=None, on_tool=None, agent_snapshot=None, execution_policy="", workflow_identity=None, retrieval_query=None, on_tool_context=None):
        self.tool_access, self.on_tool = tool_access, on_tool
        self.on_tool_context = on_tool_context
        self.agent_snapshot, self.execution_policy = agent_snapshot, execution_policy
        self.workflow_identity, self.retrieval_query = workflow_identity, retrieval_query
        self.repo = repository
        self.provider = provider
        self.retrieval = retrieval
        self.observer = observer
        self.on_text = on_text
        self.cancel_requested = cancel_requested or (lambda: False)

    def check_cancelled(self):
        if self.cancel_requested():
            raise asyncio.CancelledError

    async def execute(self, agent_id, conversation_id, prompt):
        configured = self.repo.agent(agent_id).validate()
        agent = (self.agent_snapshot or configured).validate()
        if agent.id != agent_id:
            raise AppError('Execution agent snapshot does not match its principal.')
        prompt = validate_text(prompt, "Message", 12000)
        prompt = redact(prompt)
        self.repo.require_conversation(conversation_id, agent_id)
        history = [] if agent.settings.rag_enabled else self.repo.messages(conversation_id, completed_only=True)
        n = agent.settings.history_messages
        messages = [{"role": m['role'], "content": m['content']} for m in (history[-n:] if n else [])]
        # A bounded recent window is sent; the full transcript remains in SQLite.
        while messages and sum(len(m['content'].encode('utf-8')) for m in messages) > 24000:
            messages.pop(0)
        messages.append({"role": "user", "content": prompt})
        rid = self.repo.begin_run(conversation_id, agent, prompt, workflow_identity=self.workflow_identity)
        started = monotonic()
        requests, spent = 0, 0
        input_tokens, output_tokens = 0, 0
        usage_known = True
        steps = []
        sources = []
        recorder = Recorder(self.repo, run_id=rid, observer=self.observer)
        try:
            tool_session = self.tool_access.session(agent.id) if self.tool_access else None
            self.check_cancelled()
            settings = agent.settings
            if settings.rag_enabled:
                if self.retrieval is None:
                    raise AppError('Document retrieval is not configured. Check embedding credentials.')
                remaining = settings.timeout_seconds - (monotonic() - started)
                recorder.start('retrieval')
                sources, spent = await asyncio.wait_for(self.retrieval.retrieve(
                    agent.id, self.retrieval_query or prompt, rid, timeout=remaining, budget=settings.token_budget), remaining)
                if monotonic() - started > settings.timeout_seconds:
                    raise AppError('Task deadline reached.')
                recorder.complete('retrieval', completed=len(sources), outcome='sources_selected' if sources else 'no_evidence')
                if not sources:
                    self.check_cancelled()
                    recorder.start('save')
                    self.check_cancelled()
                    self.repo.finish_run(rid, 'completed', 0, 0, 0, monotonic() - started, [], answer=NO_EVIDENCE)
                    recorder.complete('save', outcome='no_evidence')
                    return NO_EVIDENCE
                # Source scope is resolved afresh for each question. Prior answers may
                # contain deleted/revoked documents, so grounded mode omits transcript context.
                messages = [evidence_message(sources, prompt)]
            for step in range(1, settings.max_steps + 1):
                round_messages = list(messages)
                while True:
                    self.check_cancelled()
                    remaining = settings.timeout_seconds - (monotonic() - started)
                    if remaining <= 0:
                        raise AppError("Task deadline reached.")
                    instructions = instructions_for(agent, step, settings.max_steps) + self.execution_policy
                    schemas = tool_session.schemas() if tool_session else []
                    practice = self.tool_access and self.tool_access.practice
                    if practice and self.tool_access.practice_files:
                        schemas.append(deepcopy(PRACTICE_SCHEMA))
                    if schemas:
                        from ai_agent_pro.tool_calling import TOOL_POLICY
                        from ai_agent_pro.safety import BOUNDARY
                        instructions = instructions.replace(BOUNDARY, TOOL_POLICY, 1)
                        if practice:
                            instructions = instructions.replace('No shell, network connectors, writes or arbitrary code execution are available.', 'No shell, network connectors or arbitrary code execution are available. Only the exact human-approved practice replacement may write a file.')
                            instructions += '\nPractice changes require human approval. Propose at most one replacement; no file has changed until approval and a successful tool result. Selected practice filenames: ' + repr(self.tool_access.practice_files)
                    if settings.rag_enabled:
                        self.retrieval.knowledge.require_sources(agent.id, sources)
                        instructions += RAG_POLICY
                    if tool_session:
                        from ai_agent_pro.tool_calling import MAX_MODEL_REQUESTS, tool_reservation
                        if requests >= MAX_MODEL_REQUESTS:
                            raise AppError('Model request limit reached; execution stopped.')
                        reserve = tool_reservation(instructions, round_messages, schemas, settings.max_output_tokens)
                    else:
                        reserve = token_reservation(instructions, round_messages, settings.max_output_tokens)
                    if spent + reserve > settings.token_budget:
                        raise AppError("Task token budget cannot cover the next request. Shorten context or raise the budget.")
                    if tool_session:
                        # Permission/source revalidation may consume time too.
                        remaining = settings.timeout_seconds - (monotonic() - started)
                        if remaining <= 0:
                            raise AppError('Task deadline reached.')
                    requests += 1
                    stage = 'answer' if settings.max_steps == 1 else ('draft' if step == 1 else 'review')
                    recorder.start(stage, ordinal=requests if tool_session else step, total=settings.max_steps if not tool_session else None)
                    # Explicit UI selection must produce an actual tool result,
                    # rather than a text-only claim. After that, allow the model
                    # to finish (or request another approved sequential call).
                    require_tool = bool(schemas and not tool_session.results)
                    stream_arguments = {'tools': schemas, 'tool_choice': 'required' if require_tool else 'auto'} if schemas else {}
                    if tool_session and self.on_text and step == settings.max_steps:
                        self.on_text('')
                    if self.on_text and step == settings.max_steps:
                        preview = ''

                        def delta(text):
                            nonlocal preview
                            self.check_cancelled()
                            preview += text
                            if len(preview) > 100000:
                                raise AppError('The model response exceeded the local size limit.')
                            try:
                                self.on_text(streaming_redact(preview))
                            except Exception:
                                logger.warning('Streaming preview unavailable.')

                        stream_arguments['on_delta'] = delta
                    result = await asyncio.wait_for(self.provider.complete(
                        model=settings.model, instructions=instructions, messages=list(round_messages),
                        max_output_tokens=settings.max_output_tokens, timeout=remaining, **stream_arguments), timeout=remaining)
                    self.check_cancelled()
                    if result.input_tokens is None or result.output_tokens is None:
                        usage_known = False
                        spent += reserve
                    else:
                        input_tokens += result.input_tokens
                        output_tokens += result.output_tokens
                        spent += result.input_tokens + result.output_tokens
                    steps.append({"step": requests if tool_session else step, "stage": "answer" if settings.max_steps == 1 else
                                  ("draft" if step == 1 else "review"), "input_tokens": result.input_tokens,
                                  "output_tokens": result.output_tokens})
                    recorder.complete(stage, ordinal=requests if tool_session else step, completed=step, total=settings.max_steps)
                    if monotonic() - started > settings.timeout_seconds:
                        raise AppError("Task deadline reached.")
                    if spent > settings.token_budget:
                        raise AppError("Reported token usage exceeded the task budget; execution stopped.")
                    if result.calls:
                        if tool_session is None:
                            raise AppError('The model requested tools without execution approval.')
                        if practice and result.calls[0].name == PRACTICE_NAME:
                            from ai_agent_pro.tool_calling import MAX_TOOL_CALLS, wire_name
                            from ai_agent_pro.tools.contracts import parse_json
                            call = result.calls[0]
                            if len(result.calls) != 1 or call.call_id in tool_session.seen or len(tool_session.seen) >= MAX_TOOL_CALLS:
                                raise AppError('Duplicate or excessive practice change call rejected.')
                            try:
                                arguments = parse_json(call.arguments)
                            except (ValueError, TypeError, RecursionError):
                                raise AppError('Practice change arguments are invalid.') from None
                            self.check_cancelled()
                            tool_session.recheck()
                            checkpoint = dict(messages=round_messages + list(result.output), step=step,
                                settings=settings.to_dict(), instructions=agent.instructions, execution_policy=self.execution_policy,
                                requests=requests, spent=spent, input_tokens=input_tokens,
                                output_tokens=output_tokens, usage_known=usage_known, steps=steps,
                                elapsed=monotonic()-started, sources=[asdict(s) for s in sources],
                                knowledge_results=[dict(data=data, context=tool_session.executor.permissions.context(
                                    tool_session.grants[wire_name(definition)]))
                                    for definition, data in tool_session.results if definition.permission == 'knowledge.read'])
                            Approvals(self.repo, practice).propose(rid, agent.id, call, arguments,
                                self.tool_access.practice_files, checkpoint)
                            recorder.emit('approval', 'started', outcome='Awaiting human decision')
                            raise ApprovalPaused()
                        prepared = tool_session.prepare(result.calls)
                        def notify(definition, ordinal, state, data, elapsed=None):
                            # Retain only fixed registry names/statuses, never source excerpts or arguments.
                            recorder.emit('tool', state, ordinal, elapsed=elapsed,
                                          outcome=f'{definition.name} v{definition.version}')
                            if self.on_tool_context and state == 'completed':
                                self.on_tool_context(dict(tool=definition.name,version=definition.version,ordinal=ordinal,state=state,data=deepcopy(data)))
                            if self.on_tool:
                                try:
                                    self.on_tool(dict(tool=definition.name, version=definition.version,
                                        ordinal=ordinal, state=state, data=deepcopy(data)))
                                except Exception:
                                    logger.warning('Tool activity observer unavailable.')
                        remaining = settings.timeout_seconds - (monotonic() - started)
                        if remaining <= 0:
                            raise AppError('Task deadline reached.')
                        output = await tool_session.execute(prepared, timeout=remaining,
                            check_cancelled=self.check_cancelled, notify=notify)
                        round_messages.extend(result.output)
                        round_messages.append(output)
                        continue
                    if require_tool:
                        raise AppError('The model did not call an explicitly selected tool. No answer was saved.')
                    if step < settings.max_steps:
                        messages.append({"role": "assistant", "content": redact(result.text)})
                    answer = result.text
                    break
            if settings.rag_enabled:
                validate_citations(answer, sources)
            if tool_session:
                tool_session.recheck()
            self.check_cancelled()
            recorder.start('save')
            # The worker's save observer and Stop share a lock: either Stop
            # wins here or Saving begins and late Stop is rejected.
            self.check_cancelled()
            if tool_session:
                tool_session.recheck()
            if monotonic() - started > settings.timeout_seconds:
                raise AppError('Task deadline reached.')
            self.repo.finish_run(rid, 'completed', requests,
                                 input_tokens if usage_known else None, output_tokens if usage_known else None,
                                 monotonic() - started, steps, answer=answer, sources=sources)
            recorder.complete('save')
            return redact(answer)
        except ApprovalPaused:
            raise
        except asyncio.CancelledError:
            error = 'Cancelled by user or disconnected session; usage may be unavailable.'
            recorder.fail(error)
            self.repo.finish_run(rid, 'interrupted', requests, None, None,
                                 monotonic() - started, steps, error=error)
            raise
        except Exception as exc:
            error = str(exc) if isinstance(exc, AppError) else (
                "Task deadline reached." if isinstance(exc, asyncio.TimeoutError) else
                "Unexpected execution failure. Check application configuration; no automatic retry was made.")
            # On failed API calls total usage cannot be known, including possibly billable failed calls.
            recorder.fail(error)
            self.repo.finish_run(rid, 'failed', requests, None, None,
                                 monotonic() - started, steps, error=error)
            logger.warning("Execution %s failed: %s", rid, error)
            raise AppError(error) from None
