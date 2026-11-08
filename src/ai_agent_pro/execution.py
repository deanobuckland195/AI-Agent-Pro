"""Session-owned, bounded chat execution; no Streamlit calls from worker threads."""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
import threading
from time import monotonic

from ai_agent_pro.config import AppError
from ai_agent_pro.approvals import ApprovalPaused
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.safety import redact

ACTIVE = frozenset({'Preparing', 'Using tools', 'Retrieving sources', 'Drafting', 'Reviewing', 'Generating', 'Saving', 'Stopping'})
PHASES = {'tool': 'Using tools', 'retrieval': 'Retrieving sources', 'draft': 'Drafting', 'review': 'Reviewing',
          'answer': 'Generating', 'save': 'Saving'}


@dataclass(frozen=True)
class Snapshot:
    state: str
    text: str
    error: str | None
    tools: tuple[dict, ...] = ()

    @property
    def active(self):
        return self.state in ACTIVE


class ChatJob:
    """Latest-only mailbox, scoped to the owning Streamlit session.

    At most 100k characters are held; deltas are never stored in SQLite or an
    unbounded event queue. Missing UI heartbeats cancel an abandoned job.
    """
    def __init__(self, repo, provider, agent_id, conversation_id, prompt, retrieval=None,
                 *, lease_seconds=15, tool_access=None, approval_id=None, practice=None, operation=None):
        self._tool_access = tool_access
        self._operation = operation
        self._approval_id, self._practice = approval_id, practice
        self._tools = {}
        self.agent_id, self.conversation_id = agent_id, conversation_id
        self._lock = threading.RLock()
        self._state, self._text, self._error = 'Preparing', '', None
        self._cancelled = False
        self._loop = self._task = None
        self._seen = monotonic()
        self._lease = lease_seconds
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(repo, provider, prompt, retrieval),
                                        name='agentpro-chat', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def snapshot(self):
        with self._lock:
            self._seen = monotonic()
            return Snapshot(self._state, self._text, self._error, tuple(deepcopy(list(self._tools.values()))))

    def cancel(self):
        with self._lock:
            # Saving is synchronous and atomic; completion wins a late Stop.
            if self._state not in ACTIVE or self._state in ('Saving', 'Stopping'):
                return False
            self._cancelled, self._state = True, 'Stopping'
            if self._loop and self._task:
                try:
                    self._loop.call_soon_threadsafe(self._task.cancel)
                except RuntimeError:
                    pass  # The worker has already completed and closed its loop.
            return True

    def _phase(self, event):
        if event['state'] == 'started':
            with self._lock:
                if not self._cancelled:
                    self._state = PHASES.get(event['stage'], self._state)

    def _tool(self, event):
        from ai_agent_pro.tools.contracts import redact_values
        safe = redact_values(deepcopy(event))
        # Knowledge excerpts are short-lived model context, not a second source
        # cache that could outlive document deletion or assignment revocation.
        if safe['tool'] == 'knowledge_search' and safe['data'] is not None:
            safe['data'] = {'method': safe['data']['method'], 'source_count': len(safe['data']['sources'])}
        with self._lock:
            self._tools[safe['ordinal']] = safe

    def _preview(self, text):
        with self._lock:
            if not self._cancelled:
                self._text = text

    def _run(self, repo, provider, prompt, retrieval):
        async def execute():
            with self._lock:
                self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()

            async def watchdog():
                while True:
                    await asyncio.sleep(min(1, self._lease))
                    with self._lock:
                        abandoned = monotonic() - self._seen > self._lease
                    if abandoned:
                        self.cancel()
                        return

            watcher = asyncio.create_task(watchdog())
            try:
                engine = AgentEngine(repo, provider, retrieval, observer=self._phase,
                    on_text=self._preview, cancel_requested=lambda: self._cancelled,
                    tool_access=self._tool_access, on_tool=self._tool)
                if self._operation:
                    return await self._operation(engine)
                if self._approval_id:
                    from ai_agent_pro.approval_resume import resume
                    return await resume(engine, self._approval_id, self.agent_id, self.conversation_id, self._practice)
                return await engine.execute(self.agent_id, self.conversation_id, prompt)
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                with self._lock:
                    self._loop = self._task = None

        try:
            result = asyncio.run(execute())
        except ApprovalPaused:
            with self._lock:
                self._state, self._text = 'Awaiting approval', ''
        except asyncio.CancelledError:
            with self._lock:
                self._state, self._text = 'Cancelled', ''
        except Exception as exc:
            with self._lock:
                self._state, self._text = 'Failed', ''
                self._error = redact(str(exc)) if isinstance(exc, AppError) else 'Generation failed safely. Check local configuration and try again.'
        else:
            with self._lock:
                self._state, self._text = 'Completed', result
        finally:
            self._done.set()

    def wait(self, timeout=5):
        """Bounded join for tests/shutdown; the UI polls snapshots instead."""
        return self._done.wait(timeout)
