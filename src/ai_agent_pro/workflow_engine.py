"""Sequential coordinator over isolated existing agent executions.

Only trusted host UI code supplies StepAccess. Definitions, handoffs and saved
checkpoints cannot grant tools or transfer a human decision to another agent.
"""
import asyncio
from dataclasses import dataclass
import json
from time import monotonic
from uuid import uuid4

from ai_agent_pro.approvals import ApprovalPaused, encode
from ai_agent_pro.approval_resume import resume
from ai_agent_pro.config import Agent, AppError, Settings
from ai_agent_pro.database import now
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.practice import digest
from ai_agent_pro.tool_calling import tool_reservation
from ai_agent_pro.tools.contracts import ToolFailure
from ai_agent_pro.tools.filesystem import read_file
from ai_agent_pro.workflow_contracts import Limits, POLICY, artifact, step_input
from ai_agent_pro.workflow_store import WorkflowStore


@dataclass(frozen=True)
class StepAccess:
    agent_id: str
    tools: object = None
    context_executor: object = None
    files: tuple = ()
    owned: bool = False

    def validate(self, agent_id):
        if self.agent_id != agent_id:
            raise AppError('Step permission principal does not match the configured agent.')
        if len(self.files) > 32 or len(set(self.files)) != len(self.files):
            raise AppError('Explicit step file manifests are limited to 32 distinct files.')
        if self.tools:
            session = self.tools.session(agent_id)
            for alias, d in session.definitions.items():
                if d.permission == 'project.read' and set(session.grants[alias].files) != set(self.files):
                    raise AppError('Project file permissions must be selected for this step, never inherited.')
        return self


def file_hash(permission, name):
    authority = permission.context_executor or (permission.tools.executor if permission.tools else None)
    if authority is None or name not in permission.files:
        raise AppError('Handoff contains project context not explicitly authorised for this step.')
    try:
        text, size = read_file(str(authority.permissions.root), authority.permissions.identity, name)
    except ToolFailure:
        raise AppError('Authorised workflow project context changed or is unsafe.') from None
    return digest((str(size)+'\n'+text).encode())


class BudgetedProvider:
    def __init__(self, engine, backend, ordinal):
        self.engine,self.backend,self.ordinal = engine,backend,ordinal

    async def complete(self, **kwargs):
        e = self.engine
        e.guard()
        reserve = tool_reservation(kwargs['instructions'],kwargs['messages'],kwargs.get('tools',[]),kwargs['max_output_tokens'])
        e.reserve(reserve)
        kwargs['timeout'] = min(kwargs['timeout'],e.remaining())
        result = await asyncio.wait_for(self.backend.complete(**kwargs),kwargs['timeout'])
        if any(v is not None and (type(v) is not int or v < 0) for v in (result.input_tokens,result.output_tokens)):
            raise AppError('Workflow model usage was invalid.')
        measured = None if result.input_tokens is None or result.output_tokens is None else result.input_tokens+result.output_tokens
        e.charge(reserve,measured,len(result.calls))
        e.guard()
        return result


class BudgetedEmbeddings:
    def __init__(self,engine,backend):
        self.engine,self.backend = engine,backend
        self.model,self.dimensions = backend.model,backend.dimensions

    async def embed(self,texts,*,timeout):
        self.engine.guard()
        reserve = sum(len(t.encode()) for t in texts)
        self.engine.reserve(reserve)
        result = await asyncio.wait_for(self.backend.embed(texts,timeout=min(timeout,self.engine.remaining())),min(timeout,self.engine.remaining()))
        self.engine.charge(reserve,result.input_tokens,0)
        return result


class WorkflowEngine:
    def __init__(self,repo,provider,*,retrieval_factory=None,workspace=None,on_text=None,on_tool=None,observer=None,cancel_requested=None):
        self.repo,self.provider,self.store = repo,provider,WorkflowStore(repo)
        self.retrieval_factory,self.workspace = retrieval_factory,workspace
        self.on_text,self.on_tool,self.observer = on_text,on_tool,observer
        self.cancel_requested = cancel_requested or (lambda:False)

    def elapsed(self):
        return self.base_seconds+monotonic()-self.started

    def remaining(self):
        seconds = self.limits.timeout_seconds-self.elapsed()
        if seconds <= 0:
            raise AppError('Workflow execution deadline reached.')
        return seconds

    def guard(self):
        if self.cancel_requested():
            raise asyncio.CancelledError
        row = self.store.get(self.wid)
        if row['owner'] != self.owner or row['status'] != 'running':
            raise AppError('Workflow lease or execution ownership changed.')
        if row['cancel_requested']:
            raise asyncio.CancelledError
        self.remaining()
        self.authorize_context(self.provenance,self.permission)
        return False

    def authorize_context(self,provenance,permission):
        docs = provenance.get('documents',[])
        with self.repo.connect() as db:
            for doc,generation in docs:
                if not db.execute('''SELECT 1 FROM documents d JOIN agent_collections a ON a.collection_id=d.collection_id
                    WHERE d.id=? AND d.active_generation=? AND a.agent_id=?''',(doc,generation,permission.agent_id)).fetchone():
                    raise AppError('Handoff document context is not authorised for the next agent or has been revoked.')
        for name,expected in provenance.get('files',{}).items():
            if file_hash(permission,name) != expected:
                raise AppError('Handoff project file changed; no downstream agent received stale context.')

    def reserve(self,tokens):
        self.guard()
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workflow_runs WHERE id=? AND owner=?',(self.wid,self.owner)).fetchone()
            if not row or row['status'] != 'running' or row['cancel_requested']:
                raise AppError('Workflow is no longer authorised to make a model request.')
            if row['requests'] >= self.limits.max_requests or row['tokens']+tokens > self.limits.token_budget:
                raise AppError('Workflow model-request or token budget cannot cover another request.')
            db.execute('UPDATE workflow_runs SET requests=requests+1,tokens=tokens+?,active_seconds=?,updated_at=? WHERE id=?',
                (tokens,self.elapsed(),now(),self.wid))

    def charge(self,reserve,actual,tools):
        if actual is not None and (type(actual) is not int or actual < 0):
            raise AppError('Workflow model usage was invalid.')
        failure = None
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workflow_runs WHERE id=? AND owner=?',(self.wid,self.owner)).fetchone()
            if not row:
                raise AppError('Workflow ownership changed after a model request.')
            total = row['tokens'] + (actual-reserve if actual is not None else 0)
            calls = row['tool_calls']+tools
            db.execute('UPDATE workflow_runs SET tokens=?,tool_calls=?,active_seconds=?,updated_at=? WHERE id=?',
                       (total,calls,self.elapsed(),now(),self.wid))
            if total > self.limits.token_budget or calls > self.limits.max_tool_calls:
                failure = 'Workflow token or tool-call budget exceeded; no further action may execute.'
        if failure:
            raise AppError(failure)

    def collect(self,event):
        # Persist only provenance IDs, never raw results, excerpts or grants.
        if event['state'] == 'completed' and event['tool'] == 'knowledge_search':
            docs = {tuple(x) for x in self.provenance.get('documents',[])}
            docs.update((s['document_id'],s['generation_id']) for s in event['data']['sources'])
            self.provenance['documents'] = sorted(docs)
            self.save_provenance()

    def save_provenance(self):
        with self.repo.connect() as db:
            db.execute('UPDATE workflow_steps SET provenance=? WHERE workflow_id=? AND ordinal=?',
                       (encode(self.provenance),self.wid,self.ordinal))

    def child(self,step):
        if not step['run_id']:
            return None
        with self.repo.connect() as db:
            r = db.execute('SELECT * FROM runs WHERE id=?',(step['run_id'],)).fetchone()
            return dict(r) if r else None

    @staticmethod
    def checkpoint(step):
        return digest(encode(dict(workflow_id=step['workflow_id'],ordinal=step['ordinal'],agent_id=step['agent_id'],
            run_id=step['run_id'],output=step['output'],provenance=step['provenance'])).encode())

    def complete_step(self,step,answer):
        value = artifact(answer)
        with self.repo.connect() as db:
            refs = db.execute('SELECT document_id,generation_id FROM run_sources WHERE run_id=?',(step['run_id'],)).fetchall()
        docs = {tuple(x) for x in self.provenance.get('documents',[])}
        docs.update((r['document_id'],r['generation_id']) for r in refs)
        self.provenance['documents'] = sorted(docs)
        if len(docs)>100:
            raise AppError('Workflow handoff provenance limit exceeded.')
        self.authorize_context(self.provenance,self.permission)
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workflow_runs WHERE id=? AND owner=?',(self.wid,self.owner)).fetchone()
            if not row or row['cancel_requested'] or row['cursor'] != self.ordinal:
                raise AppError('Workflow was cancelled or its checkpoint changed before the handoff.')
            saved = dict(step,output=encode(value),provenance=encode(self.provenance))
            db.execute("UPDATE workflow_steps SET status='completed',output=?,provenance=?,checkpoint_hash=? WHERE workflow_id=? AND ordinal=?",
                (saved['output'],saved['provenance'],self.checkpoint(saved),self.wid,self.ordinal))
            db.execute('UPDATE workflow_runs SET cursor=cursor+1,active_seconds=?,updated_at=? WHERE id=?',
                       (self.elapsed(),now(),self.wid))
            self.store.event(db,self.wid,self.ordinal,'completed','Validated structured handoff saved')
        return value

    async def run(self,wid,accesses=None,*,approval_id=None):
        self.wid = wid
        try:
            approval_run = None
            if approval_id:
                with self.repo.connect() as db:
                    action = db.execute('SELECT run_id FROM approvals WHERE id=?',(approval_id,)).fetchone()
                if not action:
                    raise AppError('Workflow approval does not exist.')
                approval_run = action['run_id']
            self.owner,row = self.store.claim(wid,approval_run_id=approval_run)
        except BaseException:
            for permission in (accesses or {}).values():
                if permission.owned and permission.context_executor:
                    permission.context_executor.permissions.close()
            raise
        self.started,self.base_seconds = monotonic(),row['active_seconds']
        self.limits = Limits(**row['data']['definition']['limits']).validate()
        accesses = accesses or {}
        last = None
        permissions = list(accesses.values())
        current_task = asyncio.current_task()
        async def cancellation_watch():
            while True:
                await asyncio.sleep(.2)
                if self.store.get(wid)['cancel_requested']:
                    current_task.cancel()
                    return
        watcher = asyncio.create_task(cancellation_watch())
        try:
            for ordinal in range(row['cursor'],len(row['data']['definition']['steps'])):
                self.ordinal = ordinal
                spec = row['data']['definition']['steps'][ordinal]
                snapshot = row['data']['agents'][ordinal]
                self.permission = accesses.get(ordinal,StepAccess(spec['agent_id'])).validate(spec['agent_id'])
                steps = self.store.steps(wid)
                step = steps[ordinal]
                if ordinal and (steps[ordinal-1]['status'] != 'completed' or self.checkpoint(steps[ordinal-1]) != steps[ordinal-1]['checkpoint_hash']):
                    raise AppError('Workflow handoff checkpoint changed or is incomplete.')
                previous = json.loads(steps[ordinal-1]['output']) if ordinal else None
                self.provenance = json.loads(steps[ordinal-1]['provenance']) if ordinal else {}
                # A pending/completed child has its own additionally collected context.
                if step['run_id']:
                    self.provenance = json.loads(step['provenance'])
                self.guard()
                if self.permission.tools and any(n in ('project_read','project_search') for n,v in self.permission.tools.selected):
                    files = dict(self.provenance.get('files',{}))
                    files.update({name:file_hash(self.permission,name) for name in self.permission.files})
                    self.provenance['files'] = files
                prompt = step_input(row['goal'],spec,previous,ordinal)
                if not step['conversation_id']:
                    cid = str(uuid4())
                    with self.repo.connect() as db:
                        db.execute('BEGIN IMMEDIATE')
                        db.execute('INSERT INTO conversations VALUES (?,?,?,?)',(cid,spec['agent_id'],f"Workflow: {spec['role']}",now()))
                        db.execute("UPDATE workflow_steps SET conversation_id=?,status='running',input=?,provenance=? WHERE workflow_id=? AND ordinal=?",
                            (cid,prompt,encode(self.provenance),wid,ordinal))
                        self.store.event(db,wid,ordinal,'running',spec['role'])
                    step = self.store.steps(wid)[ordinal]
                elif not step['run_id']:
                    with self.repo.connect() as db:
                        db.execute("UPDATE workflow_steps SET status='running',input=?,provenance=? WHERE workflow_id=? AND ordinal=? AND run_id IS NULL",(prompt,encode(self.provenance),wid,ordinal))
                child = self.child(step)
                if child and child['status'] == 'completed':
                    with self.repo.connect() as db:
                        message = db.execute("SELECT content FROM messages WHERE run_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",(step['run_id'],)).fetchone()
                    if not message:
                        raise AppError('Completed workflow child has no saved output; it will not be repeated.')
                    last = self.complete_step(step,message['content'])
                    continue
                if child and child['status'] != 'waiting_approval':
                    raise AppError('Interrupted or failed workflow child cannot be retried automatically.')
                actor = Agent(snapshot['id'],snapshot['name'],snapshot['description'],snapshot['instructions'],Settings(**snapshot['settings'])).validate()
                retrieval = None
                if actor.settings.rag_enabled:
                    if not self.retrieval_factory:
                        raise AppError('Workflow document retrieval is not configured.')
                    retrieval = self.retrieval_factory(actor.id,self)
                engine = AgentEngine(self.repo,BudgetedProvider(self,self.provider,ordinal),retrieval,
                    observer=self.observer,on_text=self.on_text,on_tool=self.on_tool,on_tool_context=self.collect,cancel_requested=self.guard,
                    tool_access=self.permission.tools,agent_snapshot=actor,execution_policy=POLICY+'\nStep role: '+spec['role'],
                    workflow_identity=(wid,ordinal,self.owner),retrieval_query=row['goal']+'\n'+spec['task'])
                if child:
                    if not approval_id or self.workspace is None:
                        raise AppError('Review and approve the exact pending workflow action before resuming.')
                    with self.repo.connect() as db:
                        action = db.execute('SELECT run_id FROM approvals WHERE id=?',(approval_id,)).fetchone()
                    if not action or action['run_id'] != child['id']:
                        raise AppError('Approval belongs to a different workflow step.')
                    answer = await asyncio.wait_for(resume(engine,approval_id,actor.id,step['conversation_id'],self.workspace),self.remaining())
                    step = self.store.steps(wid)[ordinal]
                    last = self.complete_step(step,answer)
                    # Approve is consent for one action, never for subsequent agents.
                    self.store.end(wid,self.owner,'completed' if ordinal+1 == len(steps) else 'ready',self.elapsed())
                    return encode(last)
                self.save_provenance()
                answer = await asyncio.wait_for(engine.execute(actor.id,step['conversation_id'],prompt),self.remaining())
                step = self.store.steps(wid)[ordinal]
                last = self.complete_step(step,answer)
            self.store.end(wid,self.owner,'completed',self.elapsed())
            return encode(last)
        except ApprovalPaused:
            step = self.store.steps(wid)[self.ordinal]
            with self.repo.connect() as db:
                db.execute("UPDATE workflow_steps SET status='waiting_approval' WHERE workflow_id=? AND ordinal=?",(wid,self.ordinal))
                cp = db.execute('SELECT state FROM execution_checkpoints WHERE run_id=?',(step['run_id'],)).fetchone()
            if cp and cp['state']:
                saved = json.loads(cp['state'])
                docs = {tuple(x) for x in self.provenance.get('documents',[])}
                docs.update((s['document_id'],s['generation_id']) for s in saved['sources'])
                self.provenance['documents'] = sorted(docs)
                self.save_provenance()
            if self.store.get(wid)['cancel_requested']:
                self.store.end(wid,self.owner,'cancelled',self.elapsed(),'Cancelled before the approval pause.')
                raise asyncio.CancelledError
            self.store.end(wid,self.owner,'waiting_approval',self.elapsed())
            raise
        except BaseException as exc:
            cancelled = isinstance(exc,asyncio.CancelledError) or self.store.get(wid)['cancel_requested']
            message = 'Workflow cancelled; completed actions are retained.' if cancelled else str(exc) if isinstance(exc,AppError) else 'Workflow execution deadline reached; no automatic retry was made.' if isinstance(exc,asyncio.TimeoutError) else 'Workflow failed safely; no automatic retry was made.'
            with self.repo.connect() as db:
                db.execute("UPDATE workflow_steps SET status=? WHERE workflow_id=? AND ordinal=? AND status NOT IN ('completed','waiting_approval')",('cancelled' if cancelled else 'failed',wid,self.ordinal))
            self.store.end(wid,self.owner,'cancelled' if cancelled else 'failed',self.elapsed(),message)
            if isinstance(exc,asyncio.CancelledError):
                raise
            raise AppError(message) from None
        finally:
            watcher.cancel()
            await asyncio.gather(watcher,return_exceptions=True)
            for permission in permissions:
                if permission.owned and permission.context_executor:
                    permission.context_executor.permissions.close()
