"""Durable action-specific decisions and single-consumer execution checkpoints."""
import difflib
import json
from time import time
from uuid import uuid4

from ai_agent_pro.config import AppError
from ai_agent_pro.database import now
from ai_agent_pro.practice import NAME, SCHEMA, content, digest, filename
from ai_agent_pro.tools.contracts import ToolFailure, validate, redact_values

TTL = 900


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


class ApprovalPaused(Exception):
    pass


class Approvals:
    def __init__(self, repo, workspace):
        self.repo, self.workspace = repo, workspace

    def _row(self, db, action_id, agent_id, run_id):
        row = db.execute('''SELECT a.* FROM approvals a JOIN runs r ON r.id=a.run_id
            JOIN conversations c ON c.id=r.conversation_id
            WHERE a.id=? AND a.agent_id=? AND a.run_id=? AND c.agent_id=?''',
            (action_id, agent_id, run_id, agent_id)).fetchone()
        if not row:
            raise AppError('Approval does not belong to this agent and execution.')
        value = dict(row)
        try:
            p = json.loads(value['proposal'])
        except (ValueError, TypeError, RecursionError):
            raise AppError('Stored approval is corrupted; no file may be modified.') from None
        required = {'id','run_id','agent_id','call_id','tool','version','arguments','after_hash','expires_at','path','before','before_hash','file_identity','workspace_identity','project_identity','checkpoint_hash'}
        if type(p) is not dict or set(p) != required:
            raise AppError('Stored approval fields are invalid; no file may be modified.')
        if digest(value['proposal'].encode()) != value['binding'] or p['id'] != action_id or p['run_id'] != run_id or p['agent_id'] != agent_id or p['tool'] != NAME:
            raise AppError('Approval binding is invalid; no action may execute.')
        try:
            validate(SCHEMA['parameters'], p['arguments'], 'invalid_input')
            filename(p['arguments']['path'])
            content(p['arguments']['content'])
        except ToolFailure:
            raise AppError('Approval arguments are invalid.') from None
        if p['after_hash'] != digest(p['arguments']['content'].encode()):
            raise AppError('Approval content hash is invalid.')
        if p['version'] != '1.0.0' or p['path'] != str(self.workspace.path / p['arguments']['path']):
            raise AppError('Approval tool version or workspace binding is invalid.')
        value['payload'] = p
        return value

    def propose(self, run_id, agent_id, call, arguments, allowed_files, checkpoint):
        try:
            validate(SCHEMA['parameters'], arguments, 'invalid_input')
        except ToolFailure:
            raise AppError('Practice change arguments do not match the approved schema.') from None
        name = filename(arguments['path'])
        if name not in allowed_files:
            raise AppError('This practice file was not explicitly selected for this execution.')
        after = content(arguments['content'])
        before = self.workspace.read(name)
        if before['before'] == after:
            raise AppError('The proposed change is empty; no approval was created.')
        action_id = str(uuid4())
        proposal = dict(id=action_id, run_id=run_id, agent_id=agent_id, call_id=call.call_id,
            tool=NAME, version='1.0.0', arguments=arguments, after_hash=digest(after.encode()),
            expires_at=time() + TTL, path=str(self.workspace.path / name), **before)
        state = encode(redact_values(checkpoint))
        proposal['checkpoint_hash'] = digest(state.encode())
        raw = encode(proposal)
        if len(state.encode()) > 131072:
            raise AppError('Approval checkpoint exceeded its context limit.')
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            owner = db.execute('''SELECT 1 FROM runs r JOIN conversations c ON c.id=r.conversation_id
                WHERE r.id=? AND r.status='running' AND c.agent_id=?''', (run_id, agent_id)).fetchone()
            if not owner:
                raise AppError('Execution is no longer running.')
            db.execute('INSERT INTO approvals(id,run_id,agent_id,proposal,binding,status,created_at) VALUES (?,?,?,?,?,?,?)',
                       (action_id, run_id, agent_id, raw, digest(raw.encode()), 'pending', now()))
            db.execute('INSERT INTO execution_checkpoints VALUES (?,?,?)', (run_id, state, 'ready'))
            db.execute("UPDATE runs SET status='waiting_approval',requests=?,input_tokens=?,output_tokens=?,elapsed=?,steps=? WHERE id=?",
                (checkpoint['requests'], checkpoint['input_tokens'] if checkpoint['usage_known'] else None,
                 checkpoint['output_tokens'] if checkpoint['usage_known'] else None, checkpoint['elapsed'], encode(checkpoint['steps']), run_id))
        return action_id

    def list(self, agent_id, conversation_id):
        self.expire()
        with self.repo.connect() as db:
            ids = db.execute('''SELECT a.id,a.run_id FROM approvals a JOIN runs r ON r.id=a.run_id
                WHERE a.agent_id=? AND r.conversation_id=? ORDER BY a.created_at DESC LIMIT 20''', (agent_id, conversation_id)).fetchall()
            return [self._row(db, r['id'], agent_id, r['run_id']) for r in ids]

    @staticmethod
    def diff(row):
        p = row['payload']
        lines = difflib.unified_diff(p['before'].splitlines(keepends=True), p['arguments']['content'].splitlines(keepends=True),
            fromfile=p['path'] + ' (before)', tofile=p['path'] + ' (after)')
        return ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in lines)

    def _stop(self, db, row, status, message):
        db.execute('UPDATE approvals SET status=?,decided_at=?,error=? WHERE id=?', (status, now(), message, row['id']))
        db.execute("UPDATE runs SET status='interrupted',finished_at=?,error=? WHERE id=? AND status='waiting_approval'", (now(), message, row['run_id']))
        db.execute("UPDATE execution_checkpoints SET state=NULL,phase='finished' WHERE run_id=?", (row['run_id'],))

    def expire(self):
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT * FROM approvals WHERE status IN ('pending','approved')").fetchall():
                if json.loads(row['proposal'])['expires_at'] <= time():
                    self._stop(db, row, 'expired', 'Approval expired; create a new request. No file was modified.')

    def decide(self, action_id, agent_id, run_id, binding, approve):
        self.expire()
        failure = None
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._row(db, action_id, agent_id, run_id)
            if row['status'] != 'pending' or row['binding'] != binding or type(approve) is not bool:
                raise AppError('Approval is stale, already decided or does not match the displayed action.')
            if not approve:
                self._stop(db, row, 'denied', 'File change denied. No file was modified.')
            else:
                try:
                    current = self.workspace.read(row['payload']['arguments']['path'])
                    if any(current[k] != row['payload'][k] for k in current):
                        raise AppError('File changed since proposal; request a new approval.')
                    self._checkpoint(db, run_id)
                except AppError as exc:
                    failure = str(exc)
                    self._stop(db, row, 'invalid', failure)
                else:
                    db.execute("UPDATE approvals SET status='approved',decided_at=? WHERE id=?", (now(), action_id))
        if failure:
            raise AppError(failure)

    def cancel(self, action_id, agent_id, run_id):
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._row(db, action_id, agent_id, run_id)
            if row['status'] not in ('pending', 'approved'):
                raise AppError('Only an unexecuted approval can be cancelled.')
            self._stop(db, row, 'cancelled', 'Paused execution cancelled. No file was modified.')

    def _checkpoint(self, db, run_id):
        row = db.execute('SELECT * FROM execution_checkpoints WHERE run_id=?', (run_id,)).fetchone()
        if not row or row['state'] is None:
            raise AppError('Checkpoint is unavailable or its knowledge sources were deleted; create a new request.')
        return dict(row)

    def claim(self, action_id, agent_id, conversation_id, workflow_identity=None):
        self.expire()
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            managed = db.execute('SELECT s.workflow_id,s.ordinal,w.owner,w.status,w.cancel_requested FROM workflow_steps s JOIN workflow_runs w ON w.id=s.workflow_id WHERE s.conversation_id=?',(conversation_id,)).fetchone()
            if managed and (workflow_identity != (managed['workflow_id'],managed['ordinal'],managed['owner']) or managed['status'] != 'running' or managed['cancel_requested']):
                raise AppError('Workflow approvals must resume through the owning coordinator.')
            row = db.execute('SELECT run_id FROM approvals WHERE id=?', (action_id,)).fetchone()
            if not row:
                raise AppError('Approval no longer exists.')
            row = self._row(db, action_id, agent_id, row['run_id'])
            cp = self._checkpoint(db, row['run_id'])
            if digest(cp['state'].encode()) != row['payload']['checkpoint_hash']:
                raise AppError('Execution checkpoint binding changed.')
            if row['status'] not in ('approved', 'applying', 'applied') or cp['phase'] == 'generating':
                raise AppError('Approval cannot be resumed or was already consumed.')
            changed = db.execute("UPDATE runs SET status='running',active_since=? WHERE id=? AND conversation_id=? AND status='waiting_approval'", (now(), row['run_id'], conversation_id)).rowcount
            if changed != 1:
                raise AppError('Execution is already claimed, finalized or belongs to another conversation.')
        return row, json.loads(cp['state'])

    def apply(self, row, check=lambda: None):
        """Only a claimed continuation calls this; CAS prevents duplicate writers."""
        check()
        p = row['payload']
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            fresh = self._row(db, row['id'], p['agent_id'], p['run_id'])
            p = fresh['payload']
            if fresh['status'] != 'approved' or p['expires_at'] <= time():
                raise AppError('Approval is expired, consumed or needs crash recovery.')
            if db.execute("SELECT status FROM runs WHERE id=?", (p['run_id'],)).fetchone()[0] != 'running':
                raise AppError('Execution must claim its checkpoint before applying a change.')
            db.execute("UPDATE approvals SET status='applying' WHERE id=?", (row['id'],))
            db.execute("UPDATE execution_checkpoints SET phase='applying' WHERE run_id=?", (p['run_id'],))
        # A crash from here never permits a second write. Recovery reconciles
        # the expected post-image or reports uncertainty without reapplying.
        self.workspace.replace(p, check=check)
        self.applied(row)

    def applied(self, row):
        with self.repo.connect() as db:
            db.execute("UPDATE approvals SET status='applied',executed_at=? WHERE id=? AND status='applying'", (now(), row['id']))
            db.execute("UPDATE execution_checkpoints SET phase='applied' WHERE run_id=?", (row['run_id'],))

    def reconcile(self, row):
        if row['status'] == 'applying':
            p = row['payload']
            current = self.workspace.read(p['arguments']['path'])
            if (current['before_hash'] != p['after_hash'] or current['workspace_identity'] != p['workspace_identity']
                    or current['project_identity'] != p['project_identity']):
                raise AppError('Interrupted file publication has an uncertain outcome. It will not be repeated; inspect the practice file.')
            self.applied(row)
        elif row['status'] != 'applied':
            raise AppError('No completed file action is available for recovery.')
