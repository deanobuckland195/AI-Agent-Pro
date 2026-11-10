"""Immutable definitions, atomic coordinator claims and durable step checkpoints."""
from dataclasses import asdict, replace
import json
from time import time
from uuid import uuid4

from ai_agent_pro.config import AppError
from ai_agent_pro.database import now
from ai_agent_pro.practice import digest
from ai_agent_pro.safety import redact, validate_text
from ai_agent_pro.approvals import encode
from ai_agent_pro.workflow_contracts import definition, Limits, six_stage


class WorkflowStore:
    def __init__(self, repo):
        self.repo = repo

    def save(self, value, workflow_id=None, version=1):
        definition(value)
        if type(version) is not int or not 1 <= version <= 1000:
            raise AppError('Workflow versions must be positive bounded integers.')
        for step in value['steps']:
            self.repo.agent(step['agent_id'])
        raw = encode(value)
        if redact(raw) != raw:
            raise AppError('Do not store credentials in workflow definitions.')
        identifier = workflow_id or str(uuid4())
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM workflow_definitions WHERE id=? AND version=?',(identifier,version)).fetchone():
                raise AppError('Workflow versions are immutable; save a new version.')
            db.execute('INSERT INTO workflow_definitions VALUES (?,?,?,?,?,?)',
                (identifier,version,value['name'],raw,digest(raw.encode()),now()))
        return identifier,version

    def definitions(self):
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM workflow_definitions ORDER BY created_at DESC,id,version DESC')]

    def start(self, identifier, version, goal):
        goal = redact(validate_text(goal,'Workflow goal',2000))
        with self.repo.connect() as db:
            row = db.execute('SELECT * FROM workflow_definitions WHERE id=? AND version=?',(identifier,version)).fetchone()
        if not row or digest(row['definition'].encode()) != row['binding']:
            raise AppError('Workflow definition is unavailable or changed.')
        value = definition(json.loads(row['definition']))
        agents = [self.repo.agent(s['agent_id']) for s in value['steps']]
        snapshots = [asdict(replace(a,settings=replace(a.settings,history_messages=0,
            max_output_tokens=min(a.settings.max_output_tokens,1200) if six_stage(value) else a.settings.max_output_tokens))) for a in agents]
        raw = encode(dict(definition=value,agents=snapshots,goal=goal))
        wid = str(uuid4())
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('''INSERT INTO workflow_runs(id,definition_id,version,snapshot,binding,status,goal,created_at,updated_at)
                VALUES (?,?,?,?,?,'ready',?,?,?)''',(wid,identifier,version,raw,digest(raw.encode()),goal,now(),now()))
            for i,agent in enumerate(agents):
                db.execute("INSERT INTO workflow_steps(workflow_id,ordinal,agent_id,status) VALUES (?,?,?,'pending')",(wid,i,agent.id))
            self.event(db,wid,0,'ready','Workflow created; no permissions restored or inherited')
        return wid

    @staticmethod
    def event(db, wid, ordinal, state, message):
        db.execute('INSERT INTO workflow_events(workflow_id,ordinal,state,message,created_at) VALUES (?,?,?,?,?)',
                   (wid,ordinal,state,message,now()))

    def get(self, wid):
        with self.repo.connect() as db:
            row = db.execute('SELECT * FROM workflow_runs WHERE id=?',(wid,)).fetchone()
        if not row or digest(row['snapshot'].encode()) != row['binding']:
            raise AppError('Workflow snapshot is missing or corrupted.')
        result = dict(row)
        result['data'] = json.loads(row['snapshot'])
        if result['data'].get('goal') != result['goal']:
            raise AppError('Workflow goal binding is corrupted.')
        definition(result['data']['definition'])
        return result

    def runs(self):
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM workflow_runs ORDER BY created_at DESC LIMIT 100')]

    def steps(self, wid):
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM workflow_steps WHERE workflow_id=? ORDER BY ordinal',(wid,))]

    def claim(self, wid, approval_run_id=None):
        value = self.get(wid)
        limits = Limits(**value['data']['definition']['limits']).validate()
        owner = str(uuid4())
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workflow_runs WHERE id=?',(wid,)).fetchone()
            if row['status'] not in ('ready','waiting_approval') or row['cancel_requested']:
                raise AppError('Workflow is already claimed or cannot continue. Recover an interrupted execution explicitly.')
            if approval_run_id is not None:
                step = db.execute('SELECT run_id FROM workflow_steps WHERE workflow_id=? AND ordinal=?',(wid,row['cursor'])).fetchone()
                if row['status'] != 'waiting_approval' or not step or step['run_id'] != approval_run_id:
                    raise AppError('Approval does not belong to the current paused workflow step.')
            elif row['status'] == 'waiting_approval':
                raise AppError('A paused workflow requires its own explicit action approval.')
            if row['active_seconds'] >= limits.timeout_seconds:
                raise AppError('Workflow execution time is exhausted.')
            db.execute("UPDATE workflow_runs SET status='running',owner=?,lease_until=?,updated_at=?,error=NULL WHERE id=?",
                (owner,time()+limits.timeout_seconds-row['active_seconds']+10,now(),wid))
        return owner, self.get(wid)

    def end(self,wid,owner,status,elapsed,error=None):
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute('''UPDATE workflow_runs SET status=?,active_seconds=MAX(active_seconds,?),
                owner=NULL,lease_until=NULL,error=?,updated_at=? WHERE id=? AND owner=?''',
                (status,elapsed,redact(error) if error else None,now(),wid,owner)).rowcount
            if changed != 1:
                raise AppError('Workflow execution ownership changed; no work was repeated.')
            if status in ('failed','cancelled','interrupted'):
                for step in db.execute('SELECT run_id FROM workflow_steps WHERE workflow_id=?',(wid,)).fetchall():
                    if step['run_id']:
                        db.execute("UPDATE approvals SET status='cancelled',error=? WHERE run_id=? AND status IN ('pending','approved')",
                            ('Owning workflow finalized; this action cannot execute.',step['run_id']))
                        db.execute("UPDATE runs SET status='interrupted',finished_at=?,error=? WHERE id=? AND status='waiting_approval'",
                            (now(),'Owning workflow finalized before this action executed.',step['run_id']))
                        db.execute("UPDATE execution_checkpoints SET state=NULL,phase='finished' WHERE run_id=? AND phase IN ('ready','applied')",(step['run_id'],))
            self.event(db,wid,0,status,redact(error) if error else status)

    def cancel(self,wid):
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workflow_runs WHERE id=?',(wid,)).fetchone()
            if not row or row['status'] in ('completed','failed','cancelled','interrupted'):
                raise AppError('Workflow is already finalized.')
            db.execute('UPDATE workflow_runs SET cancel_requested=1 WHERE id=?',(wid,))
            if row['status'] != 'running':
                # Revoke pending file actions atomically with coordinator cancellation.
                for step in db.execute('SELECT run_id FROM workflow_steps WHERE workflow_id=?',(wid,)).fetchall():
                    if step['run_id']:
                        db.execute("UPDATE approvals SET status='cancelled',decided_at=?,error='Workflow cancelled; no new action is allowed.' WHERE run_id=? AND status IN ('pending','approved')",(now(),step['run_id']))
                        db.execute("UPDATE runs SET status='interrupted',finished_at=?,error='Owning workflow cancelled.' WHERE id=? AND status='waiting_approval'",(now(),step['run_id']))
                        db.execute("UPDATE execution_checkpoints SET phase='finished',state=NULL WHERE run_id=?",(step['run_id'],))
                db.execute("UPDATE workflow_runs SET status='cancelled',updated_at=? WHERE id=?",(now(),wid))
                db.execute("UPDATE workflow_steps SET status='cancelled' WHERE workflow_id=? AND ordinal=? AND status!='completed'",(wid,row['cursor']))
            self.event(db,wid,row['cursor'],'cancel_requested','Cancellation requested; completed actions are retained')

    def recover(self,wid):
        value = self.get(wid)
        if value['status'] != 'running' or value['lease_until'] is None or value['lease_until'] > time():
            raise AppError('Wait for the workflow lease deadline before explicit recovery.')
        all_steps = self.steps(wid)
        if value['cursor'] >= len(all_steps):
            self.end(wid,value['owner'],'completed',value['active_seconds'])
            return
        step = all_steps[value['cursor']]
        if step['run_id']:
            with self.repo.connect() as db:
                child = db.execute('SELECT status FROM runs WHERE id=?',(step['run_id'],)).fetchone()
            if child['status'] == 'running':
                self.repo.recover_run(step['run_id'])
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            fresh = db.execute('SELECT * FROM workflow_runs WHERE id=?',(wid,)).fetchone()
            if fresh['status'] != 'running' or fresh['lease_until'] > time() or fresh['owner'] != value['owner']:
                raise AppError('Workflow lease changed during recovery.')
            child = db.execute('SELECT status FROM runs WHERE id=?',(step['run_id'],)).fetchone() if step['run_id'] else None
            status = 'ready' if child is None or child['status'] == 'completed' else 'waiting_approval' if child['status'] == 'waiting_approval' else 'interrupted'
            if fresh['cancel_requested']:
                status = 'cancelled'
            db.execute('UPDATE workflow_runs SET status=?,owner=NULL,lease_until=NULL,updated_at=? WHERE id=?',(status,now(),wid))
            if child is None:
                db.execute("UPDATE workflow_steps SET status='pending' WHERE workflow_id=? AND ordinal=?",(wid,step['ordinal']))
            self.event(db,wid,step['ordinal'],status,'Recovered without repeating uncertain requests or completed actions')

    def sync(self,wid):
        value = self.get(wid)
        if value['status'] == 'waiting_approval':
            step = self.steps(wid)[value['cursor']]
            with self.repo.connect() as db:
                child = db.execute('SELECT status,error FROM runs WHERE id=?',(step['run_id'],)).fetchone()
                if child and child['status'] not in ('waiting_approval','running'):
                    db.execute("UPDATE workflow_runs SET status='failed',error=?,updated_at=? WHERE id=? AND status='waiting_approval'",
                        (child['error'] or 'Workflow child was finalized outside its coordinator.',now(),wid))
                    db.execute("UPDATE workflow_steps SET status='failed' WHERE workflow_id=? AND ordinal=? AND status!='completed'",(wid,value['cursor']))
        return self.get(wid)
