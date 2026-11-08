"""SQLite repository. Connections are short-lived and all values are parameterized."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sqlite3
from uuid import uuid4
from ai_agent_pro.config import Agent, AppError, Settings
from ai_agent_pro.safety import redact, validate_text


def now():
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise AppError('Database path must not be a symbolic link.')
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        with self.connect() as db:
            from ai_agent_pro.migrations import migrate
            migrate(db)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_agent(self, agent):
        agent.validate()
        with self.connect() as db:
            db.execute("""INSERT INTO agents VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, description=excluded.description,
                instructions=excluded.instructions, settings=excluded.settings""",
                       (agent.id, redact(agent.name.strip()), redact(agent.description.strip()),
                        redact(agent.instructions.strip()), json.dumps(agent.settings.to_dict())))

    def agents(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM agents ORDER BY name, id").fetchall()
        return [Agent(r['id'], r['name'], r['description'], r['instructions'],
                      Settings(**json.loads(r['settings']))).validate() for r in rows]

    def agent(self, agent_id):
        for agent in self.agents():
            if agent.id == agent_id:
                return agent
        raise AppError("Agent no longer exists.")

    def delete_agent(self, agent_id):
        with self.connect() as db:
            if db.execute('SELECT 1 FROM workflow_steps WHERE agent_id=?',(agent_id,)).fetchone():
                raise AppError('This agent is referenced by retained workflow executions.')
            if db.execute("""SELECT 1 FROM runs r JOIN conversations c ON c.id=r.conversation_id
                          WHERE c.agent_id=? AND r.status IN ('running','waiting_approval')""", (agent_id,)).fetchone():
                raise AppError("Finish or recover the running task first.")
            db.execute("DELETE FROM agents WHERE id=?", (agent_id,))

    def new_conversation(self, agent_id, title="New conversation"):
        title = validate_text(title, "Title", 100)
        self.agent(agent_id)
        cid = str(uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, agent_id, redact(title), now()))
        return cid

    def conversations(self, agent_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM conversations WHERE agent_id=? ORDER BY created_at DESC", (agent_id,))]

    def require_conversation(self, cid, agent_id):
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM conversations WHERE id=? AND agent_id=?", (cid, agent_id)).fetchone()
        if not row:
            raise AppError("Conversation does not belong to the selected agent.")

    def messages(self, cid, completed_only=False):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT m.* FROM messages m JOIN runs r ON r.id=m.run_id
                WHERE m.conversation_id=? AND (?=0 OR r.status='completed') ORDER BY m.id""",
                                              (cid, int(completed_only)))]

    def begin_run(self, cid, agent, prompt, *, workflow_identity=None):
        self.require_conversation(cid, agent.id)
        rid = str(uuid4())
        try:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                managed = db.execute('''SELECT s.workflow_id,s.ordinal,s.status,w.owner,w.status AS workflow_status,w.cancel_requested
                    FROM workflow_steps s JOIN workflow_runs w ON w.id=s.workflow_id WHERE s.conversation_id=?''', (cid,)).fetchone()
                if managed:
                    expected = (managed['workflow_id'],managed['ordinal'],managed['owner'])
                    if workflow_identity != expected or managed['workflow_status'] != 'running' or managed['status'] != 'running' or managed['cancel_requested']:
                        raise AppError('Workflow conversations must execute through their owning coordinator.')
                elif workflow_identity is not None:
                    raise AppError('Workflow step identity is not bound to this conversation.')
                db.execute("""INSERT INTO runs
                    (id,conversation_id,status,started_at,model,settings) VALUES (?,?,'running',?,?,?)""",
                           (rid, cid, now(), agent.settings.model, json.dumps(agent.settings.to_dict())))
                if managed:
                    if db.execute('UPDATE workflow_steps SET run_id=? WHERE workflow_id=? AND ordinal=? AND run_id IS NULL',
                        (rid,managed['workflow_id'],managed['ordinal'])).rowcount != 1:
                        raise AppError('Workflow child was already started; it will not be repeated.')
                db.execute("INSERT INTO messages(conversation_id,run_id,role,content,created_at) VALUES (?,?,'user',?,?)",
                           (cid, rid, redact(prompt), now()))
                db.execute("UPDATE conversations SET title=? WHERE id=? AND title='New conversation'",
                           (redact(prompt[:70]), cid))
        except sqlite3.IntegrityError:
            raise AppError("This conversation already has a running task.") from None
        return rid

    def finish_run(self, rid, status, requests, input_tokens, output_tokens, elapsed, steps, error=None, answer=None, sources=()):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM runs WHERE id=? AND status='running'", (rid,)).fetchone()
            if not row:
                raise AppError("Task was interrupted or already finalized.")
            if sources:
                from ai_agent_pro.rag.repository import KnowledgeRepository
                agent_id = db.execute('SELECT agent_id FROM conversations WHERE id=?', (row['conversation_id'],)).fetchone()[0]
                KnowledgeRepository(self).save_sources(db, rid, agent_id, sources)
            db.execute("""UPDATE runs SET status=?,finished_at=?,requests=?,input_tokens=?,output_tokens=?,
                       elapsed=?,steps=?,error=? WHERE id=?""",
                       (status, now(), requests, input_tokens, output_tokens, elapsed,
                        json.dumps(steps), redact(error) if error else None, rid))
            if answer is not None:
                db.execute("INSERT INTO messages(conversation_id,run_id,role,content,created_at) VALUES (?,?,'assistant',?,?)",
                           (row['conversation_id'], rid, redact(answer), now()))

    def runs(self, agent_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT r.*, c.title FROM runs r
                JOIN conversations c ON c.id=r.conversation_id WHERE c.agent_id=?
                ORDER BY r.started_at DESC""", (agent_id,))]

    def clear_conversation(self, cid):
        with self.connect() as db:
            if db.execute('SELECT 1 FROM workflow_steps WHERE conversation_id=?',(cid,)).fetchone():
                raise AppError('This conversation is a retained workflow checkpoint and cannot be cleared independently.')
            if db.execute("SELECT 1 FROM runs WHERE conversation_id=? AND status IN ('running','waiting_approval')", (cid,)).fetchone():
                raise AppError("Finish or recover the running task first.")
            db.execute("DELETE FROM runs WHERE conversation_id=?", (cid,))
            db.execute("UPDATE conversations SET title='New conversation' WHERE id=?", (cid,))

    def recover_run(self, rid):
        # Explicit recovery only after the task deadline; never assume a second UI session is dead.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=? AND status='running'", (rid,)).fetchone()
            if not row:
                raise AppError("No running task to recover.")
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(row['active_since'] or row['started_at'])).total_seconds()
            if age < json.loads(row['settings'])['timeout_seconds'] + 10:
                raise AppError("Wait until the task deadline plus 10 seconds before recovery.")
            checkpoint = db.execute('SELECT phase FROM execution_checkpoints WHERE run_id=?', (rid,)).fetchone()
            if checkpoint and checkpoint['phase'] in ('ready', 'applying', 'applied'):
                db.execute("UPDATE runs SET status='waiting_approval',error=NULL WHERE id=?", (rid,))
                return
            db.execute("UPDATE runs SET status='interrupted',finished_at=?,error=? WHERE id=?",
                       (now(), "Recovered after interrupted application session; usage may be unavailable.", rid))
