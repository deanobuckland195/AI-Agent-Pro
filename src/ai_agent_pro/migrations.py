"""Transactional, forward-only SQLite schema migrations."""
from ai_agent_pro.config import AppError

BASE_SCHEMA = """
            CREATE TABLE IF NOT EXISTS agents (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
              instructions TEXT NOT NULL, settings TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
              title TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
              model TEXT NOT NULL, settings TEXT NOT NULL, requests INTEGER NOT NULL DEFAULT 0,
              input_tokens INTEGER, output_tokens INTEGER, elapsed REAL NOT NULL DEFAULT 0,
              error TEXT, steps TEXT NOT NULL DEFAULT '[]');
            CREATE UNIQUE INDEX IF NOT EXISTS one_running_task
              ON runs(conversation_id) WHERE status = 'running';
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
              created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS message_history ON messages(conversation_id, id);
"""

RAG_SCHEMA = """
CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE agent_collections (
 agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
 collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
 PRIMARY KEY(agent_id, collection_id));
CREATE TABLE documents (
 id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
 filename TEXT NOT NULL, kind TEXT NOT NULL, content_hash TEXT NOT NULL, original BLOB NOT NULL,
 size INTEGER NOT NULL, created_at TEXT NOT NULL, active_generation TEXT,
 UNIQUE(collection_id, content_hash));
CREATE TABLE index_generations (
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 model TEXT NOT NULL, dimensions INTEGER NOT NULL, version TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE document_chunks (
 id TEXT PRIMARY KEY, generation_id TEXT NOT NULL REFERENCES index_generations(id) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL, text TEXT NOT NULL, location TEXT NOT NULL, content_hash TEXT NOT NULL,
 embedding BLOB NOT NULL, UNIQUE(generation_id, ordinal));
CREATE TABLE index_jobs (
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 status TEXT NOT NULL, started_at TEXT NOT NULL, deadline_seconds INTEGER NOT NULL,
 requests INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER, reserved_tokens INTEGER NOT NULL DEFAULT 0,
 error TEXT, warning TEXT);
CREATE UNIQUE INDEX one_index_job ON index_jobs(document_id) WHERE status='running';
CREATE TABLE run_sources (
 run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, label TEXT NOT NULL,
 document_id TEXT REFERENCES documents(id) ON DELETE SET NULL, generation_id TEXT NOT NULL,
 filename TEXT NOT NULL, location TEXT NOT NULL, excerpt TEXT, score REAL NOT NULL,
 PRIMARY KEY(run_id, label));
CREATE TABLE retrieval_usage (
 run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
 requests INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER, reserved_tokens INTEGER NOT NULL DEFAULT 0);
CREATE INDEX chunks_generation ON document_chunks(generation_id);
CREATE INDEX documents_collection ON documents(collection_id);
"""

TELEMETRY_SCHEMA = """
ALTER TABLE index_jobs ADD COLUMN finished_at TEXT;
ALTER TABLE index_jobs ADD COLUMN elapsed REAL;
CREATE TABLE execution_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
 job_id TEXT REFERENCES index_jobs(id) ON DELETE CASCADE,
 stage TEXT NOT NULL, ordinal INTEGER NOT NULL DEFAULT 0,
 state TEXT NOT NULL CHECK(state IN ('started','completed','failed')),
 recorded_at TEXT NOT NULL, elapsed_seconds REAL, completed_units INTEGER, total_units INTEGER,
 outcome TEXT, error_category TEXT,
 CHECK((run_id IS NULL) <> (job_id IS NULL)));
CREATE INDEX run_events ON execution_events(run_id,id);
CREATE INDEX job_events ON execution_events(job_id,id);
"""


def migrate(db):
    # BEGIN IMMEDIATE serializes simultaneous application startups. No executescript:
    # it would commit before the migration and break transactional rollback.
    db.execute("BEGIN IMMEDIATE")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > 5:
        raise AppError("Database schema is newer than this application. Upgrade the application.")
    if version < 1:
        for statement in BASE_SCHEMA.split(";"):
            if statement.strip():
                db.execute(statement)
    if version < 2:
        for statement in RAG_SCHEMA.split(";"):
            if statement.strip():
                db.execute(statement)
        db.execute("""CREATE TRIGGER scrub_deleted_sources BEFORE DELETE ON documents BEGIN
            UPDATE run_sources SET filename='Deleted source', location='', excerpt=NULL
            WHERE document_id=OLD.id; END""")
    if version < 3:
        for statement in TELEMETRY_SCHEMA.split(';'):
            if statement.strip():
                db.execute(statement)
    if version < 4:
        for statement in APPROVAL_SCHEMA.split(';'):
            if statement.strip():
                db.execute(statement)
        db.execute("""CREATE TRIGGER invalidate_approval_context BEFORE DELETE ON documents BEGIN
            UPDATE execution_checkpoints SET state=NULL,phase='invalidated'
            WHERE instr(state,OLD.id)>0; END""")
    if version < 5:
        for statement in WORKFLOW_SCHEMA.split(';'):
            if statement.strip():
                db.execute(statement)
    db.execute("PRAGMA user_version = 5")


APPROVAL_SCHEMA = """
ALTER TABLE runs ADD COLUMN active_since TEXT;
CREATE TABLE approvals (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
 agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
 proposal TEXT NOT NULL, binding TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','approved','denied','applying','applied','expired','cancelled','invalid')),
 created_at TEXT NOT NULL, decided_at TEXT, executed_at TEXT, error TEXT);
CREATE TABLE execution_checkpoints (
 run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE, state TEXT, phase TEXT NOT NULL);
DROP INDEX one_running_task;
CREATE UNIQUE INDEX one_running_task ON runs(conversation_id) WHERE status IN ('running','waiting_approval');
"""


WORKFLOW_SCHEMA = """
CREATE TABLE workflow_definitions (
 id TEXT NOT NULL,version INTEGER NOT NULL,name TEXT NOT NULL,definition TEXT NOT NULL,
 binding TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(id,version));
CREATE TABLE workflow_runs (
 id TEXT PRIMARY KEY,definition_id TEXT NOT NULL,version INTEGER NOT NULL,snapshot TEXT NOT NULL,binding TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('ready','running','waiting_approval','completed','failed','cancelled','interrupted')),
 goal TEXT NOT NULL,cursor INTEGER NOT NULL DEFAULT 0,requests INTEGER NOT NULL DEFAULT 0,
 tokens INTEGER NOT NULL DEFAULT 0,tool_calls INTEGER NOT NULL DEFAULT 0,active_seconds REAL NOT NULL DEFAULT 0,
 owner TEXT,lease_until REAL,cancel_requested INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL,updated_at TEXT NOT NULL,error TEXT,
 FOREIGN KEY(definition_id,version) REFERENCES workflow_definitions(id,version));
CREATE TABLE workflow_steps (
 workflow_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,ordinal INTEGER NOT NULL,
 agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,status TEXT NOT NULL,
 conversation_id TEXT UNIQUE REFERENCES conversations(id) ON DELETE RESTRICT,
 run_id TEXT UNIQUE REFERENCES runs(id) ON DELETE RESTRICT,input TEXT,output TEXT,provenance TEXT NOT NULL DEFAULT '{}',checkpoint_hash TEXT,
 PRIMARY KEY(workflow_id,ordinal));
CREATE TABLE workflow_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,workflow_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL,state TEXT NOT NULL,message TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE INDEX workflow_event_order ON workflow_events(workflow_id,id);
"""
