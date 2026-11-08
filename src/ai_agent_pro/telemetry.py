"""Best-effort execution events, fixed labels and isolated presentation observers."""
import logging
import sqlite3
from time import monotonic
from ai_agent_pro.database import now

STAGES = {'approval': 'Human approval', 'tool': 'Tool execution', 'retrieval': 'Retrieve sources', 'answer': 'Answer', 'draft': 'Draft',
          'review': 'Review', 'save': 'Save result', 'extraction': 'Extract text',
          'chunking': 'Create chunks', 'embeddings': 'Embed chunks', 'publish': 'Publish index', 'task': 'Task'}


class Recorder:
    def __init__(self, repo, *, run_id=None, job_id=None, observer=None):
        self.repo, self.run_id, self.job_id, self.observer = repo, run_id, job_id, observer
        self.active, self.started = None, None

    def emit(self, stage, state, ordinal=0, elapsed=None, completed=None, total=None, outcome=None, category=None):
        event = dict(stage=stage, state=state, ordinal=ordinal, recorded_at=now(), elapsed_seconds=elapsed,
                     completed_units=completed, total_units=total, outcome=outcome, error_category=category)
        try:
            with self.repo.connect() as db:
                db.execute('''INSERT INTO execution_events(run_id,job_id,stage,ordinal,state,recorded_at,
                    elapsed_seconds,completed_units,total_units,outcome,error_category) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (self.run_id, self.job_id, stage, ordinal, state, event['recorded_at'], elapsed, completed, total, outcome, category))
        except sqlite3.Error:
            logging.getLogger('ai_agent_pro').warning('Execution telemetry could not be persisted.')
        if self.observer:
            try:
                self.observer(event)
            except Exception:
                # A broken view must never repeat a request or turn a saved answer into failure.
                logging.getLogger('ai_agent_pro').warning('Progress observer unavailable.')

    def start(self, stage, ordinal=0, total=None):
        self.active, self.started = (stage, ordinal), monotonic()
        self.emit(stage, 'started', ordinal, total=total)

    def complete(self, stage, ordinal=0, completed=None, total=None, outcome=None):
        elapsed = max(0, monotonic() - self.started) if self.active == (stage, ordinal) else None
        self.emit(stage, 'completed', ordinal, elapsed, completed, total, outcome)
        self.active, self.started = None, None

    def fail(self, error):
        stage, ordinal = self.active or ('task', 0)
        elapsed = max(0, monotonic() - self.started) if self.started is not None else None
        self.emit(stage, 'failed', ordinal, elapsed=elapsed, category=error_category(error))
        self.active, self.started = None, None


def events(repo, *, run_id=None, job_id=None):
    column, identifier = ('run_id', run_id) if run_id else ('job_id', job_id)
    with repo.connect() as db:
        return [dict(r) for r in db.execute(f'SELECT * FROM execution_events WHERE {column}=? ORDER BY id LIMIT 500', (identifier,))]


def error_category(message):
    text = (message or '').lower()
    for words, label in [(('credential', 'api_key'), 'Credentials'), (('quota', 'rate limit'), 'Rate or quota'),
                         (('output limit reached',), 'Output limit'),
                         (('deadline', 'timed out'), 'Timeout'), (('budget', 'reservation'), 'Budget'),
                         (('citation',), 'Citations'), (('interrupted', 'cancelled'), 'Interrupted'),
                         (('connect',), 'Connection'), (('source', 'collection'), 'Knowledge sources'),
                         (('pdf', 'extract'), 'Extraction')]:
        if any(word in text for word in words):
            return label
    return 'Other application error'
