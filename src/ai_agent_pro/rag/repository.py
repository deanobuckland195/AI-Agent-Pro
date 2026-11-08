"""Transactional knowledge-base persistence; no network calls inside transactions."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import sqlite3
from uuid import uuid4
from ai_agent_pro.config import AppError
from ai_agent_pro.database import now
from ai_agent_pro.rag.models import MAX_TOTAL_BYTES, MAX_TOTAL_CHUNKS, INDEX_TIMEOUT
from ai_agent_pro.safety import redact, validate_text


class KnowledgeRepository:
    def __init__(self, repository):
        self.repo = repository

    def collections(self):
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM collections ORDER BY name,id')]

    def create_collection(self, name):
        name = redact(validate_text(name, 'Collection name', 80))
        cid = str(uuid4())
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT COUNT(*) FROM collections').fetchone()[0] >= 100:
                raise AppError('The local knowledge base supports at most 100 collections.')
            db.execute('INSERT INTO collections VALUES (?,?,?)', (cid, name, now()))
        return cid

    def assigned(self, agent_id):
        with self.repo.connect() as db:
            return [r[0] for r in db.execute('SELECT collection_id FROM agent_collections WHERE agent_id=? ORDER BY collection_id', (agent_id,))]

    def assign(self, agent_id, collection_ids):
        self.repo.agent(agent_id)
        ids = set(collection_ids)
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not ids <= {r[0] for r in db.execute('SELECT id FROM collections')}:
                raise AppError('Collection no longer exists.')
            if db.execute("SELECT 1 FROM runs r JOIN conversations c ON c.id=r.conversation_id WHERE c.agent_id=? AND r.status='running'", (agent_id,)).fetchone():
                raise AppError('Finish or recover the running task before changing collections.')
            db.execute('DELETE FROM agent_collections WHERE agent_id=?', (agent_id,))
            db.executemany('INSERT INTO agent_collections VALUES (?,?)', [(agent_id, cid) for cid in ids])

    def delete_collection(self, cid):
        with self.repo.connect() as db:
            db.execute('DELETE FROM collections WHERE id=?', (cid,))

    def add_document(self, cid, filename, kind, data):
        did, digest = str(uuid4()), sha256(data).hexdigest()
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM collections WHERE id=?', (cid,)).fetchone():
                raise AppError('Collection no longer exists.')
            existing = db.execute('SELECT id FROM documents WHERE collection_id=? AND content_hash=?', (cid, digest)).fetchone()
            if existing:
                return existing[0]
            if db.execute('SELECT COUNT(*) FROM documents WHERE collection_id=?', (cid,)).fetchone()[0] >= 100:
                raise AppError('A collection supports at most 100 documents.')
            if db.execute('SELECT COALESCE(SUM(size),0) FROM documents').fetchone()[0] + len(data) > MAX_TOTAL_BYTES:
                raise AppError('The knowledge base exceeds its 100 MiB upload quota.')
            db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,NULL)',
                       (did, cid, redact(filename), kind, digest, data, len(data), now()))
        return did

    def documents(self, cid):
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('''SELECT d.id,d.filename,d.size,d.created_at,d.active_generation,
                (SELECT status FROM index_jobs j WHERE j.document_id=d.id ORDER BY j.started_at DESC LIMIT 1) AS last_status,
                (SELECT error FROM index_jobs j WHERE j.document_id=d.id ORDER BY j.started_at DESC LIMIT 1) AS error,
                (SELECT warning FROM index_jobs j WHERE j.document_id=d.id ORDER BY j.started_at DESC LIMIT 1) AS warning
                FROM documents d WHERE collection_id=? ORDER BY created_at DESC''', (cid,))]

    def document(self, did):
        with self.repo.connect() as db:
            row = db.execute('SELECT * FROM documents WHERE id=?', (did,)).fetchone()
        if row is None:
            raise AppError('Document no longer exists.')
        return dict(row)

    def library_documents(self, cid):
        """Library metadata only: never load original upload BLOBs for listing."""
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('''SELECT d.id,d.filename,d.kind,d.size,d.created_at,
                d.active_generation,g.model,g.dimensions,g.version,g.created_at AS indexed_at,
                (SELECT COUNT(*) FROM document_chunks c WHERE c.generation_id=d.active_generation) AS chunks,
                j.status AS last_status,j.error,j.warning,j.started_at AS job_started_at,j.deadline_seconds
                FROM documents d LEFT JOIN index_generations g ON g.id=d.active_generation
                LEFT JOIN index_jobs j ON j.id=(SELECT id FROM index_jobs WHERE document_id=d.id ORDER BY started_at DESC,id DESC LIMIT 1)
                WHERE d.collection_id=? ORDER BY d.created_at DESC,d.id''', (cid,))]

    def jobs(self, did):
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM index_jobs WHERE document_id=? ORDER BY started_at DESC,id DESC LIMIT 20', (did,))]

    def delete_document(self, did):
        # Originals are BLOBs in the same transaction: no orphan files or cleanup races.
        with self.repo.connect() as db:
            db.execute('DELETE FROM documents WHERE id=?', (did,))

    def begin_index(self, did):
        jid = str(uuid4())
        try:
            with self.repo.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                if not db.execute('SELECT 1 FROM documents WHERE id=?', (did,)).fetchone():
                    raise AppError('Document no longer exists.')
                db.execute('INSERT INTO index_jobs(id,document_id,status,started_at,deadline_seconds) VALUES (?,?,\'running\',?,?)',
                           (jid, did, now(), INDEX_TIMEOUT))
        except sqlite3.IntegrityError:
            raise AppError('This document already has a running indexing job.') from None
        return jid

    def charge_index(self, jid, reserve):
        with self.repo.connect() as db:
            if not db.execute("UPDATE index_jobs SET requests=requests+1,reserved_tokens=reserved_tokens+? WHERE id=? AND status='running'", (reserve, jid)).rowcount:
                raise AppError('Indexing was deleted or interrupted.')

    def finish_index(self, jid, chunks, vectors, model, dimensions, version, usage, warning):
        gid = str(uuid4())
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute("SELECT * FROM index_jobs WHERE id=? AND status='running'", (jid,)).fetchone()
            if not job:
                raise AppError('Indexing was deleted or interrupted.')
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(job['started_at'])).total_seconds()
            if age > job['deadline_seconds']:
                raise AppError('Indexing deadline reached.')
            did = job['document_id']
            count = db.execute('''SELECT COUNT(*) FROM document_chunks c JOIN index_generations g
                                ON c.generation_id=g.id WHERE g.document_id<>?''', (did,)).fetchone()[0]
            if count + len(chunks) > MAX_TOTAL_CHUNKS:
                raise AppError('Knowledge base exceeds its 20,000 chunk quota.')
            db.execute('INSERT INTO index_generations VALUES (?,?,?,?,?,?)', (gid, did, model, dimensions, version, now()))
            db.executemany('INSERT INTO document_chunks VALUES (?,?,?,?,?,?,?)',
                           [(str(uuid4()), gid, i, c.text, c.location, c.content_hash, vector)
                            for i, (c, vector) in enumerate(zip(chunks, vectors, strict=True))])
            db.execute('UPDATE documents SET active_generation=? WHERE id=?', (gid, did))
            db.execute('DELETE FROM index_generations WHERE document_id=? AND id<>?', (did, gid))
            db.execute("UPDATE index_jobs SET status='completed',input_tokens=?,warning=?,finished_at=?,elapsed=? WHERE id=?", (usage, warning, now(), age, jid))
        return gid

    def fail_index(self, jid, error):
        with self.repo.connect() as db:
            row = db.execute("SELECT started_at FROM index_jobs WHERE id=? AND status='running'", (jid,)).fetchone()
            if row:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(row['started_at'])).total_seconds()
                db.execute("UPDATE index_jobs SET status='failed',error=?,finished_at=?,elapsed=? WHERE id=? AND status='running'", (redact(error), now(), age, jid))

    def recover_index(self, did):
        with self.repo.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM index_jobs WHERE document_id=? AND status='running'", (did,)).fetchone()
            if not row:
                raise AppError('No running indexing job to recover.')
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(row['started_at'])).total_seconds()
            if age < row['deadline_seconds'] + 10:
                raise AppError('Wait until the indexing deadline plus 10 seconds before recovery.')
            db.execute("UPDATE index_jobs SET status='interrupted',finished_at=?,error='Interrupted indexing; retry explicitly. Usage may be unavailable.' WHERE id=?", (now(), row['id']))

    def candidates(self, agent_id, model, dimensions, version):
        # Authorization precedes vector scoring; callers cannot provide collection IDs.
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute('''SELECT c.*,d.id AS document_id,d.filename,g.id AS generation_id
                FROM document_chunks c JOIN index_generations g ON g.id=c.generation_id
                JOIN documents d ON d.active_generation=g.id
                JOIN agent_collections a ON a.collection_id=d.collection_id
                WHERE a.agent_id=? AND g.model=? AND g.dimensions=? AND g.version=?
                ORDER BY d.id,c.ordinal LIMIT 20001''', (agent_id, model, dimensions, version))]

    def validate_sources(self, db, agent_id, sources):
        for source in sources:
            if not db.execute('''SELECT 1 FROM documents d JOIN agent_collections a ON a.collection_id=d.collection_id
                WHERE d.id=? AND d.active_generation=? AND a.agent_id=?''',
                              (source.document_id, source.generation_id, agent_id)).fetchone():
                raise AppError('Knowledge sources changed during the task. Ask again to retrieve current documents.')

    def require_sources(self, agent_id, sources):
        with self.repo.connect() as db:
            self.validate_sources(db, agent_id, sources)

    def save_sources(self, db, rid, agent_id, sources):
        self.validate_sources(db, agent_id, sources)
        for source in sources:
            s = asdict(source)
            db.execute('INSERT INTO run_sources VALUES (?,?,?,?,?,?,?,?)',
                       (rid, s['label'], s['document_id'], s['generation_id'], s['filename'], s['location'], s['excerpt'], s['score']))

    def sources(self, rid, agent_id):
        with self.repo.connect() as db:
            rows = db.execute('''SELECT s.*,
                EXISTS(SELECT 1 FROM documents d JOIN agent_collections a ON a.collection_id=d.collection_id
                       WHERE d.id=s.document_id AND a.agent_id=?) AS available
                FROM run_sources s JOIN runs r ON r.id=s.run_id JOIN conversations c ON c.id=r.conversation_id
                WHERE s.run_id=? AND c.agent_id=? ORDER BY s.label''', (agent_id, rid, agent_id)).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            if not value['available']:
                value.update(filename='Source unavailable', excerpt=None, location='')
            result.append(value)
        return result

    def charge_query(self, rid, reserve):
        with self.repo.connect() as db:
            db.execute('INSERT INTO retrieval_usage(run_id,requests,reserved_tokens) VALUES (?,1,?)', (rid, reserve))

    def query_usage(self, rid, tokens):
        with self.repo.connect() as db:
            db.execute('UPDATE retrieval_usage SET input_tokens=? WHERE run_id=?', (tokens, rid))

    def usage(self, rid):
        with self.repo.connect() as db:
            row = db.execute('SELECT * FROM retrieval_usage WHERE run_id=?', (rid,)).fetchone()
        return dict(row) if row else None
