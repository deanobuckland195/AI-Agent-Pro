"""Failure injection, quotas, cancellation and transactional recovery."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from unittest.mock import AsyncMock
import pytest
from ai_agent_pro.config import AppError
from ai_agent_pro.database import Repository
from ai_agent_pro.migrations import BASE_SCHEMA
from ai_agent_pro.rag.chunking import chunk_sections
from ai_agent_pro.rag.extraction import extract
from ai_agent_pro.rag.models import EmbeddingBatch, Section
from ai_agent_pro.rag.validation import validate_upload
from test_rag_retrieval import kb, indexed, retrieve


def test_storage_quota_is_atomic(kb, monkeypatch):
    import ai_agent_pro.rag.repository as module
    monkeypatch.setattr(module, 'MAX_TOTAL_BYTES', 10)
    did = kb.storage.upload(kb.cid, 'a.txt', b'first')
    with pytest.raises(AppError, match='quota'):
        kb.storage.upload(kb.cid, 'b.txt', b'second')
    assert [d['id'] for d in kb.repository.documents(kb.cid)] == [did]


def test_document_limit(kb):
    for i in range(100):
        kb.storage.upload(kb.cid, f'{i}.txt', f'content {i}'.encode())
    with pytest.raises(AppError, match='100 documents'):
        kb.storage.upload(kb.cid, 'extra.txt', b'extra')
    assert len(kb.repository.documents(kb.cid)) == 100


def test_chunk_storage_quota_rolls_back(kb, monkeypatch):
    import ai_agent_pro.rag.repository as module
    monkeypatch.setattr(module, 'MAX_TOTAL_CHUNKS', 0)
    did = kb.storage.upload(kb.cid, 'notes.txt', b'London deployment')
    with pytest.raises(AppError, match='chunk quota'):
        asyncio.run(kb.indexer.index(did))
    assert kb.repository.document(did)['active_generation'] is None
    with kb.repository.repo.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM index_generations').fetchone()[0] == 0


def test_corrupted_vectors_fail_closed_and_reindex_repairs(kb, repo, agent):
    did = indexed(kb)
    with repo.connect() as db:
        db.execute("UPDATE document_chunks SET embedding=?", (b'broken',))
    with pytest.raises(AppError, match='embeddings are invalid'):
        retrieve(kb, repo, agent, 'deployment')
    before = kb.indexer.embeddings.calls
    asyncio.run(kb.indexer.index(did))
    assert kb.indexer.embeddings.calls == before + 1
    (sources, _), _ = retrieve(kb, repo, agent, 'deployment')
    assert sources[0].document_id == did


def test_later_embedding_batch_failure_leaves_no_partial_index(kb):
    text = '\n'.join(f'{i:04d} London deployment ' + 'x' * 560 for i in range(45))
    did = kb.storage.upload(kb.cid, 'large.txt', text.encode())
    original = kb.indexer.embeddings.embed
    calls = []
    async def failing(texts, *, timeout):
        calls.append(texts)
        if len(calls) == 2:
            raise AppError('Embedding quota exhausted.')
        return await original(texts, timeout=timeout)
    kb.indexer.embeddings.embed = failing
    with pytest.raises(AppError, match='quota'):
        asyncio.run(kb.indexer.index(did))
    assert len(calls) == 2
    assert kb.repository.document(did)['active_generation'] is None
    with kb.repository.repo.connect() as db:
        job = db.execute('SELECT * FROM index_jobs').fetchone()
        assert job['requests'] == 2 and job['input_tokens'] is None
        assert db.execute('SELECT COUNT(*) FROM document_chunks').fetchone()[0] == 0


def test_cancelled_index_is_recorded_and_recoverable(kb):
    did = kb.storage.upload(kb.cid, 'notes.txt', b'London deployment')
    kb.indexer.embeddings.embed = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(kb.indexer.index(did))
    assert kb.repository.documents(kb.cid)[0]['last_status'] == 'failed'
    kb.repository.begin_index(did)


def test_indexing_deadline_prevents_request(kb, monkeypatch):
    import ai_agent_pro.rag.indexing as module
    values = iter([0, 181])
    monkeypatch.setattr(module, 'monotonic', lambda: next(values, 181))
    did = kb.storage.upload(kb.cid, 'notes.txt', b'London deployment')
    with pytest.raises(AppError, match='deadline'):
        asyncio.run(kb.indexer.index(did))
    assert kb.indexer.embeddings.calls == 0


def test_unknown_usage_is_explicit(kb, repo, agent):
    async def unknown(texts, *, timeout):
        return EmbeddingBatch([[1, 0, 0, 0, 0] for t in texts], None)
    kb.indexer.embeddings.embed = unknown
    indexed(kb)
    (_, spent), rid = retrieve(kb, repo, agent, 'deployment')
    assert spent == len('deployment')
    assert kb.repository.usage(rid)['input_tokens'] is None
    with repo.connect() as db:
        assert db.execute('SELECT input_tokens FROM index_jobs').fetchone()[0] is None


def test_overreported_embedding_usage_rejected(kb):
    did = kb.storage.upload(kb.cid, 'notes.txt', b'London deployment')
    kb.indexer.embeddings.embed = AsyncMock(return_value=EmbeddingBatch([[1, 0, 0, 0, 0]], 1000000))
    with pytest.raises(AppError, match='reservation'):
        asyncio.run(kb.indexer.index(did))


def test_oversized_query_never_calls_provider(kb, repo, agent, conversation):
    indexed(kb)
    before = kb.indexer.embeddings.calls
    rid = repo.begin_run(conversation, agent, 'question')
    with pytest.raises(AppError, match='retrieval input'):
        asyncio.run(kb.retrieval.retrieve(agent.id, 'x' * 6001, rid, timeout=10, budget=40000))
    assert kb.indexer.embeddings.calls == before


def test_extraction_and_chunk_limits(monkeypatch):
    with pytest.raises(AppError, match='text limit'):
        extract('txt', b'a' * 1000001)
    import ai_agent_pro.rag.chunking as module
    monkeypatch.setattr(module, 'MAX_CHUNKS', 1)
    with pytest.raises(AppError, match='chunk limit'):
        chunk_sections([Section('x' * 100)], size=32, overlap=0)


@pytest.mark.parametrize('name', ['bad\nname.txt', 'bad\tname.txt', 'bad\x7fname.txt'])
def test_filename_controls_rejected(name):
    with pytest.raises(AppError, match='control'):
        validate_upload(name, b'hello')


def test_migration_failure_rolls_back_all_new_tables(tmp_path, monkeypatch):
    import ai_agent_pro.migrations as module
    path = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(BASE_SCHEMA)
        db.execute('PRAGMA user_version=1')
    monkeypatch.setattr(module, 'RAG_SCHEMA', module.RAG_SCHEMA + '; INVALID SQL;')
    with pytest.raises(sqlite3.Error):
        Repository(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='collections'").fetchone()


def test_concurrent_startups_and_private_database_permissions(tmp_path):
    path = tmp_path / 'parallel.sqlite3'
    with ThreadPoolExecutor(max_workers=3) as pool:
        repositories = list(pool.map(lambda _: Repository(path), range(3)))
    assert len(repositories) == 3
    assert path.stat().st_mode & 0o777 == 0o600
    with repositories[0].connect() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 5


def test_database_symlink_rejected(tmp_path):
    target = tmp_path / 'target.sqlite3'
    target.touch()
    link = tmp_path / 'link.sqlite3'
    link.symlink_to(target)
    with pytest.raises(AppError, match='symbolic link'):
        Repository(link)


def test_invalid_pdf_job_safe_failure(kb):
    did = kb.storage.upload(kb.cid, 'broken.pdf', b'%PDF-broken')
    with pytest.raises(AppError, match='PDF'):
        asyncio.run(kb.indexer.index(did))
    assert kb.repository.documents(kb.cid)[0]['last_status'] == 'failed'
