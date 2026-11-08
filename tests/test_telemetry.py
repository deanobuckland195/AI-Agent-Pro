"""Migration compatibility, truthful progress and observer isolation."""
import asyncio
from dataclasses import replace
import sqlite3
from unittest.mock import AsyncMock
import pytest
from ai_agent_pro.config import AppError
from ai_agent_pro.database import Repository
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.provider import Completion
from ai_agent_pro.telemetry import events
from ai_agent_pro.presentation.progress import workflow_rows
from ai_agent_pro.migrations import BASE_SCHEMA, RAG_SCHEMA
from test_rag_retrieval import kb, indexed


def test_stages_timing_and_observer_isolation(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3)))
    provider = type('Provider', (), {'complete': AsyncMock(return_value=Completion('Answer', 10, 3))})()
    seen = []
    def broken_view(event):
        seen.append(event)
        raise RuntimeError('sk-privateabcdefgh')
    result = asyncio.run(AgentEngine(repo, provider, observer=broken_view).execute(agent.id, conversation, 'Question'))
    assert result == 'Answer' and provider.complete.call_count == 3
    rid = repo.runs(agent.id)[0]['id']
    rows = events(repo, run_id=rid)
    assert [(r['stage'], r['state']) for r in rows] == [
        ('draft', 'started'), ('draft', 'completed'), ('review', 'started'), ('review', 'completed'),
        ('review', 'started'), ('review', 'completed'), ('save', 'started'), ('save', 'completed')]
    assert all(r['elapsed_seconds'] >= 0 for r in rows if r['state'] == 'completed')
    assert len(seen) == len(rows)
    assert 'sk-private' not in str(rows)
    assert repo.runs(agent.id)[0]['status'] == 'completed'


def test_failure_records_safe_category_and_no_success_event(repo, agent, conversation):
    provider = type('Provider', (), {'complete': AsyncMock(side_effect=AppError('OpenAI request timed out.'))})()
    with pytest.raises(AppError):
        asyncio.run(AgentEngine(repo, provider).execute(agent.id, conversation, 'Question'))
    rows = events(repo, run_id=repo.runs(agent.id)[0]['id'])
    assert [(r['stage'], r['state']) for r in rows] == [('answer', 'started'), ('answer', 'failed')]
    assert rows[-1]['error_category'] == 'Timeout'


def test_indexing_progress_matches_real_batches_and_reuse(kb, repo):
    seen = []
    kb.indexer.observer = seen.append
    did = indexed(kb)
    assert any(e['stage'] == 'embeddings' and e['outcome'] == 'batch_completed' and e['completed_units'] == e['total_units'] == 1 for e in seen)
    assert seen[-1]['stage'] == 'publish' and seen[-1]['state'] == 'completed'
    job = kb.repository.jobs(did)[0]
    assert job['finished_at'] and job['elapsed'] >= 0
    assert len(events(repo, job_id=job['id'])) == len(seen)
    seen.clear()
    before = kb.indexer.embeddings.calls
    asyncio.run(kb.indexer.index(did))
    assert kb.indexer.embeddings.calls == before
    assert any(e['outcome'] == 'cached' and e['total_units'] == 0 for e in seen)
    assert not any(e['outcome'] == 'batch_completed' for e in seen)


def test_no_evidence_workflow_has_no_generation_stage(repo, agent, conversation, kb):
    repo.save_agent(replace(agent, settings=replace(agent.settings, rag_enabled=True)))
    provider = type('Provider', (), {'complete': AsyncMock()})()
    asyncio.run(AgentEngine(repo, provider, kb.retrieval).execute(agent.id, conversation, 'Question'))
    rows = events(repo, run_id=repo.runs(agent.id)[0]['id'])
    assert {r['stage'] for r in rows} == {'retrieval', 'save'}
    assert rows[1]['outcome'] == 'no_evidence'
    provider.complete.assert_not_called()


def test_v2_upgrade_preserves_legacy_rows_without_fabricating_telemetry(tmp_path):
    path = tmp_path / 'v2.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(BASE_SCHEMA + RAG_SCHEMA)
        db.execute('PRAGMA user_version=2')
        db.execute('INSERT INTO collections VALUES (?,?,?)', ('legacy', 'Legacy collection', '2026-01-01'))
    repo = Repository(path)
    with repo.connect() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 5
        assert db.execute('SELECT name FROM collections').fetchone()[0] == 'Legacy collection'
        assert db.execute('SELECT COUNT(*) FROM execution_events').fetchone()[0] == 0
    assert workflow_rows([]) == []


def test_started_only_event_remains_started_without_inferred_completion(repo, agent, conversation):
    from ai_agent_pro.telemetry import Recorder
    rid = repo.begin_run(conversation, agent, 'Question')
    Recorder(repo, run_id=rid).start('answer', ordinal=1)
    row = workflow_rows(events(repo, run_id=rid))[0]
    assert row['State'] == 'started'
    assert row['Duration (s)'] is None


def test_clearing_conversation_cascades_telemetry(repo, agent, conversation):
    provider = type('Provider', (), {'complete': AsyncMock(return_value=Completion('Answer', 1, 1))})()
    asyncio.run(AgentEngine(repo, provider).execute(agent.id, conversation, 'Question'))
    rid = repo.runs(agent.id)[0]['id']
    assert events(repo, run_id=rid)
    repo.clear_conversation(conversation)
    assert not events(repo, run_id=rid)


def test_document_deletion_cascades_index_events(kb, repo):
    did = indexed(kb)
    jid = kb.repository.jobs(did)[0]['id']
    assert events(repo, job_id=jid)
    kb.repository.delete_document(did)
    assert not events(repo, job_id=jid)
