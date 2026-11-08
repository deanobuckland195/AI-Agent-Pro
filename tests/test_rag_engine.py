"""RAG engine integration, budgets, citations and changing source permissions."""
import asyncio
from dataclasses import replace
import json
from unittest.mock import AsyncMock
import pytest
from ai_agent_pro.config import AppError
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.provider import Completion
from ai_agent_pro.rag.context import NO_EVIDENCE
from test_rag_retrieval import kb, indexed


def enable(repo, agent, **settings):
    changed = replace(agent, settings=replace(agent.settings, rag_enabled=True, **settings))
    repo.save_agent(changed)
    return changed


def provider(text='The deployment region is London [S1].'):
    class Chat:
        complete = AsyncMock(return_value=Completion(text, 100, 20))
    return Chat()


def run(repo, agent, conversation, kb, chat, prompt='Where is the deployment?'):
    return asyncio.run(AgentEngine(repo, chat, kb.retrieval).execute(agent.id, conversation, prompt))


def test_rag_context_and_citations_persist_through_reviews(repo, agent, conversation, kb):
    did = indexed(kb)
    agent = enable(repo, agent, max_steps=3)
    chat = provider()
    assert '[S1]' in run(repo, agent, conversation, kb, chat)
    assert chat.complete.call_count == 3
    for call in chat.complete.call_args_list:
        assert 'untrusted data' in call.kwargs['instructions']
        assert 'Document-grounded mode' in call.kwargs['instructions']
        assert 'tools' not in call.kwargs
        payload = json.loads(call.kwargs['messages'][0]['content'])
        assert payload['untrusted_document_sources'][0]['label'] == 'S1'
        assert payload['untrusted_document_sources'][0]['excerpt'].startswith('The deployment')
    rid = repo.runs(agent.id)[0]['id']
    sources = kb.repository.sources(rid, agent.id)
    assert sources[0]['document_id'] == did
    assert sources[0]['available']
    assert kb.repository.usage(rid)['requests'] == 1
    assert repo.runs(agent.id)[0]['requests'] == 3


def test_empty_or_irrelevant_knowledge_answers_without_chat_call(repo, agent, conversation, kb):
    enable(repo, agent)
    chat = provider()
    assert run(repo, agent, conversation, kb, chat) == NO_EVIDENCE
    chat.complete.assert_not_called()
    assert repo.runs(agent.id)[0]['status'] == 'completed'
    indexed(kb, 'The database is Postgres.')
    assert run(repo, agent, conversation, kb, chat, 'Where is London deployment?') == NO_EVIDENCE
    chat.complete.assert_not_called()


@pytest.mark.parametrize('text', ['Unsupported [S99]', 'London without citation', 'Unsupported [s1]'])
def test_bad_citations_fail_without_saving_answer(repo, agent, conversation, kb, text):
    indexed(kb)
    enable(repo, agent)
    with pytest.raises(AppError, match='citation'):
        run(repo, agent, conversation, kb, provider(text))
    assert repo.runs(agent.id)[0]['status'] == 'failed'
    assert [m['role'] for m in repo.messages(conversation)] == ['user']
    assert kb.repository.sources(repo.runs(agent.id)[0]['id'], agent.id) == []


def test_explicit_insufficient_answer_allowed(repo, agent, conversation, kb):
    indexed(kb)
    enable(repo, agent)
    assert run(repo, agent, conversation, kb, provider(NO_EVIDENCE)) == NO_EVIDENCE


def test_delete_scrubs_citations_and_excludes_old_answer_context(repo, agent, conversation, kb):
    did = indexed(kb)
    enable(repo, agent)
    chat = provider()
    run(repo, agent, conversation, kb, chat)
    rid = repo.runs(agent.id)[0]['id']
    kb.repository.delete_document(did)
    source = kb.repository.sources(rid, agent.id)[0]
    assert source['excerpt'] is None and not source['available']
    with repo.connect() as db:
        assert db.execute('SELECT excerpt FROM run_sources WHERE run_id=?', (rid,)).fetchone()[0] is None
    assert run(repo, agent, conversation, kb, chat) == NO_EVIDENCE
    assert chat.complete.call_count == 1
    indexed(kb, 'The deployment region is Manchester.')
    chat.complete.return_value = Completion('Manchester [S1]', 30, 10)
    run(repo, agent, conversation, kb, chat)
    sent = str(chat.complete.call_args.kwargs['messages'])
    assert 'London' not in sent and 'Manchester' in sent


def test_deletion_during_generation_rolls_back_answer_and_sources(repo, agent, conversation, kb):
    did = indexed(kb)
    enable(repo, agent)
    chat = provider()
    async def deleting(**kwargs):
        kb.repository.delete_document(did)
        return Completion('London [S1]', 30, 10)
    chat.complete.side_effect = deleting
    with pytest.raises(AppError, match='sources changed'):
        run(repo, agent, conversation, kb, chat)
    assert repo.runs(agent.id)[0]['status'] == 'failed'
    assert len(repo.messages(conversation)) == 1


def test_reindex_preserves_historical_citation_snapshot(repo, agent, conversation, kb):
    did = indexed(kb)
    enable(repo, agent)
    run(repo, agent, conversation, kb, provider())
    rid = repo.runs(agent.id)[0]['id']
    original = kb.repository.sources(rid, agent.id)[0]
    asyncio.run(kb.indexer.index(did))
    after = kb.repository.sources(rid, agent.id)[0]
    assert after['generation_id'] == original['generation_id']
    assert after['excerpt'] == original['excerpt']
    assert after['generation_id'] != kb.repository.document(did)['active_generation']


def test_reindex_during_generation_rejects_stale_answer(repo, agent, conversation, kb):
    did = indexed(kb)
    enable(repo, agent)
    chat = provider()
    async def reindexing(**kwargs):
        await kb.indexer.index(did)
        return Completion('London [S1]', 30, 10)
    chat.complete.side_effect = reindexing
    with pytest.raises(AppError, match='sources changed'):
        run(repo, agent, conversation, kb, chat)
    assert repo.runs(agent.id)[0]['status'] == 'failed'
    assert len(repo.messages(conversation)) == 1


def test_clear_and_agent_delete_preserve_shared_knowledge(repo, agent, conversation, kb):
    did = indexed(kb)
    enable(repo, agent)
    run(repo, agent, conversation, kb, provider())
    rid = repo.runs(agent.id)[0]['id']
    repo.clear_conversation(conversation)
    assert kb.repository.sources(rid, agent.id) == []
    assert kb.repository.usage(rid) is None
    assert kb.repository.document(did)['active_generation']
    repo.delete_agent(agent.id)
    assert kb.repository.assigned(agent.id) == []
    assert kb.repository.document(did)['active_generation']


def test_revoked_collection_hides_source_and_is_not_reused(repo, agent, conversation, kb):
    indexed(kb)
    enable(repo, agent)
    run(repo, agent, conversation, kb, provider())
    rid = repo.runs(agent.id)[0]['id']
    kb.repository.assign(agent.id, [])
    assert kb.repository.sources(rid, agent.id)[0]['excerpt'] is None
    chat = provider()
    with pytest.raises(AppError, match='Assign'):
        run(repo, agent, conversation, kb, chat)
    chat.complete.assert_not_called()


def test_embedding_failures_are_persisted_safely(repo, agent, conversation, kb):
    indexed(kb)
    enable(repo, agent)
    kb.retrieval.embeddings.embed = AsyncMock(side_effect=RuntimeError('sk-privateabcdefgh'))
    with pytest.raises(AppError, match='Unexpected'):
        run(repo, agent, conversation, kb, provider())
    record = repo.runs(agent.id)[0]
    assert record['status'] == 'failed' and 'sk-private' not in record['error']
    assert kb.repository.usage(record['id'])['input_tokens'] is None


def test_retrieved_content_included_in_task_budget(repo, agent, conversation, kb):
    indexed(kb)
    enable(repo, agent, token_budget=1024)
    chat = provider()
    with pytest.raises(AppError, match='budget'):
        run(repo, agent, conversation, kb, chat)
    chat.complete.assert_not_called()


def test_query_embedding_is_within_overall_deadline(repo, agent, conversation, kb, monkeypatch):
    indexed(kb)
    enable(repo, agent)
    cancelled = []
    async def slow(*args, **kwargs):
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)
    kb.retrieval.embeddings.embed = slow
    import ai_agent_pro.engine as engine
    values = iter([0, 59.99])
    monkeypatch.setattr(engine, 'monotonic', lambda: next(values, 60.1))
    with pytest.raises(AppError, match='deadline'):
        run(repo, agent, conversation, kb, provider())
    assert cancelled
    assert repo.runs(agent.id)[0]['status'] == 'failed'


def test_enabled_rag_without_service_fails_closed(repo, agent, conversation):
    enable(repo, agent)
    chat = provider()
    with pytest.raises(AppError, match='not configured'):
        asyncio.run(AgentEngine(repo, chat).execute(agent.id, conversation, 'Question'))
    chat.complete.assert_not_called()


def test_invalid_document_mode_setting(repo, agent):
    with pytest.raises(AppError, match='Document mode'):
        repo.save_agent(replace(agent, settings=replace(agent.settings, rag_enabled='yes')))
