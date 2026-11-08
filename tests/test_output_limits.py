"""Output-limit termination is diagnosed, bounded and never committed as an answer."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_agent_pro.analytics import Analytics, Filters
from ai_agent_pro.config import Settings, AppError, MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.execution import ChatJob
from ai_agent_pro.provider import Completion, ResponseStatusError
from ai_agent_pro.telemetry import events
from test_provider import provider_with, response_body
from test_streaming import SSE, stream_events, sdk_provider
from test_ui import launch, button, finish_chat
from test_rag_retrieval import kb, indexed
from test_rag_engine import enable


def incomplete_response(reason='max_output_tokens', usage=True):
    body = response_body(status='incomplete', text='Unfinished code: ```python\n', usage=usage)
    body['incomplete_details'] = {'reason': reason} if reason is not None else None
    if usage:
        body['usage'].update(output_tokens=1200, total_tokens=1220)
    return body


def limit_provider(streaming, reason='max_output_tokens', usage=True):
    if streaming:
        data = stream_events(chunks=('Unfinished code: ', '```python\n'), terminal='incomplete')
        data[-1]['response'] = incomplete_response(reason, usage)
        return sdk_provider(SSE(data))
    return provider_with(lambda request: httpx.Response(200, json=incomplete_response(reason, usage)))


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('reason', ['max_output_tokens', 'content_filter', None, 'sk-privateabcdefgh'])
def test_real_sdk_preserves_safe_incomplete_diagnostics(streaming, reason):
    seen = []
    kwargs = {'on_delta': seen.append} if streaming else {}
    with pytest.raises(ResponseStatusError) as caught:
        asyncio.run(limit_provider(streaming, reason).complete(model='gpt-4.1-mini', instructions='Policy',
            messages=[{'role': 'user', 'content': 'Ten detailed steps with code'}],
            max_output_tokens=1200, timeout=2, **kwargs))
    error = caught.value
    assert error.api_status == 'incomplete'
    assert error.reported_output_tokens == 1200
    assert 'API status: incomplete' in str(error)
    assert 'reported output tokens: 1,200' in str(error)
    assert 'Unfinished code' not in str(error) and 'private' not in str(error)
    if reason == 'max_output_tokens':
        assert error.reason == reason and 'Output limit reached' in str(error)
        assert '1,200-token output cap' in str(error)
    else:
        assert 'Output limit reached' not in str(error)
    if streaming:
        assert seen == ['Unfinished code: ', '```python\n']


@pytest.mark.parametrize('streaming', [False, True])
def test_unknown_usage_is_not_inferred_from_cap(streaming):
    kwargs = {'on_delta': lambda text: None} if streaming else {}
    with pytest.raises(ResponseStatusError) as caught:
        asyncio.run(limit_provider(streaming, usage=False).complete(model='gpt-4.1-mini', instructions='Policy',
            messages=[{'role': 'user', 'content': 'Hello'}], max_output_tokens=1200, timeout=2, **kwargs))
    assert caught.value.reported_output_tokens is None
    assert 'reported output tokens:' not in str(caught.value)


@pytest.mark.parametrize('streaming', [False, True])
def test_engine_persists_reason_but_no_partial_answer_and_unlocks(repo, agent, conversation, streaming):
    preview = []
    with pytest.raises(AppError, match='Output limit reached'):
        asyncio.run(AgentEngine(repo, limit_provider(streaming), on_text=preview.append if streaming else None)
                    .execute(agent.id, conversation, 'Ten steps with code'))
    run = repo.runs(agent.id)[0]
    assert run['status'] == 'failed' and run['requests'] == 1
    assert 'termination reason: max_output_tokens' in run['error']
    assert run['input_tokens'] is None and run['output_tokens'] is None
    assert 'reported output tokens: 1,200' in run['error']
    assert run['steps'] == '[]'
    assert [m['role'] for m in repo.messages(conversation)] == ['user']
    assert repo.messages(conversation, completed_only=True) == []
    assert events(repo, run_id=run['id'])[-1]['error_category'] == 'Output limit'
    assert Analytics(repo).summary(Filters())['generation_tokens'] is None
    assert repo.begin_run(conversation, agent, 'Manual retry')


def test_limit_on_draft_does_not_run_reviews(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3)))
    error = ResponseStatusError(SimpleNamespace(status='incomplete',
        incomplete_details=SimpleNamespace(reason='max_output_tokens'), usage=None), 1200)
    provider = SimpleNamespace(complete=AsyncMock(side_effect=error))
    with pytest.raises(AppError, match='Output limit reached'):
        asyncio.run(AgentEngine(repo, provider, on_text=lambda text: None).execute(agent.id, conversation, 'Ten steps'))
    assert provider.complete.await_count == 1
    assert repo.runs(agent.id)[0]['status'] == 'failed'
    assert len(repo.messages(conversation)) == 1


def test_worker_discards_output_limited_preview(repo, agent, conversation):
    job = ChatJob(repo, limit_provider(True), agent.id, conversation, 'Ten steps').start()
    assert job.wait()
    snapshot = job.snapshot()
    assert snapshot.state == 'Failed' and snapshot.text == ''
    assert 'termination reason: max_output_tokens' in snapshot.error
    assert len(repo.messages(conversation)) == 1


def test_incomplete_rag_does_not_commit_even_when_partial_text_has_valid_citation(repo, agent, conversation, kb):
    indexed(kb)
    enable(repo, agent)
    data = stream_events(chunks=('London [S1]. ',), terminal='incomplete')
    data[-1]['response'] = incomplete_response()
    with pytest.raises(AppError, match='Output limit reached'):
        asyncio.run(AgentEngine(repo, sdk_provider(SSE(data)), kb.retrieval, on_text=lambda text: None)
                    .execute(agent.id, conversation, 'Where is the deployment?'))
    run = repo.runs(agent.id)[0]
    assert run['status'] == 'failed'
    assert kb.repository.sources(run['id'], agent.id) == []
    assert len(repo.messages(conversation)) == 1


@pytest.mark.parametrize('cap', [MIN_OUTPUT_TOKENS, 4096, 8192, MAX_OUTPUT_TOKENS])
def test_output_limit_supported_and_persisted(repo, agent, cap):
    configured = replace(agent, settings=replace(agent.settings, max_output_tokens=cap))
    repo.save_agent(configured)
    assert repo.agent(agent.id).settings.max_output_tokens == cap


@pytest.mark.parametrize('cap', [MIN_OUTPUT_TOKENS - 1, MAX_OUTPUT_TOKENS + 1, True, 4096.5])
def test_output_limit_cannot_bypass_bounds(cap):
    with pytest.raises(AppError, match='max_output_tokens'):
        replace(Settings(), max_output_tokens=cap).validate()


def test_larger_cap_does_not_override_total_task_budget(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_output_tokens=8192, token_budget=8192)))
    provider = SimpleNamespace(complete=AsyncMock(return_value=Completion('Answer', 1, 1)))
    with pytest.raises(AppError, match='budget'):
        asyncio.run(AgentEngine(repo, provider, on_text=lambda text: None).execute(agent.id, conversation, 'Hello'))
    provider.complete.assert_not_called()


def test_larger_cap_sent_unchanged_in_stream_request(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_output_tokens=8192)))
    requests = []
    result = asyncio.run(AgentEngine(repo, sdk_provider(SSE(stream_events()), requests), on_text=lambda text: None)
                         .execute(agent.id, conversation, 'Hello'))
    assert result == 'Hello world' and len(requests) == 1
    assert requests[0]['max_output_tokens'] == 8192
    assert len(repo.messages(conversation)) == 2


def test_ui_shows_actionable_output_limit_and_saves_manual_configuration(tmp_path, monkeypatch):
    from ai_agent_pro.config import Agent
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Tutorial assistant', '', 'Be helpful')
    repo.save_agent(agent)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    error = ResponseStatusError(SimpleNamespace(status='incomplete',
        incomplete_details=SimpleNamespace(reason='max_output_tokens'), usage=None), 1200)
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', AsyncMock(side_effect=error))
    at.sidebar.radio[0].set_value('Chat').run()
    assert any('1,200 output tokens/request' in c.value for c in at.caption)
    at.chat_input[0].set_value('Ten steps with code').run()
    finish_chat(at)
    assert any('Output limit reached' in e.value for e in at.error)
    assert any('4,096–8,192' in c.value for c in at.caption)
    button(at, 'Agent settings').click().run()
    limit = next(n for n in at.number_input if n.label == 'Output tokens per request')
    assert limit.value == 1200  # Existing agents are not silently changed.
    assert limit.proto.max == MAX_OUTPUT_TOKENS
    limit.set_value(8192)
    button(at, 'Save agent').click().run()
    assert repo.agent(agent.id).settings.max_output_tokens == 8192
    assert repo.agent(agent.id).settings.token_budget == 40000
    assert not at.exception
