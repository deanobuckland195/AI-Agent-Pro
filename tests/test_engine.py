import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock
import pytest
from ai_agent_pro.config import AppError
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.provider import Completion


class FakeProvider:
    def __init__(self):
        self.complete = AsyncMock(return_value=Completion('Helpful answer', 100, 30))


def execute(repo, provider, agent, cid, prompt='Help me plan'):
    return asyncio.run(AgentEngine(repo, provider).execute(agent.id, cid, prompt))


def test_three_stage_workflow_and_history(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3)))
    provider = FakeProvider()
    assert execute(repo, provider, agent, conversation) == 'Helpful answer'
    assert provider.complete.call_count == 3
    assert provider.complete.call_args_list[1].kwargs['messages'][-1]['role'] == 'assistant'
    run = repo.runs(agent.id)[0]
    assert (run['requests'], run['input_tokens'], run['output_tokens']) == (3, 300, 90)
    assert len(repo.messages(conversation)) == 2
    execute(repo, provider, agent, conversation, 'Follow up')
    assert provider.complete.call_args_list[3].kwargs['messages'][0]['content'] == 'Help me plan'


def test_zero_history_and_agent_isolation(repo, agent, conversation):
    provider = FakeProvider()
    execute(repo, provider, agent, conversation)
    repo.save_agent(replace(agent, settings=replace(agent.settings, history_messages=0)))
    execute(repo, provider, agent, conversation, 'Second')
    assert len(provider.complete.call_args.kwargs['messages']) == 1
    second = repo.new_conversation(agent.id)
    execute(repo, provider, agent, second, 'Fresh')
    assert provider.complete.call_args.kwargs['messages'] == [{'role': 'user', 'content': 'Fresh'}]


def test_budget_prevents_api_call(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, token_budget=1024)))
    provider = FakeProvider()
    with pytest.raises(AppError, match='budget'):
        execute(repo, provider, agent, conversation)
    provider.complete.assert_not_called()
    assert repo.runs(agent.id)[0]['status'] == 'failed'


def test_budget_stops_later_steps(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3, token_budget=3000)))
    provider = FakeProvider()
    provider.complete.return_value = Completion('draft', 2000, 200)
    with pytest.raises(AppError, match='budget'):
        execute(repo, provider, agent, conversation)
    assert provider.complete.call_count == 1
    assert repo.runs(agent.id)[0]['requests'] == 1


@pytest.mark.parametrize('exception,expected', [(AppError('Safe API failure'), 'Safe API failure'),
    (RuntimeError('sk-privateabcdefgh'), 'Unexpected'), (asyncio.TimeoutError(), 'deadline')])
def test_errors_persist_without_leaks(repo, agent, conversation, exception, expected):
    provider = FakeProvider()
    provider.complete.side_effect = exception
    with pytest.raises(AppError, match=expected):
        execute(repo, provider, agent, conversation)
    run = repo.runs(agent.id)[0]
    assert run['status'] == 'failed'
    assert run['input_tokens'] is None
    assert 'sk-private' not in run['error']
    assert repo.messages(conversation, completed_only=True) == []


def test_actual_deadline_cancels_provider(repo, agent, conversation, monkeypatch):
    provider = FakeProvider()
    cancelled = []
    async def slow(**kwargs):
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)
    provider.complete.side_effect = slow
    # Exercise asyncio cancellation with a short remaining task deadline.
    import ai_agent_pro.engine as engine
    values = iter([0, 59.99])
    monkeypatch.setattr(engine, 'monotonic', lambda: next(values, 60.1))
    with pytest.raises(AppError, match='deadline'):
        execute(repo, provider, agent, conversation)
    assert cancelled


def test_unknown_usage(repo, agent, conversation):
    provider = FakeProvider()
    provider.complete.return_value = Completion('Answer', None, None)
    execute(repo, provider, agent, conversation)
    assert repo.runs(agent.id)[0]['input_tokens'] is None


@pytest.mark.parametrize('prompt', ['', 'x' * 12001, 'bad\x00input'])
def test_invalid_inputs_never_call_provider(repo, agent, conversation, prompt):
    provider = FakeProvider()
    with pytest.raises(AppError):
        execute(repo, provider, agent, conversation, prompt)
    provider.complete.assert_not_called()
    assert repo.runs(agent.id) == []
