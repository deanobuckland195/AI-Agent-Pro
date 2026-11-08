"""SDK-backed SSE streams and atomic streamed engine execution (offline only)."""
import asyncio
from dataclasses import replace
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_agent_pro.config import AppError
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.provider import Completion
from ai_agent_pro.safety import streaming_redact
from test_provider import provider_with, response_body
from test_rag_retrieval import kb, indexed
from test_rag_engine import enable


def stream_events(chunks=('Hello ', 'world'), terminal='completed', usage=True):
    response = response_body(text='', usage=False)
    response.update(status='in_progress', output=[])
    events = [dict(type='response.created', response=response),
              dict(type='response.output_item.added', output_index=0,
                   item=dict(id='msg_test', type='message', role='assistant', status='in_progress', content=[])),
              dict(type='response.content_part.added', output_index=0, content_index=0,
                   item_id='msg_test', part=dict(type='output_text', text='', annotations=[]))]
    events += [dict(type='response.output_text.delta', delta=chunk, output_index=0,
                    content_index=0, item_id='msg_test', logprobs=[]) for chunk in chunks]
    if terminal:
        events.append(dict(type='response.' + terminal,
                           response=response_body(status=terminal, text=''.join(chunks), usage=usage)))
    return [dict(event, sequence_number=i) for i, event in enumerate(events)]


def encode_event(event):
    return ('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode()


class SSE(httpx.AsyncByteStream):
    def __init__(self, events, *, pause=False, failure=None):
        self.events, self.pause, self.failure = events, pause, failure
        self.closed = False
        self.emitted = 0

    async def __aiter__(self):
        for event in self.events:
            self.emitted += 1
            yield encode_event(event)
            await asyncio.sleep(0)
        if self.failure:
            raise self.failure('private transport detail')
        if self.pause:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def sdk_provider(wire, requests=None):
    def handler(request):
        body = json.loads(request.content)
        assert body['stream'] is True and body['store'] is False
        assert 'tools' not in body and 'background' not in body
        if requests is not None:
            requests.append(body)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)
    return provider_with(handler)


def invoke_stream(provider, callback, timeout=2):
    return provider.complete(model='gpt-4.1-mini', instructions='System policy',
                             messages=[{'role': 'user', 'content': 'Hi'}],
                             max_output_tokens=100, timeout=timeout, on_delta=callback)


def test_sdk_stream_is_progressive_and_final_usage_is_authoritative():
    wire = SSE(stream_events())
    seen, requests = [], []
    def delta(text):
        seen.append(text)
        assert wire.emitted < len(wire.events)
    result = asyncio.run(invoke_stream(sdk_provider(wire, requests), delta))
    assert seen == ['Hello ', 'world']
    assert result == Completion('Hello world', 20, 10)
    assert wire.closed and len(requests) == 1


@pytest.mark.parametrize('terminal,match', [(None, 'connection ended'), ('incomplete', 'incomplete'), ('failed', 'failed')])
def test_terminal_failure_never_becomes_completion(terminal, match):
    wire = SSE(stream_events(terminal=terminal))
    with pytest.raises(AppError, match=match):
        asyncio.run(invoke_stream(sdk_provider(wire), lambda text: None))
    assert wire.closed


@pytest.mark.parametrize('failure,match', [(httpx.ReadError, 'connection'), (httpx.ReadTimeout, 'timed out')])
def test_mid_stream_transport_failures_close_connection(failure, match):
    wire = SSE(stream_events(terminal=None), failure=failure)
    with pytest.raises(AppError, match=match) as error:
        asyncio.run(invoke_stream(sdk_provider(wire), lambda text: None))
    assert wire.closed and 'private' not in str(error.value)


def test_stalled_stream_obeys_whole_request_deadline():
    wire = SSE(stream_events(terminal=None), pause=True)
    with pytest.raises(AppError, match='timed out'):
        asyncio.run(invoke_stream(sdk_provider(wire), lambda text: None, timeout=.05))
    assert wire.closed


def test_sdk_stream_explicit_cancellation_closes_connection():
    wire = SSE(stream_events(terminal=None), pause=True)
    async def scenario():
        arrived = asyncio.Event()
        task = asyncio.create_task(invoke_stream(sdk_provider(wire), lambda text: arrived.set()))
        await asyncio.wait_for(arrived.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert wire.closed


def test_stream_usage_may_be_unknown():
    result = asyncio.run(invoke_stream(sdk_provider(SSE(stream_events(usage=False))), lambda text: None))
    assert result.input_tokens is None and result.output_tokens is None


def test_stream_size_limit_is_enforced_before_preview():
    wire = SSE(stream_events(chunks=('x' * 100001,), terminal=None))
    seen = []
    with pytest.raises(AppError, match='size limit'):
        asyncio.run(invoke_stream(sdk_provider(wire), seen.append))
    assert seen == [] and wire.closed


@pytest.mark.parametrize('secret', ['sk-privateabcdefgh', 'Bearer privateabcdefgh',
                                    'api key: privateabcdefgh', 'password:\nprivateabcdefgh'])
def test_split_secret_is_never_published_in_preview(secret):
    text = 'A safe introduction. ' + secret + '\nMore words '
    for end in range(len(text) + 1):
        preview = streaming_redact(text[:end])
        assert 'private' not in preview and 'sk-' not in preview
    assert '[REDACTED]' in streaming_redact(text)


def fake_stream(text='An answer ', fail=None):
    async def complete(on_delta=None, **kwargs):
        if on_delta:
            for character in text:
                on_delta(character)
                await asyncio.sleep(0)
        if fail:
            raise fail
        return Completion(text, 10, 5)
    return type('Fake', (), {'complete': AsyncMock(side_effect=complete)})()


def test_engine_streams_only_final_stage_and_saves_once(repo, agent, conversation):
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3)))
    provider, seen = fake_stream(), []
    def preview(text):
        seen.append(text)
        assert all(m['role'] != 'assistant' for m in repo.messages(conversation))
    result = asyncio.run(AgentEngine(repo, provider, on_text=preview).execute(agent.id, conversation, 'Hi'))
    assert result == 'An answer ' and seen[-1] == result
    assert [bool(c.kwargs.get('on_delta')) for c in provider.complete.call_args_list] == [False, False, True]
    assert len(repo.messages(conversation)) == 2


@pytest.mark.parametrize('when', ['before', 'streaming', 'between_stages', 'before_save'])
def test_engine_cancellation_releases_lock_without_partial_message(repo, agent, conversation, when):
    cancelled = when == 'before'
    provider = fake_stream()
    def preview(text):
        nonlocal cancelled
        if when == 'streaming':
            cancelled = True
    def observer(event):
        nonlocal cancelled
        if when == 'between_stages' and event['state'] == 'completed' and event['stage'] == 'draft':
            cancelled = True
        if when == 'before_save' and event['state'] == 'completed' and event['stage'] == 'answer':
            cancelled = True
    if when == 'between_stages':
        repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3)))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(AgentEngine(repo, provider, observer=observer, on_text=preview,
                    cancel_requested=lambda: cancelled).execute(agent.id, conversation, 'Hi'))
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'
    assert [m['role'] for m in repo.messages(conversation)] == ['user']
    assert provider.complete.call_count <= 1
    assert repo.messages(conversation, completed_only=True) == []
    assert repo.begin_run(conversation, agent, 'Retry')


def test_engine_failed_stream_does_not_save_partial(repo, agent, conversation):
    seen = []
    with pytest.raises(AppError, match='connection'):
        asyncio.run(AgentEngine(repo, fake_stream(fail=AppError('Lost connection')), on_text=seen.append)
                    .execute(agent.id, conversation, 'Hi'))
    assert any(seen) and repo.runs(agent.id)[0]['status'] == 'failed'
    assert len(repo.messages(conversation)) == 1


@pytest.mark.parametrize('answer,valid', [('London [S1]. ', True), ('London [S99]. ', False)])
def test_streaming_rag_validates_before_commit(repo, agent, conversation, kb, answer, valid):
    indexed(kb)
    enable(repo, agent)
    seen = []
    async def execute():
        return await AgentEngine(repo, fake_stream(answer), kb.retrieval, on_text=seen.append).execute(
            agent.id, conversation, 'Where is the deployment?')
    if valid:
        asyncio.run(execute())
    else:
        with pytest.raises(AppError, match='citation'):
            asyncio.run(execute())
    assert any(seen)
    run = repo.runs(agent.id)[0]
    assert run['status'] == ('completed' if valid else 'failed')
    assert len(repo.messages(conversation)) == (2 if valid else 1)
    assert bool(kb.repository.sources(run['id'], agent.id)) == valid


def test_streaming_rag_source_deleted_during_generation_is_not_committed(repo, agent, conversation, kb):
    did = indexed(kb)
    enable(repo, agent)
    seen = []
    async def complete(on_delta, **kwargs):
        on_delta('London [S1]. ')
        kb.repository.delete_document(did)
        return Completion('London [S1]. ', 20, 10)
    provider = type('Fake', (), {'complete': AsyncMock(side_effect=complete)})()
    with pytest.raises(AppError, match='sources changed'):
        asyncio.run(AgentEngine(repo, provider, kb.retrieval, on_text=seen.append).execute(
            agent.id, conversation, 'Where is the deployment?'))
    assert seen and repo.runs(agent.id)[0]['status'] == 'failed'
    assert len(repo.messages(conversation)) == 1
    assert kb.repository.sources(repo.runs(agent.id)[0]['id'], agent.id) == []


def test_no_evidence_does_not_start_stream(repo, agent, conversation, kb):
    enable(repo, agent)
    provider = fake_stream()
    asyncio.run(AgentEngine(repo, provider, kb.retrieval, on_text=lambda text: None).execute(
        agent.id, conversation, 'Where is the deployment?'))
    provider.complete.assert_not_called()
    assert repo.runs(agent.id)[0]['status'] == 'completed'


def test_empty_completed_stream_is_not_an_answer():
    wire = SSE(stream_events(chunks=('',)))
    with pytest.raises(AppError, match='no text'):
        asyncio.run(invoke_stream(sdk_provider(wire), lambda text: None))
    assert wire.closed
