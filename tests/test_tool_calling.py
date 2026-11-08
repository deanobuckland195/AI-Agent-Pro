"""Controlled Responses loops over real Stage 4A workers; no paid API calls."""
import asyncio
from dataclasses import replace
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_agent_pro.analytics import Analytics, Filters
from ai_agent_pro.config import AppError
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.execution import ChatJob
from ai_agent_pro.telemetry import events
from ai_agent_pro.tool_calling import ToolAccess, tool_reservation
from test_provider import provider_with, response_body
from test_tools import tools
from test_rag_retrieval import kb, indexed
from test_rag_engine import enable
from test_streaming import SSE

ALIAS = 'calculator__v1_0_0'


def function_item(name=ALIAS, arguments='{"expression":"2+2"}', call_id='call_1', status='completed'):
    return dict(type='function_call', id='fc_' + call_id, name=name, arguments=arguments,
                call_id=call_id, status=status)


def function_body(**kwargs):
    body = response_body(text='')
    body['output'] = [function_item(**kwargs)]
    return body


def access(tools, agent, selected=(('calculator', '1.0.0'),)):
    for name, version in selected:
        tools.permissions.allow(agent.id, name, version)
    return ToolAccess(tools, selected)


def run(repo, agent, conversation, provider, selection, prompt='Calculate 2+2', **kwargs):
    return asyncio.run(AgentEngine(repo, provider, tool_access=selection, **kwargs).execute(
        agent.id, conversation, prompt))


def test_sdk_sequential_loop_versions_outputs_analytics_and_privacy(repo, agent, conversation, tools):
    # A wider host grant must never widen this execution's selected tools.
    tools.permissions.allow(agent.id, 'python_inspect')
    selection = access(tools, agent)
    requests, activity = [], []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['store'] is False and body['parallel_tool_calls'] is False
        assert 'previous_response_id' not in body
        assert [t['name'] for t in body['tools']] == [ALIAS]
        assert body['tools'][0]['strict'] is True
        assert body['tools'][0]['parameters']['additionalProperties'] is False
        assert 'You have no tools' not in body['instructions']
        if len(requests) < 3:
            return httpx.Response(200, json=function_body(call_id=f'call_{len(requests)}'))
        outputs = [m for m in body['input'] if m.get('type') == 'function_call_output']
        assert [m['call_id'] for m in outputs] == ['call_1', 'call_2']
        assert all(json.loads(m['output']) == {'ok': True, 'data': {'value': 4}} for m in outputs)
        assert len([m for m in body['input'] if m.get('type') == 'function_call']) == 2
        return httpx.Response(200, json=response_body(text='**Result:** 4'))
    assert run(repo, agent, conversation, provider_with(handler), selection, on_tool=activity.append) == '**Result:** 4'
    record = repo.runs(agent.id)[0]
    assert (record['requests'], record['input_tokens'], record['output_tokens']) == (3, 60, 30)
    summary = Analytics(repo).summary(Filters())
    assert (summary['generation_tokens'], summary['known_stages'], summary['stages']) == (90, 3, 3)
    assert [a['state'] for a in activity] == ['started', 'completed'] * 2
    assert activity[1]['data'] == {'value': 4}
    retained = events(repo, run_id=record['id'])
    assert len([e for e in retained if e['stage'] == 'tool']) == 4
    assert all(e['outcome'] == 'calculator v1.0.0' for e in retained if e['stage'] == 'tool')
    assert 'expression' not in json.dumps(retained)
    assert len(repo.messages(conversation)) == 2


@pytest.mark.parametrize('arguments', ['{', '[]', 'null', '{"expression":true}',
    '{"expression":"2","extra":1}', '{"expression":"1","expression":"2"}',
    '{"expression":NaN}', '{"expression":"' + '1' * 257 + '"}', '[' * 1500])
def test_bad_arguments_never_start_executor(repo, agent, conversation, tools, monkeypatch, arguments):
    selection = access(tools, agent)
    execute = AsyncMock()
    monkeypatch.setattr(tools, 'execute', execute)
    with pytest.raises(AppError, match='arguments') as error:
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200, json=function_body(arguments=arguments))), selection)
    assert 'expression' not in str(error.value)
    execute.assert_not_called()
    assert repo.runs(agent.id)[0]['status'] == 'failed'
    assert len(repo.messages(conversation)) == 1


@pytest.mark.parametrize('name', ['calculator', 'calculator__v1_1_0', 'python_inspect__v1_0_0', 'shell', 'sk-privateabcdefgh'])
def test_unknown_or_unselected_names_fail_closed(repo, agent, conversation, tools, monkeypatch, name):
    selection = access(tools, agent)
    execute = AsyncMock()
    monkeypatch.setattr(tools, 'execute', execute)
    with pytest.raises(AppError, match='approved') as error:
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200, json=function_body(name=name))), selection)
    execute.assert_not_called()
    assert name not in str(error.value)


def test_selection_requires_exact_agent_grant(repo, agent, conversation, tools):
    tools.permissions.allow('other-agent', 'calculator')
    provider = type('Fake', (), {'complete': AsyncMock()})()
    with pytest.raises(AppError, match='not approved'):
        run(repo, agent, conversation, provider, ToolAccess(tools, (('calculator', '1.0.0'),)))
    provider.complete.assert_not_called()


@pytest.mark.parametrize('when', ['model', 'worker', 'after_result', 'before_save'])
def test_revocation_invalidates_execution_and_prevents_save(repo, agent, conversation, tools, monkeypatch, when):
    selection = access(tools, agent)
    count = 0
    original = tools.execute
    async def execute(*args, **kwargs):
        result = await original(*args, **kwargs)
        if when == 'worker':
            tools.permissions.revoke(agent.id, 'calculator')
        return result
    monkeypatch.setattr(tools, 'execute', execute)
    def handler(request):
        nonlocal count
        count += 1
        if when == 'model':
            tools.permissions.revoke(agent.id, 'calculator')
        return httpx.Response(200, json=function_body() if count == 1 else response_body())
    def on_tool(event):
        if when == 'after_result' and event['state'] == 'completed':
            tools.permissions.revoke(agent.id, 'calculator')
    def observer(event):
        if when == 'before_save' and event['stage'] == 'save':
            tools.permissions.revoke(agent.id, 'calculator')
    with pytest.raises(AppError, match='permissions'):
        run(repo, agent, conversation, provider_with(handler), selection, on_tool=on_tool, observer=observer)
    assert count == (2 if when == 'before_save' else 1)
    assert len(repo.messages(conversation)) == 1


def test_duplicate_call_never_executes_twice(repo, agent, conversation, tools):
    activity = []
    with pytest.raises(AppError, match='duplicate'):
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200, json=function_body())),
            access(tools, agent), on_tool=activity.append)
    assert [e['state'] for e in activity] == ['started', 'completed']


def test_tool_call_and_model_request_limits(repo, agent, conversation, tools):
    count, activity = [], []
    def handler(request):
        count.append(request)
        return httpx.Response(200, json=function_body(call_id=f'call_{len(count)}'))
    with pytest.raises(AppError, match='limit'):
        run(repo, agent, conversation, provider_with(handler), access(tools, agent), on_tool=activity.append)
    assert len(count) == 6 and len(activity) == 10
    assert repo.runs(agent.id)[0]['requests'] == 6


def test_model_request_limit_is_global_across_reviews(repo, agent, conversation, tools, monkeypatch):
    monkeypatch.setattr('ai_agent_pro.tool_calling.MAX_MODEL_REQUESTS', 2)
    repo.save_agent(replace(agent, settings=replace(agent.settings, max_steps=3)))
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=function_body() if len(requests) == 1 else response_body())
    with pytest.raises(AppError, match='request limit'):
        run(repo, agent, conversation, provider_with(handler), access(tools, agent))
    assert len(requests) == 2
    assert Analytics(repo).summary(Filters())['generation_tokens'] == 60


@pytest.mark.parametrize('known', [True, False])
def test_tokens_account_for_tool_rounds_and_unknown_usage(repo, agent, conversation, tools, known):
    repo.save_agent(replace(agent, settings=replace(agent.settings, token_budget=6000)))
    count = []
    def handler(request):
        count.append(request)
        body = function_body(call_id=f'call_{len(count)}')
        body['usage'] = {'input_tokens': 5900, 'output_tokens': 101, 'total_tokens': 6001} if known else None
        return httpx.Response(200, json=body)
    with pytest.raises(AppError, match='budget'):
        run(repo, agent, conversation, provider_with(handler), access(tools, agent))
    assert len(count) <= 2
    assert repo.runs(agent.id)[0]['input_tokens'] is None


def test_context_reservation_includes_schemas_and_results():
    assert tool_reservation('abc', [{'type':'function_call_output','call_id':'c','output':'x'*100}], [{'name':'n'}], 10) > 1140
    with pytest.raises(AppError, match='context size'):
        tool_reservation('', [{'output':'x'*131072}], [], 100)


def test_parallel_call_batch_is_rejected_before_any_worker(repo, agent, conversation, tools, monkeypatch):
    selection = access(tools, agent)
    execute = AsyncMock()
    monkeypatch.setattr(tools, 'execute', execute)
    body = function_body()
    body['output'].append(function_item(call_id='call_2'))
    with pytest.raises(AppError, match='one sequential'):
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200, json=body)), selection)
    execute.assert_not_called()


def sse_tool_events(*, complete=True, done=True, arguments_done=True, arguments='{"expression":"2+2"}'):
    item = function_item(arguments=arguments)
    initial = response_body(text='')
    initial.update(status='in_progress', output=[])
    result = function_body(arguments=arguments)
    items = [dict(type='response.created', response=initial),
             dict(type='response.output_item.added', output_index=0, item=dict(item, status='in_progress', arguments='')),
             dict(type='response.function_call_arguments.delta', item_id=item['id'], output_index=0, delta=arguments)]
    if arguments_done:
        items.append(dict(type='response.function_call_arguments.done', item_id=item['id'], output_index=0, arguments=arguments))
    if done:
        items.append(dict(type='response.output_item.done', output_index=0, item=item))
    if complete:
        items.append(dict(type='response.completed', response=result))
    return [dict(event, sequence_number=i) for i, event in enumerate(items)]


@pytest.mark.parametrize('stream_options', [{'complete':False}, {'done':False}, {'arguments_done':False}])
def test_partial_streamed_calls_never_execute(repo, agent, conversation, tools, monkeypatch, stream_options):
    selection = access(tools, agent)
    execute = AsyncMock()
    monkeypatch.setattr(tools, 'execute', execute)
    wire = SSE(sse_tool_events(**stream_options))
    with pytest.raises(AppError, match='incomplete tool call|connection ended'):
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200,
            headers={'content-type':'text/event-stream'}, stream=wire)), selection, on_text=lambda text: None)
    execute.assert_not_called()
    assert wire.closed


def test_complete_sdk_streamed_call_then_streamed_answer(repo, agent, conversation, tools):
    from test_streaming import stream_events
    wires, bodies, previews, activity = [], [], [], []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body['stream'] and body['store'] is False
        wire = SSE(sse_tool_events() if len(bodies) == 1 else stream_events())
        wires.append(wire)
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, stream=wire)
    assert run(repo, agent, conversation, provider_with(handler), access(tools, agent),
        on_text=previews.append, on_tool=activity.append) == 'Hello world'
    assert all(w.closed for w in wires)
    assert any('Hello' in p for p in previews)
    assert not any('expression' in p for p in previews)
    assert activity[-1]['data'] == {'value': 4}
    assert bodies[1]['input'][-1]['call_id'] == 'call_1'


@pytest.mark.parametrize('status', ['in_progress', 'incomplete'])
def test_unfinished_nonstream_call_rejected(repo, agent, conversation, tools, status):
    with pytest.raises(AppError, match='invalid'):
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200, json=function_body(status=status))), access(tools, agent))


def test_worker_failure_stops_loop_without_partial_answer(repo, agent, conversation, tools):
    activity = []
    with pytest.raises(AppError, match='arithmetic'):
        run(repo, agent, conversation, provider_with(lambda r: httpx.Response(200,
            json=function_body(arguments='{"expression":"__import__(1)"}'))), access(tools, agent), on_tool=activity.append)
    assert [e['state'] for e in activity] == ['started', 'failed']
    assert len(repo.messages(conversation)) == 1


def test_task_deadline_and_cancellation_wrap_executor(repo, agent, conversation, tools, monkeypatch):
    selection = access(tools, agent)
    cancelled = []
    arrived = asyncio.Event()
    async def slow(*args, **kwargs):
        try:
            arrived.set()
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    monkeypatch.setattr(tools, 'execute', slow)
    async def scenario():
        task = asyncio.create_task(AgentEngine(repo, provider_with(lambda r: httpx.Response(200,
            json=function_body())), tool_access=selection).execute(agent.id, conversation, 'Hi'))
        await asyncio.wait_for(arrived.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert cancelled and repo.runs(agent.id)[0]['status'] == 'interrupted'
    assert len(repo.messages(conversation)) == 1


def test_tool_results_redacted_without_retaining_source_in_sqlite(repo, agent, conversation, tools, tmp_path):
    (tmp_path / 'notes.txt').write_text('password: private-value\nVisible line')
    tools.permissions.approve_files(agent.id, ['notes.txt'])
    selection = access(tools, agent, (('project_read', '1.0.0'),))
    requests, activity = [], []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=function_body(name='project_read__v1_0_0', arguments=json.dumps(dict(path='notes.txt',start_line=1,line_count=10))))
        assert 'private-value' not in json.dumps(body)
        assert '[REDACTED]' in body['input'][-1]['output']
        return httpx.Response(200, json=response_body())
    run(repo, agent, conversation, provider_with(handler), selection, on_tool=activity.append)
    assert 'private-value' not in json.dumps(activity)
    assert 'Visible line' in json.dumps(activity)
    assert 'Visible line' not in json.dumps(events(repo, run_id=repo.runs(agent.id)[0]['id']))


def test_rag_tool_loop_preserves_citations(repo, agent, conversation, tools, kb):
    indexed(kb)
    enable(repo, agent)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=function_body() if len(requests) == 1 else response_body(text='London [S1].'))
    run(repo, agent, conversation, provider_with(handler), access(tools, agent), prompt='Where is the deployment?', retrieval=kb.retrieval)
    record = repo.runs(agent.id)[0]
    assert record['status'] == 'completed' and kb.repository.sources(record['id'], agent.id)


def test_knowledge_deletion_after_tool_result_blocks_followup(repo, agent, conversation, tools, kb):
    did = indexed(kb)
    selection = access(tools, agent, (('knowledge_search', '1.0.0'),))
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=function_body(name='knowledge_search__v1_0_0', arguments='{"query":"London","max_results":2}'))
    def observe(event):
        if event['state'] == 'completed':
            kb.repository.delete_document(did)
    with pytest.raises(AppError, match='sources changed'):
        run(repo, agent, conversation, provider_with(handler), selection, on_tool=observe)
    assert len(requests) == 1 and len(repo.messages(conversation)) == 1


def test_job_snapshot_has_sanitized_results(repo, agent, conversation, tools):
    requests = []
    def handler(request):
        requests.append(request)
        from test_streaming import stream_events
        return httpx.Response(200, headers={'content-type':'text/event-stream'},
            stream=SSE(sse_tool_events() if len(requests) == 1 else stream_events()))
    job = ChatJob(repo, provider_with(handler), agent.id, conversation, 'Calculate', tool_access=access(tools, agent)).start()
    assert job.wait(5)
    snapshot = job.snapshot()
    assert snapshot.state == 'Completed' and snapshot.tools[0]['data'] == {'value':4}


def test_whole_task_deadline_cancels_tool_wait(repo, agent, conversation, tools, monkeypatch):
    from time import monotonic
    selection = access(tools, agent)
    repo.save_agent(replace(agent, settings=replace(agent.settings, timeout_seconds=5)))
    extra = 0
    monkeypatch.setattr('ai_agent_pro.engine.monotonic', lambda: monotonic() + extra)
    cancelled = []
    async def slow(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    monkeypatch.setattr(tools, 'execute', slow)
    def handler(request):
        nonlocal extra
        extra = 4.8
        return httpx.Response(200, json=function_body())
    with pytest.raises(AppError, match='deadline'):
        run(repo, agent, conversation, provider_with(handler), selection)
    assert cancelled and repo.runs(agent.id)[0]['status'] == 'failed'


def test_oversized_cumulative_results_stop_before_followup(repo, agent, conversation, tools, tmp_path, monkeypatch):
    from ai_agent_pro.tools.permissions import ToolBudget
    monkeypatch.setattr('ai_agent_pro.tool_calling.ToolBudget', lambda **kwargs: ToolBudget(result_bytes=1))
    requests, activity = [], []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=function_body())
    with pytest.raises(AppError, match='budget'):
        run(repo, agent, conversation, provider_with(handler), access(tools, agent), on_tool=activity.append)
    assert len(requests) == 1
    assert [e['state'] for e in activity] == ['started', 'failed']


def test_reasoning_items_replayed_with_call_output(repo, agent, conversation, tools):
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            response = function_body()
            response['output'].insert(0, dict(type='reasoning', id='rs_test', summary=[]))
            return httpx.Response(200, json=response)
        assert [m.get('type') for m in body['input']][-3:] == ['reasoning', 'function_call', 'function_call_output']
        return httpx.Response(200, json=response_body())
    run(repo, agent, conversation, provider_with(handler), access(tools, agent))


def test_observer_cannot_mutate_tool_results_sent_to_model(repo, agent, conversation, tools):
    count = []
    def handler(request):
        count.append(request)
        if len(count) == 1:
            return httpx.Response(200, json=function_body())
        assert json.loads(json.loads(request.content)['input'][-1]['output'])['data']['value'] == 4
        return httpx.Response(200, json=response_body())
    def observe(event):
        if event['data'] is not None:
            event['data']['value'] = 999
        raise RuntimeError('private observer details')
    run(repo, agent, conversation, provider_with(handler), access(tools, agent), on_tool=observe)


def test_missing_knowledge_binding_does_not_touch_database(tmp_path):
    from ai_agent_pro.tools.permissions import ToolPermissions
    with_db = tmp_path / 'data'
    with_db.mkdir()
    (with_db / 'agentpro.sqlite3').write_bytes(b'not a database')
    authority = ToolPermissions(tmp_path, bind_knowledge=False)
    assert authority.database_identity is None and authority._database_lease is None


def test_job_does_not_cache_knowledge_excerpts(repo, agent, conversation):
    from test_engine import FakeProvider
    job = ChatJob(repo, FakeProvider(), agent.id, conversation, 'Hi')
    job._tool(dict(tool='knowledge_search', version='1.0.0', ordinal=1, state='completed',
                   data=dict(method='local_lexical', sources=[dict(excerpt='private source excerpt')])))
    snapshot = job.snapshot()
    assert snapshot.tools[0]['data'] == {'method':'local_lexical', 'source_count':1}
    assert 'private source excerpt' not in str(snapshot)
    snapshot.tools[0]['data']['source_count'] = 999
    assert job.snapshot().tools[0]['data']['source_count'] == 1


@pytest.mark.parametrize('streaming', [False, True])
def test_text_claim_without_required_call_is_not_saved(repo, agent, conversation, tools, streaming):
    from test_streaming import stream_events
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        if streaming:
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                stream=SSE(stream_events(chunks=('I used the calculator: 4',))))
        return httpx.Response(200, json=response_body(text='I used the calculator: 4'))
    with pytest.raises(AppError, match='did not call'):
        run(repo, agent, conversation, provider_with(handler), access(tools, agent),
            on_text=(lambda text: None) if streaming else None)
    assert len(bodies) == 1 and bodies[0]['tool_choice'] == 'required'
    assert repo.runs(agent.id)[0]['status'] == 'failed'
    assert [m['role'] for m in repo.messages(conversation)] == ['user']
    assert not any(e['stage'] == 'tool' for e in events(repo, run_id=repo.runs(agent.id)[0]['id']))
