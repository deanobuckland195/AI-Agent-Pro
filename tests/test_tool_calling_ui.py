"""UI approvals are agent scoped, explicit and unable to bypass file restrictions."""
import json
from unittest.mock import AsyncMock

from ai_agent_pro.config import Agent
from ai_agent_pro.provider import Completion, FunctionCall
from test_ui import launch, finish_chat
from test_tools import tools


def test_ui_opt_in_and_agent_isolation(tmp_path, monkeypatch, tools):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Calculator', '', 'Helpful')
    other = Agent.create('Other assistant', '', 'Helpful')
    repo.save_agent(agent)
    repo.save_agent(other)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    monkeypatch.setattr('ai_agent_pro.presentation.workspace.tool_executor', lambda path: tools)
    calls = []
    async def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert [s['name'] for s in kwargs['tools']] == ['calculator__v1_0_0']
            call = FunctionCall('calculator__v1_0_0', '{"expression":"2+2"}', 'call_ui')
            return Completion('', 20, 10, (call,), (dict(type='function_call', name=call.name,
                arguments=call.arguments, call_id=call.call_id),))
        assert json.loads(kwargs['messages'][-1]['output'])['data'] == {'value':4}
        kwargs['on_delta']('**4** ')
        return Completion('**4**', 20, 10)
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', AsyncMock(side_effect=complete))
    at.sidebar.radio[0].set_value('Chat').run()
    at.sidebar.selectbox[0].set_value(agent.id).run()
    assert at.multiselect[0].value == []
    at.multiselect[0].select('calculator').run()
    at.chat_input[0].set_value('Calculate').run()
    finish_chat(at)
    assert len(calls) == 2
    assert any('calculator v1.0.0 · completed' in t.value for t in at.text)
    assert not at.exception
    at.sidebar.selectbox[0].set_value(other.id).run()
    assert at.multiselect[0].value == []


def test_ui_private_filename_cannot_be_approved(tmp_path, monkeypatch, tools):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Reader', '', 'Helpful')
    repo.save_agent(agent)
    provider = AsyncMock()
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    monkeypatch.setattr('ai_agent_pro.presentation.workspace.tool_executor', lambda path: tools)
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', provider)
    at.sidebar.radio[0].set_value('Chat').run()
    at.multiselect[0].select('project_read').run()
    at.text_area[0].set_value('API.key').run()
    at.chat_input[0].set_value('Read approved files').run()
    assert not at.exception
    assert any('cannot be approved' in e.value for e in at.error)
    provider.assert_not_called()
    assert repo.runs(agent.id) == []


def test_selected_calculator_reaches_sdk_and_runs_before_answer(tmp_path, monkeypatch, tools):
    """Exercise UI -> ChatJob -> engine -> real SDK SSE -> real worker."""
    import httpx
    from ai_agent_pro.telemetry import events
    from test_provider import provider_with
    from test_streaming import SSE, stream_events
    from test_tool_calling import sse_tool_events

    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Live calculator', '', 'Helpful')
    repo.save_agent(agent)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    monkeypatch.setattr('ai_agent_pro.presentation.workspace.tool_executor', lambda path: tools)
    # A leftover host grant must not widen the current UI selection.
    tools.permissions.allow(agent.id, 'python_inspect')
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['stream'] and not body['store']
        outputs = [m for m in body['input'] if m.get('type') == 'function_call_output']
        if body.get('tools'):
            assert [t['name'] for t in body['tools']] == ['calculator__v1_0_0']
            assert body['tools'][0]['strict'] is True
            assert body['parallel_tool_calls'] is False
            assert 'You have no tools' not in body['instructions']
            if not outputs:
                # Default/auto reproduces the live symptom: a correct text-only
                # answer claiming calculator use. Required must reach the wire.
                items = (sse_tool_events() if body.get('tool_choice') == 'required' else
                         stream_events(chunks=('I used the calculator: 4',)))
            else:
                assert body['tool_choice'] == 'auto'
                assert outputs[-1]['call_id'] == 'call_1'
                assert json.loads(outputs[-1]['output']) == {'ok': True, 'data': {'value': 4}}
                items = stream_events(chunks=('Calculated: 4',))
        else:
            assert 'tool_choice' not in body and 'parallel_tool_calls' not in body
            assert 'You have no tools' in body['instructions']
            items = stream_events(chunks=('Text answer: 4',))
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=SSE(items))

    monkeypatch.setattr('ai_agent_pro.presentation.workspace.OpenAIProvider', lambda key: provider_with(handler))
    at.sidebar.radio[0].set_value('Chat').run()
    at.multiselect[0].select('calculator').run()
    at.chat_input[0].set_value('Use the calculator for 2+2').run()
    finish_chat(at)
    assert len(requests) == 2
    job = at.session_state['_chat_job']
    assert job.snapshot().state == 'Completed'
    assert job.snapshot().tools[0]['data'] == {'value': 4}
    record = repo.runs(agent.id)[0]
    assert record['requests'] == 2
    assert [e['state'] for e in events(repo, run_id=record['id']) if e['stage'] == 'tool'] == ['started', 'completed']

    # Deselecting removes schemas even though the host authority still has a
    # grant from the previous message. No automatic approval on submission.
    at.multiselect[0].unselect('calculator').run()
    at.chat_input[0].set_value('What is 2+2?').run()
    finish_chat(at)
    assert len(requests) == 3 and 'tools' not in requests[-1]
    assert at.session_state['_chat_job'].snapshot().tools == ()
