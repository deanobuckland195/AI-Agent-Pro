"""Offline approval browser harness; no real credentials or remote traffic."""
import json
import os
from pathlib import Path
import httpx
import openai

from ai_agent_pro import ui
from ai_agent_pro.presentation import workspace
from ai_agent_pro.provider import OpenAIProvider
from ai_agent_pro.practice import PracticeWorkspace, NAME
from test_streaming import SSE, stream_events
from test_tool_calling import sse_tool_events


def handler(request):
    body = json.loads(request.content)
    assert body['store'] is False and body['stream'] is True
    if not any(m.get('type') == 'function_call_output' for m in body['input']):
        assert body['tool_choice'] == 'required'
        assert [t['name'] for t in body['tools']] == [NAME]
        items = sse_tool_events(arguments=json.dumps(dict(path='example.txt', content='Approved browser text.\n')))
        items = json.loads(json.dumps(items).replace('calculator__v1_0_0', NAME))
    else:
        assert 'tools' not in body
        assert body['input'][-1]['call_id'] == 'call_1'
        items = stream_events(chunks=('Approved change completed.',))
    return httpx.Response(200, headers={'content-type':'text/event-stream'}, stream=SSE(items))


def factory(**kwargs):
    return openai.AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class OfflineProvider(OpenAIProvider):
    def __init__(self, key):
        super().__init__('offline-test-key', client_factory=factory)


if __name__ == '__main__':
    workspace.OpenAIProvider = OfflineProvider
    workspace.practice_workspace = lambda: PracticeWorkspace(Path(os.environ['AI_AGENT_PRO_BROWSER_CONTROL']))
    ui.api_key = lambda: 'offline-test-key'
    ui.main()
