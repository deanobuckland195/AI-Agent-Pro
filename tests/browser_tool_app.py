"""Test-only SDK HTTP/SSE mocks; real engine, ToolExecutor, UI and SQLite."""
import asyncio
import json
import os
from pathlib import Path

import httpx
import openai

from ai_agent_pro import ui
from ai_agent_pro.presentation import workspace
from ai_agent_pro.provider import OpenAIProvider
from test_tool_calling import sse_tool_events
from test_streaming import encode_event, stream_events


class Wire(httpx.AsyncByteStream):
    def __init__(self, items, hold=False):
        self.items, self.hold = items, hold

    async def __aiter__(self):
        for event in self.items:
            if self.hold and event['type'] == 'response.completed':
                while not (Path(os.environ['AI_AGENT_PRO_BROWSER_CONTROL']) / 'release').exists():
                    await asyncio.sleep(.02)
            yield encode_event(event)
            await asyncio.sleep(.01)

    async def aclose(self):
        pass


def handler(request):
    body = json.loads(request.content)
    assert body['store'] is False and body['stream'] is True
    assert 'previous_response_id' not in body
    assert all(t['name'] == 'calculator__v1_0_0' for t in body.get('tools', []))
    prompt = next(m['content'] for m in body['input'] if m.get('role') == 'user').lower()
    count = sum(m.get('type') == 'function_call_output' for m in body['input'])
    if not count or ('sequential' in prompt and count < 2):
        items = sse_tool_events(arguments='{' if 'malformed' in prompt else '{"expression":"2+2"}')
        if count:
            # Each API request emits a distinct call ID.
            items = json.loads(json.dumps(items).replace('call_1', 'call_2'))
        if 'partial' in prompt:
            items = sse_tool_events(done=False)
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, stream=Wire(items))
    assert json.loads(body['input'][-1]['output'])['data'] == {'value':4}
    return httpx.Response(200, headers={'content-type':'text/event-stream'},
        stream=Wire(stream_events(chunks=('### Calculated answer\n\n', '**4**\n\nReady to finish. ')), hold=True))


def factory(**kwargs):
    return openai.AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class OfflineProvider(OpenAIProvider):
    def __init__(self, key):
        super().__init__('offline-test-key', client_factory=factory)


if __name__ == '__main__':
    workspace.OpenAIProvider = OfflineProvider
    ui.api_key = lambda: 'offline-test-key'
    ui.main()
