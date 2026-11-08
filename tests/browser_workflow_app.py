"""Offline workflow browser fixture; real SDK/SSE, no paid remote calls."""
import asyncio
import json
import os
from pathlib import Path
import httpx
import openai
from ai_agent_pro import ui
from ai_agent_pro.presentation import workflows
from ai_agent_pro.provider import OpenAIProvider
from ai_agent_pro.practice import PracticeWorkspace,NAME
from test_streaming import stream_events,encode_event
from test_tool_calling import sse_tool_events


class Wire(httpx.AsyncByteStream):
    def __init__(self,items,hold=False):self.items,self.hold=items,hold
    async def __aiter__(self):
        for item in self.items:
            if self.hold and item['type']=='response.completed':
                while not (Path(os.environ['AI_AGENT_PRO_BROWSER_CONTROL'])/'release').exists():
                    await asyncio.sleep(.02)
            yield encode_event(item)
            await asyncio.sleep(.01)
    async def aclose(self):pass


def handler(request):
    body=json.loads(request.content)
    assert body['store'] is False and body['stream'] is True
    payload=json.loads(next(m['content'] for m in body['input'] if m.get('role')=='user'))
    goal=payload['goal'].lower()
    ordinal=payload['handoff']['to_step'] if payload['handoff'] else 0
    output=any(m.get('type')=='function_call_output' for m in body['input'])
    if ordinal==0 and ('approve' in goal or 'deny' in goal) and not output:
        assert [t['name'] for t in body['tools']]==[NAME]
        items=sse_tool_events(arguments=json.dumps(dict(path='example.txt',content='Reviewed workflow practice text.\n')))
        items=json.loads(json.dumps(items).replace('calculator__v1_0_0',NAME))
    else:
        assert 'tools' not in body
        text='invalid output' if 'invalid' in goal else json.dumps(dict(schema_version='1.0.0',summary='Stage '+str(ordinal+1),deliverable='Reviewed output '+str(ordinal+1),risks=[]))
        items=stream_events(chunks=(text,))
    return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Wire(items,hold='cancel' in goal))


def factory(**kwargs):
    return openai.AsyncOpenAI(**kwargs,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class OfflineProvider(OpenAIProvider):
    def __init__(self,key):super().__init__('offline-test-key',client_factory=factory)


if __name__=='__main__':
    workflows.OpenAIProvider=OfflineProvider
    workflows.practice_workspace=lambda:PracticeWorkspace(Path(os.environ['AI_AGENT_PRO_BROWSER_CONTROL']))
    ui.api_key=lambda:'offline-test-key'
    ui.main()
