import asyncio
import json
import httpx
import openai
import pytest
from ai_agent_pro.config import AppError
from ai_agent_pro.provider import OpenAIProvider


def response_body(status='completed', text='A useful answer', usage=True):
    return {'id': 'resp_test', 'object': 'response', 'created_at': 1,
            'status': status, 'model': 'gpt-4.1-mini',
            'output': [{'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}],
            'usage': {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30} if usage else None}


def invoke(provider):
    return asyncio.run(provider.complete(model='gpt-4.1-mini', instructions='System policy',
        messages=[{'role': 'user', 'content': 'Hi'}], max_output_tokens=100, timeout=2))


def provider_with(handler):
    def factory(**kwargs):
        assert kwargs['max_retries'] == 0
        assert kwargs['base_url'] == 'https://api.openai.com/v1'
        return openai.AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return OpenAIProvider('sk-test-not-real', client_factory=factory)


def test_real_sdk_serialization():
    def handler(request):
        body = json.loads(request.content)
        assert body['store'] is False
        assert body['max_output_tokens'] == 100
        assert body['instructions'] == 'System policy'
        assert body['input'][0]['role'] == 'user'
        assert 'tools' not in body
        assert 'previous_response_id' not in body
        return httpx.Response(200, json=response_body())
    result = invoke(provider_with(handler))
    assert result.text == 'A useful answer'
    assert (result.input_tokens, result.output_tokens) == (20, 10)


@pytest.mark.parametrize('status,expected', [(401, 'credentials'), (429, 'quota'),
                                           (400, 'rejected'), (500, 'unavailable')])
def test_sdk_errors_are_safe_and_not_retried(status, expected):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={'error': {'message': 'sk-privateabcdefgh', 'type': 'test'}})
    with pytest.raises(AppError, match=expected) as error:
        invoke(provider_with(handler))
    assert 'sk-private' not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize('exception,expected', [(httpx.ConnectError, 'connect'), (httpx.ReadTimeout, 'timed out')])
def test_transport_failures(exception, expected):
    def handler(request):
        raise exception('private request', request=request)
    with pytest.raises(AppError, match=expected):
        invoke(provider_with(handler))


@pytest.mark.parametrize('body,expected', [(response_body(status='incomplete'), 'incomplete'),
                                          (response_body(text=''), 'no text')])
def test_bad_responses(body, expected):
    with pytest.raises(AppError, match=expected):
        invoke(provider_with(lambda request: httpx.Response(200, json=body)))


def test_usage_unavailable():
    result = invoke(provider_with(lambda request: httpx.Response(200, json=response_body(usage=False))))
    assert result.input_tokens is None


def test_missing_key():
    with pytest.raises(AppError, match='OPENAI_API_KEY'):
        OpenAIProvider('')
