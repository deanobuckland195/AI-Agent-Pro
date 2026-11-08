"""Official OpenAI SDK adapter. Controlled function calls; no implicit retries or response storage."""
from dataclasses import dataclass
import asyncio
import json
import httpx
import openai
from ai_agent_pro.config import AppError


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str
    call_id: str


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    calls: tuple[FunctionCall, ...] = ()
    output: tuple[dict, ...] = ()


class ResponseStatusError(AppError):
    """Safe terminal diagnostics, without retaining partial text or raw API errors."""
    def __init__(self, response, output_limit):
        status = getattr(response, 'status', None)
        self.api_status = status if status in ('incomplete', 'failed', 'cancelled', 'in_progress', 'queued') else 'unknown'
        reason = getattr(getattr(response, 'incomplete_details', None), 'reason', None)
        self.reason = reason if reason in ('max_output_tokens', 'content_filter') else (
            'not_provided' if reason is None else 'unrecognized')
        count = getattr(getattr(response, 'usage', None), 'output_tokens', None)
        self.reported_output_tokens = count if type(count) is int and count >= 0 else None
        if self.api_status == 'incomplete' and self.reason == 'max_output_tokens':
            message = f'Output limit reached: the model response was incomplete at the configured {output_limit:,}-token output cap.'
        elif self.api_status == 'incomplete' and self.reason == 'content_filter':
            message = 'The model response was incomplete because a content filter stopped generation.'
        elif self.api_status == 'incomplete':
            message = 'The model response was incomplete; no recognized termination reason was provided.'
        elif self.api_status == 'cancelled':
            message = 'The API response was cancelled.'
        else:
            message = 'The model response failed to complete.'
        diagnostics = f'API status: {self.api_status}; termination reason: {self.reason}; requested output cap: {output_limit:,} tokens'
        if self.reported_output_tokens is not None:
            diagnostics += f'; reported output tokens: {self.reported_output_tokens:,}'
        super().__init__(f'{message} {diagnostics}. No partial answer was saved. No automatic retry was made.')


class OpenAIProvider:
    def __init__(self, api_key, client_factory=openai.AsyncOpenAI):
        if not isinstance(api_key, str) or not api_key.strip():
            raise AppError("Configure OPENAI_API_KEY before sending a message.")
        self._key = api_key.strip()
        self._factory = client_factory

    async def complete(self, *, model, instructions, messages, max_output_tokens, timeout, on_delta=None, tools=None, tool_choice='auto'):
        if tool_choice not in ('auto', 'required') or (tool_choice == 'required' and not tools):
            raise AppError('A required tool call needs explicitly approved tools.')
        try:
            async with asyncio.timeout(timeout):
                async with self._factory(api_key=self._key, base_url="https://api.openai.com/v1", max_retries=0, timeout=timeout) as client:
                    arguments = dict(model=model, instructions=instructions, input=messages,
                                     max_output_tokens=max_output_tokens, store=False)
                    if tools:
                        arguments.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=False)
                    stream_calls, arguments_done = {}, {}
                    if on_delta is None:
                        response = await client.responses.create(**arguments)
                    else:
                        length, completed = 0, False
                        async with client.responses.stream(**arguments) as stream:
                            async for event in stream:
                                if event.type == 'response.output_text.delta':
                                    length += len(event.delta)
                                    if length > 100000:
                                        raise AppError('The model response exceeded the local size limit.')
                                    on_delta(event.delta)
                                elif event.type == 'response.function_call_arguments.done':
                                    arguments_done[event.item_id] = event.arguments
                                elif event.type == 'response.output_item.done' and event.item.type == 'function_call':
                                    stream_calls[event.item.call_id] = event.item.model_dump(exclude_none=True, exclude={'parsed_arguments'})
                                elif event.type == 'response.function_call_arguments.delta':
                                    # Never execute or display argument deltas. Bound them before
                                    # the SDK's final response is trusted.
                                    length += len(event.delta)
                                    if length > 100000:
                                        raise AppError('The model response exceeded the local size limit.')
                                elif event.type == 'response.completed':
                                    completed = True
                                elif event.type == 'error':
                                    raise AppError('The model stream failed. No automatic retry was made.')
                                elif event.type in ('response.failed', 'response.incomplete'):
                                    raise ResponseStatusError(event.response, max_output_tokens)
                            if not completed:
                                raise AppError('The connection ended before the model response completed. No automatic retry was made.')
                            response = await stream.get_final_response()
            if response.status != "completed":
                raise ResponseStatusError(response, max_output_tokens)
            calls, output = [], []
            for item in response.output:
                if item.type not in ('message', 'reasoning', 'function_call'):
                    raise AppError('The model returned an unsupported action.')
                if item.type == 'function_call':
                    if (not tools or getattr(item, 'status', None) not in (None, 'completed') or
                            type(item.name) is not str or len(item.name) > 64 or
                            type(item.call_id) is not str or not item.call_id or len(item.call_id) > 128 or
                            type(item.arguments) is not str or len(item.arguments.encode()) > 65536):
                        raise AppError('The model returned an invalid or unapproved tool call.')
                    if on_delta is not None and (stream_calls.get(item.call_id) != item.model_dump(exclude_none=True, exclude={'parsed_arguments'})
                            or arguments_done.get(item.id) != item.arguments):
                        raise AppError('The model stream contained an incomplete tool call.')
                    calls.append(FunctionCall(item.name, item.arguments, item.call_id))
                if tools:
                    output.append(item.model_dump(exclude_none=True, exclude={'parsed_arguments'}))
            if len(calls) > 1 or len({c.call_id for c in calls}) != len(calls):
                raise AppError('Only one sequential tool call per model response is permitted.')
            if len(json.dumps(output, ensure_ascii=True).encode()) > 131072:
                raise AppError('The model response exceeded the local size limit.')
            text = response.output_text
            if not calls and (not text or not text.strip()):
                raise AppError("The model returned no text (possibly a refusal). Rephrase your request.")
            if len(text) > 100000:
                raise AppError("The model response exceeded the local size limit.")
            usage = response.usage
            return Completion(text, usage.input_tokens if usage else None, usage.output_tokens if usage else None, tuple(calls), tuple(output))
        except (openai.APITimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            raise AppError("The OpenAI request timed out. No automatic retry was made.") from None
        except openai.AuthenticationError:
            raise AppError("OpenAI rejected the API credentials. Check your private configuration.") from None
        except openai.RateLimitError:
            raise AppError("OpenAI rate or quota limit reached. Check usage and try later.") from None
        except openai.APIConnectionError:
            raise AppError("Could not connect to OpenAI. Check your network.") from None
        except openai.APIStatusError:
            raise AppError("OpenAI rejected the request or is unavailable. Check model access and settings.") from None
        except httpx.HTTPError:
            raise AppError('The connection was lost during generation. No automatic retry was made.') from None
