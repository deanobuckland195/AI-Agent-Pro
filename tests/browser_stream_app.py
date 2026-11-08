"""Offline browser-test launcher. Never imported by the real application."""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

from ai_agent_pro.config import AppError
from ai_agent_pro.provider import OpenAIProvider, Completion, ResponseStatusError
from ai_agent_pro import ui

CODE = 'def greet(name):\n    message = f"Hello, {name}"\n\n    return message'
SECOND_CODE = 'window.__codeRan = true;\nconsole.log("display only");'
PREFIX = '### Live answer\n\nFirst progressive text \n\n'
SUFFIX = ('\n\n```python\n' + CODE + '\n```\n\n```javascript\n' +
          SECOND_CODE + '\n```\n\nFinished answer.')


async def offline_complete(self, *, messages, on_delta=None, **kwargs):
    prompt = str(messages[-1]['content']).lower()
    if on_delta:
        for part in [PREFIX, 'sk-', 'private', 'abcdefgh', ' ', '\n']:
            on_delta(part)
            await asyncio.sleep(.03)
        while not (Path(os.environ['AI_AGENT_PRO_BROWSER_CONTROL']) / 'release').exists():
            await asyncio.sleep(.03)
        if 'disconnect' in prompt:
            raise AppError('The connection was lost during generation. No automatic retry was made.')
        if 'timeout' in prompt:
            raise asyncio.TimeoutError
        if 'output limit' in prompt:
            limit = kwargs['max_output_tokens']
            raise ResponseStatusError(SimpleNamespace(status='incomplete',
                incomplete_details=SimpleNamespace(reason='max_output_tokens'),
                usage=SimpleNamespace(output_tokens=limit)), limit)
        for part in [SUFFIX[:20], SUFFIX[20:]]:
            on_delta(part)
            await asyncio.sleep(.03)
    return Completion(PREFIX + 'sk-privateabcdefgh\n' + SUFFIX, 20, 10)


if __name__ == '__main__':
    OpenAIProvider.complete = offline_complete
    ui.api_key = lambda: 'offline-browser-key'
    ui.main()
