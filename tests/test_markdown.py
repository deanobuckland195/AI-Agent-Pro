"""Assistant Markdown stays a presentation concern; stored content is unchanged."""
from unittest.mock import AsyncMock
import pytest

from ai_agent_pro.config import Agent
from ai_agent_pro.presentation.markdown import safe_markdown
from ai_agent_pro.provider import Completion
from test_ui import launch, finish_chat


FORMATTED_REPLY = '''### Formatted answer

This is **bold text**.

- First bullet
- Second bullet

1. First step
2. Second step

```python
print("<example>")
```
'''

UNTRUSTED_REPLY = '''<script>window.__modelExecuted = true</script>

<img src="https://untrusted.invalid/pixel" onerror="window.__modelExecuted = true">

<iframe srcdoc="<script>parent.__modelExecuted = true</script>"></iframe>

<svg onload="window.__modelExecuted = true"></svg>

<a href="javascript:window.__modelExecuted=true">HTML link</a>

[Unsafe link](javascript:window.__modelExecuted=true)

[Data link](data:text/html,unsafe)

![Tracking image](https://untrusted.invalid/markdown-pixel)

[Reference payload][payload]

[payload]: data:text/html,unsafe
'''


@pytest.mark.parametrize('target', [
    'javascript:alert%281%29', 'JaVaScRiPt:alert%281%29',
    'jav&#97;script:alert%281%29', 'javascript&#58;alert%281%29',
    'data:text/html,unsafe', 'vbscript:unsafe', 'file:///etc/passwd', '//untrusted.invalid',
])
def test_unsafe_destinations_become_inert_labels(target):
    assert safe_markdown(f'[Label]({target})').strip() == '`Label`'
    assert safe_markdown(f'[Label][ref]\n\n[ref]: {target}').strip() == '`Label`'


def test_permitted_reference_links_are_inlined_and_isolated():
    assert safe_markdown('[Docs][ref]\n\n[ref]: https://example.com').strip() == '[Docs](https://example.com)'
    assert 'https://example.com' not in safe_markdown('[ref]')


def test_code_examples_preserve_literal_links_and_html():
    code = '```html\n<script>alert(1)</script>\n[x](javascript:alert)\n```'
    assert safe_markdown(code).strip() == code


def test_markdown_new_response_and_reloaded_history(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Markdown assistant', '', 'Be helpful')
    repo.save_agent(agent)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    reply = FORMATTED_REPLY + '\n' + UNTRUSTED_REPLY
    completion = AsyncMock(return_value=Completion(reply, 10, 5))
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', completion)
    at.sidebar.radio[0].set_value('Chat').run()
    prompt = '**Keep user input literal** <script>user()</script>'
    at.chat_input[0].set_value(prompt).run()
    finish_chat(at)

    for _ in range(2):
        assert not at.exception
        rendered = next(m for m in at.markdown if m.value.startswith('### Formatted answer'))
        assert not rendered.proto.allow_html
        assert '**bold text**' in rendered.value
        assert '```python\nprint("<example>")\n```' in rendered.value
        assert '`Data link`' in rendered.value
        assert 'markdown-pixel' not in rendered.value
        assert prompt in [t.value for t in at.text]
        cid = repo.conversations(agent.id)[0]['id']
        messages = repo.messages(cid)
        assert [(m['role'], m['content']) for m in messages] == [
            ('user', prompt), ('assistant', reply)]
        assert completion.await_count == 1
        at.run()
