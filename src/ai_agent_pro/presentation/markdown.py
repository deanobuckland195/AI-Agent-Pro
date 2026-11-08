"""Prepare untrusted Markdown without generating or enabling HTML."""
from html import unescape
from urllib.parse import urlsplit

import mistune
from mistune.renderers.markdown import MarkdownRenderer


class SafeMarkdownRenderer(MarkdownRenderer):
    def link(self, token, state):
        try:
            scheme = urlsplit(unescape(token['attrs']['url'])).scheme.lower()
        except ValueError:
            scheme = ''
        if scheme in ('http', 'https', 'mailto'):
            # Resolve references before discarding their definitions.
            return super().link({k: v for k, v in token.items() if k != 'label'}, state)
        # Unknown schemes and relative destinations become inert labels.
        return self.codespan({'raw': self.render_children(token, state)}, state)

    def image(self, token, state):
        # Generated Markdown must not silently fetch remote tracking images.
        return self.codespan({'raw': 'Image: ' + self.render_children(token, state)}, state)

    def render_referrences(self, state):
        # Permitted references have already been inlined.
        return ()


def safe_markdown(content):
    # A fresh parser isolates reference definitions between messages.
    return mistune.create_markdown(
        renderer=SafeMarkdownRenderer(), plugins=['table', 'strikethrough'])(content)
