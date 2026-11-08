"""Multiline secret redaction must preserve original source line locations."""
import pytest

from ai_agent_pro.tools.filesystem import read_file
from test_tools import tools, call, allow


@pytest.mark.parametrize('ending', ['\r', '\n', '\r\n', '\v', '\f', '\x85', '\u2028', '\u2029'])
@pytest.mark.parametrize('label', ['api_key:', 'password =', 'Authorization:'])
def test_multiline_redaction_retains_original_lines(tools, tmp_path, ending, label):
    raw = f'Before{ending}{label}{ending}private-test-value{ending}Last line{ending}'
    path = tmp_path / 'notes.txt'
    path.write_bytes(raw.encode('utf-8'))
    tools.permissions.approve_files('agent-a', ['notes.txt'])
    allow(tools, 'project_search')
    allow(tools, 'project_read')
    found = call(tools, 'project_search', {'query':'Last line','paths':[],'max_results':5})
    assert found.ok and found.data['matches'][0]['line'] == 4
    selected = call(tools, 'project_read', {'path':'notes.txt','start_line':4,'line_count':1})
    assert selected.ok and selected.data['lines'] == ['Last line']
    text, _ = read_file(str(tmp_path), tools.permissions.identity, 'notes.txt')
    assert 'private-test-value' not in text and '[REDACTED]' in text
    assert text.count(ending) == raw.count(ending)
    assert len(text.splitlines()) == len(raw.splitlines())
    assert path.read_bytes() == raw.encode('utf-8')


def test_mixed_endings_multiple_and_adjacent_secrets_preserve_locations(tools, tmp_path):
    raw = 'Before\rpassword:\r\rprivate-one\r\nMiddle\napi_key:\r\nprivate-two\rLast line\n'
    (tmp_path / 'notes.txt').write_bytes(raw.encode())
    tools.permissions.approve_files('agent-a', ['notes.txt'])
    allow(tools, 'project_search')
    expected = next(i for i, line in enumerate(raw.splitlines(), 1) if line == 'Last line')
    result = call(tools, 'project_search', {'query':'Last line','paths':[],'max_results':5})
    assert result.ok and result.data['matches'][0]['line'] == expected
    text, _ = read_file(str(tmp_path), tools.permissions.identity, 'notes.txt')
    assert 'private-one' not in text and 'private-two' not in text
    assert len(text.splitlines()) == len(raw.splitlines())
