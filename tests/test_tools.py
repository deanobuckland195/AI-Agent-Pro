"""Stage 4A tool contracts, deny-by-default grants and real isolated operations."""
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from ai_agent_pro.tools.builtins import registry, obj, text
from ai_agent_pro.tools.contracts import ToolFailure, ToolRegistry
from ai_agent_pro.tools.executor import ToolExecutor
from ai_agent_pro.tools.permissions import ToolPermissions
from test_rag_retrieval import kb, indexed


@pytest.fixture
def tools(tmp_path, repo):
    permissions = ToolPermissions(tmp_path, repo.path)
    try:
        yield ToolExecutor(permissions)
    finally:
        permissions.close()


def call(tools, name, arguments, agent='agent-a', version='1.0.0', **options):
    return asyncio.run(tools.execute(agent, name, version, arguments, **options))


def allow(tools, name, agent='agent-a'):
    tools.permissions.allow(agent, name)


def test_registry_versions_and_catalog_are_not_mutable():
    tools = registry()
    assert {t['name'] for t in tools.catalog()} == {
        'calculator', 'python_inspect', 'project_read', 'project_search', 'knowledge_search'}
    definition = tools.get('calculator', '1.0.0')
    with pytest.raises(ValueError, match='overwritten'):
        tools.register(definition)
    tools.register(replace(definition, version='1.1.0'))
    definition.input_schema['properties'].clear()
    catalog = tools.catalog()
    catalog[0]['input_schema']['properties'].clear()
    assert 'expression' in tools.get('calculator', '1.0.0').input_schema['properties']
    assert tools.get('calculator', '1.1.0').version == '1.1.0'
    assert all(t['read_only'] for t in catalog)


@pytest.mark.parametrize('change', [
    {'name': '__import__'}, {'version': 'latest'}, {'permission': 'shell'},
    {'handler': 'os:system'}, {'handler': 'other_project.tool:run'},
    {'timeout_seconds': 0}, {'timeout_seconds': float('inf')}, {'timeout_seconds': True},
    {'result_limit_bytes': 0}, {'result_limit_bytes': 65537},
    {'input_schema': {'type': 'object'}},
    {'input_schema': obj(values={'type': 'array', 'items': text(5)})},
    {'input_schema': obj(value={'type': 'string'})}, {'input_schema': obj(value={})},
    {'input_schema': obj(value={'type':'string', 'maxLength':10, 'not':{'$ref':'https://untrusted.invalid/schema'}})},
    {'input_schema': obj(value={'type':'string', 'maxLength':10, 'pattern':'(a+)+$'})},
])
def test_registry_rejects_unsafe_or_unbounded_definitions(change):
    with pytest.raises((ValueError, TypeError)):
        ToolRegistry().register(replace(registry().get('calculator', '1.0.0'), **change))


def test_no_grant_by_default_and_no_agent_or_version_inheritance(tools):
    assert call(tools, 'calculator', {'expression': '2+2'}).error_code == 'permission_denied'
    allow(tools, 'calculator')
    assert call(tools, 'calculator', {'expression': '2+2'}).data == {'value': 4}
    assert call(tools, 'calculator', {'expression': '2+2'}, agent='agent-b').error_code == 'permission_denied'
    tools.registry.register(replace(tools.registry.get('calculator', '1.0.0'), version='1.1.0'))
    assert call(tools, 'calculator', {'expression': '2+2'}, version='1.1.0').error_code == 'permission_denied'
    tools.permissions.revoke('agent-a', 'calculator')
    assert call(tools, 'calculator', {'expression': '2+2'}).error_code == 'permission_denied'


@pytest.mark.parametrize('name,version', [('missing', '1.0.0'), ('calculator', 'latest'),
                                         ('sk-privateabcdefgh', '1.0.0'), ([], {})])
def test_unknown_tool_is_safe_without_echoing_request(tools, name, version):
    result = call(tools, name, {}, version=version)
    assert result.error_code == 'unavailable'
    assert 'private' not in str(result)


@pytest.mark.parametrize('arguments', [
    {}, {'expression': '2', 'agent_id': 'admin'}, {'expression': '2', 'approved': True},
    {'expression': '2', 'handler': 'os:system'}, {'expression': '2', 'root': '/'},
    {'expression': True}, {'expression': 2}, {'expression': ''}, {'expression': '1' * 257},
    {'expression': float('nan')}, {'expression': Path('/')}, [], None,
    {'expression': '1', 'x'*129: 'not allowed'},
])
def test_invalid_inputs_are_rejected_without_worker(tools, arguments, monkeypatch):
    allow(tools, 'calculator')
    async def forbidden(*args):
        pytest.fail('Invalid input must not start a worker')
    monkeypatch.setattr(tools, '_process', forbidden)
    result = call(tools, 'calculator', arguments)
    assert result.error_code == 'invalid_input' and result.data is None


@pytest.mark.parametrize('expression,value', [('2+3*4', 14), ('-(3+4)/2', -3.5),
    ('2**10', 1024), ('7//2', 3), ('7%2', 1), ('1.5 + .5', 2), ('+2', 2)])
def test_calculator_real_worker(tools, expression, value):
    allow(tools, 'calculator')
    result = call(tools, 'calculator', {'expression': expression})
    assert result.ok and result.data == {'value': value}


@pytest.mark.parametrize('expression', [
    '__import__("os").system("touch /tmp/unsafe")', 'open("API.key").read()',
    '(1).__class__', '[x for x in range(10)]', 'True+1', '"text"',
    '2**1000000', '10**(10**10)', '(-1)**0.5', '1e309', '1/0',
    '1e100 * 10', '1 << 100', 'lambda: 1', '(x := 1)',
    '('*80+'1'+')'*80, '+'.join(['1']*60),
])
def test_calculator_rejects_code_and_extreme_work(tools, expression):
    allow(tools, 'calculator')
    result = call(tools, 'calculator', {'expression': expression})
    # Parentheses alone do not increase AST depth; nested unary expressions do.
    if expression == '('*80+'1'+')'*80:
        assert result.ok and result.data == {'value': 1}
    else:
        assert result.error_code == 'invalid_expression' and result.data is None


def test_source_inspection_never_executes_and_omits_code_and_secrets(tools, tmp_path):
    allow(tools, 'python_inspect')
    marker = tmp_path / 'MUST_NOT_EXIST'
    source = f'''import os
from pathlib import Path
Path({str(marker)!r}).write_text("executed")
secret = "sk-privateabcdefgh"
class Example:
    async def run(self):
        return 1
'''
    result = call(tools, 'python_inspect', {'source': source})
    assert result.ok and result.data['syntax_valid']
    assert result.data['imports'] == ['os', 'pathlib']
    assert result.data['classes'][0]['name'] == 'Example'
    assert result.data['functions'][0]['asynchronous']
    assert not marker.exists() and 'private' not in str(result)


def test_source_syntax_error_does_not_echo_source(tools):
    allow(tools, 'python_inspect')
    result = call(tools, 'python_inspect', {'source': 'def broken(sk-privateabcdefgh):\n'})
    assert result.ok and result.data['syntax_valid'] is False
    assert result.data['syntax_error']['line'] == 1
    assert 'private' not in str(result)


def test_source_limits_are_real(tools):
    allow(tools, 'python_inspect')
    assert call(tools, 'python_inspect', {'source': 'x=1\n'*2001}).error_code == 'resource_limit'
    assert call(tools, 'python_inspect', {'source': 'x=1;'*1500}).error_code == 'resource_limit'
    assert call(tools, 'python_inspect', {'source': '#'+'🙂'*10000}).error_code == 'invalid_input'
    result = call(tools, 'python_inspect', {'source': '\n'.join(f'def f{i}(): pass' for i in range(65))})
    assert result.ok and len(result.data['functions']) == 64 and result.data['truncated']


def test_pure_tools_cannot_use_permissions_to_access_files(tools):
    allow(tools, 'calculator')
    assert call(tools, 'project_read', {'path': 'README.md', 'start_line': 1, 'line_count': 5}).error_code == 'permission_denied'
    assert call(tools, 'knowledge_search', {'query': 'London', 'max_results': 5}).error_code == 'permission_denied'


def test_project_read_search_requires_both_grant_and_manifest(tools, tmp_path):
    (tmp_path / 'notes.txt').write_text('Hello world\nsecond line\nHello again\n')
    arguments = {'path': 'notes.txt', 'start_line': 2, 'line_count': 1}
    allow(tools, 'project_read')
    assert call(tools, 'project_read', arguments).error_code == 'permission_denied'
    tools.permissions.approve_files('agent-a', ['notes.txt'])
    result = call(tools, 'project_read', arguments)
    assert result.ok and result.data['lines'] == ['second line'] and result.data['truncated']
    search = {'query': 'HELLO', 'paths': [], 'max_results': 20}
    assert call(tools, 'project_search', search).error_code == 'permission_denied'
    allow(tools, 'project_search')
    result = call(tools, 'project_search', search)
    assert result.ok and [m['line'] for m in result.data['matches']] == [1, 3]
    assert result.data['files_searched'] == 1
    assert call(tools, 'project_search', dict(search, query='Hello.*')).data['matches'] == []


@pytest.mark.parametrize('path', [
    '../outside.txt', '/etc/passwd', 'nested/../../other.txt', 'nested/./file.py',
    'nested//file.py', 'C:/secret.txt', '\\\\server\\file.txt', '%2e%2e/outside.txt',
    'notes.txt\x00', 'notes.txt\n', './notes.txt', ' notes.txt', 'notes.txt ',
    'API.key', 'api.KEY', 'secrets.txt', 'credentials.py', 'passwords.md', 'api_key.py',
    '.env', '.env.txt', '.git/config', '.streamlit/secrets.toml', '.codex/settings.md',
    'data/notes.txt', 'venv/main.py', 'private/main.py', 'config.toml', 'config.json',
    'service.yaml', 'key.pem', 'id_rsa.txt', '*.py',
    'secret-backup/notes.md', 'credentials-v1/main.py', 'keys.py',
])
def test_sensitive_and_traversal_paths_cannot_even_be_approved(tools, path):
    with pytest.raises(ToolFailure, match='not approved'):
        tools.permissions.approve_files('agent-a', [path])


def test_symlink_and_hardlink_escapes_and_swapped_files_are_denied(tools, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside.txt')
    outside.write_text('outside project')
    try:
        (tmp_path / 'link.txt').symlink_to(outside)
        with pytest.raises(ToolFailure):
            tools.permissions.approve_files('agent-a', ['link.txt'])
        (tmp_path / 'dirlink').symlink_to(tmp_path.parent, target_is_directory=True)
        with pytest.raises(ToolFailure):
            tools.permissions.approve_files('agent-a', ['dirlink/' + outside.name])
        (tmp_path / 'hard.txt').hardlink_to(outside)
        with pytest.raises(ToolFailure):
            tools.permissions.approve_files('agent-a', ['hard.txt'])
        (tmp_path / 'safe.txt').write_text('safe')
        tools.permissions.approve_files('agent-a', ['safe.txt'])
        (tmp_path / 'safe.txt').unlink()
        (tmp_path / 'safe.txt').symlink_to(outside)
        allow(tools, 'project_read')
        result = call(tools, 'project_read', {'path': 'safe.txt', 'start_line': 1, 'line_count': 1})
        assert result.error_code == 'permission_denied' and 'outside' not in str(result)
    finally:
        outside.unlink()


def test_missing_binary_fifo_and_large_files_fail_safely(tools, tmp_path):
    import os
    for name in ['missing.txt']:
        with pytest.raises(ToolFailure):
            tools.permissions.approve_files('agent-a', [name])
    (tmp_path / 'binary.txt').write_bytes(b'\xff\x00')
    (tmp_path / 'huge.txt').write_bytes(b'x'*262145)
    os.mkfifo(tmp_path / 'pipe.txt')
    for name in ['binary.txt', 'huge.txt', 'pipe.txt']:
        with pytest.raises(ToolFailure):
            tools.permissions.approve_files('agent-a', [name])


def test_project_secret_redaction_precedes_line_selection_and_search(tools, tmp_path):
    raw = 'First\napi_key:\nprivateabcdefgh\nsk-privateabcdefgh\nLast\n'
    (tmp_path / 'safe.txt').write_text(raw)
    tools.permissions.approve_files('agent-a', ['safe.txt'])
    allow(tools, 'project_read')
    allow(tools, 'project_search')
    result = call(tools, 'project_read', {'path': 'safe.txt', 'start_line': 1, 'line_count': 50})
    assert result.ok and 'private' not in str(result) and '[REDACTED]' in str(result)
    assert result.data['lines'][4] == 'Last'
    assert call(tools, 'project_search', {'query': 'privateabcdefgh', 'paths': [], 'max_results': 20}).data['matches'] == []
    assert (tmp_path / 'safe.txt').read_text() == raw


def test_search_validates_all_paths_and_reports_truncation(tools, tmp_path):
    (tmp_path / 'safe.txt').write_text('hello\n'*30)
    tools.permissions.approve_files('agent-a', ['safe.txt'])
    allow(tools, 'project_search')
    args = {'query': 'hello', 'paths': ['safe.txt'], 'max_results': 2}
    result = call(tools, 'project_search', args)
    assert result.ok and len(result.data['matches']) == 2 and result.data['truncated']
    assert call(tools, 'project_search', dict(args, paths=['safe.txt', 'other.txt'])).error_code == 'permission_denied'
    assert call(tools, 'project_search', dict(args, max_results=2.0)).error_code == 'invalid_input'


def test_read_and_scan_size_limits(tools, tmp_path):
    files = []
    for index in range(5):
        name = f'file{index}.txt'
        (tmp_path / name).write_text('hello\n' + 'x'*230000)
        files.append(name)
    tools.permissions.approve_files('agent-a', files)
    allow(tools, 'project_search')
    assert call(tools, 'project_search', {'query': 'absent', 'paths': [], 'max_results': 20}).error_code == 'resource_limit'
    with pytest.raises(ValueError):
        tools.permissions.approve_files('agent-a', ['a.txt']*33)


def test_knowledge_is_offline_scoped_read_only_and_provenanced(tools, repo, agent, kb):
    did = indexed(kb)
    other = kb.repository.create_collection('Unassigned canary')
    hidden = kb.storage.upload(other, 'canary.txt', b'London secret canary')
    asyncio.run(kb.indexer.index(hidden))
    allow(tools, 'knowledge_search', agent.id)
    before = kb.indexer.embeddings.calls
    result = call(tools, 'knowledge_search', {'query': 'London deployment', 'max_results': 5}, agent=agent.id)
    assert result.ok and result.data['method'] == 'local_lexical'
    assert len(result.data['sources']) == 1 and result.data['sources'][0]['document_id'] == did
    assert result.data['sources'][0]['location'] and result.data['sources'][0]['generation_id']
    assert kb.indexer.embeddings.calls == before
    assert call(tools, 'knowledge_search', {'query': 'canary', 'max_results': 5}, agent=agent.id).data['sources'] == []
    allow(tools, 'knowledge_search')
    assert call(tools, 'knowledge_search', {'query': 'London', 'max_results': 5}).error_code == 'permission_denied'
    with repo.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM retrieval_usage').fetchone()[0] == 0


def test_knowledge_cannot_inject_collection_agent_or_sql(tools, agent, kb):
    indexed(kb)
    allow(tools, 'knowledge_search', agent.id)
    args = {'query': 'London', 'max_results': 5}
    for field, value in [('agent_id', 'other'), ('collection_ids', [kb.cid]), ('database', '/outside.sqlite')]:
        assert call(tools, 'knowledge_search', dict(args, **{field:value}), agent=agent.id).error_code == 'invalid_input'
    result = call(tools, 'knowledge_search', dict(args, query="'; DROP TABLE agents; --"), agent=agent.id)
    assert result.ok and result.data['sources'] == []
    assert kb.repository.assigned(agent.id) == [kb.cid]


def test_knowledge_no_indexes_no_matches_and_revocation(tools, repo, agent, kb):
    allow(tools, 'knowledge_search', agent.id)
    args = {'query': 'London', 'max_results': 5}
    assert call(tools, 'knowledge_search', args, agent=agent.id).data['sources'] == []
    indexed(kb)
    kb.repository.assign(agent.id, [])
    assert call(tools, 'knowledge_search', args, agent=agent.id).data['sources'] == []


def test_permission_root_and_database_cannot_escape(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    with pytest.raises(ValueError):
        ToolPermissions(project, project / '../outside.sqlite')
    linked = tmp_path / 'linked'
    linked.symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError):
        ToolPermissions(linked)


def test_oversized_source_is_rejected_without_starting_worker(tools, monkeypatch):
    allow(tools, 'python_inspect')
    async def forbidden(*args):
        pytest.fail('Oversized strings must not be serialized to a worker')
    monkeypatch.setattr(tools, '_process', forbidden)
    assert call(tools, 'python_inspect', {'source': 'x'*32769}).error_code == 'invalid_input'


def test_search_does_not_read_past_aggregate_byte_budget(tools, tmp_path, monkeypatch):
    from ai_agent_pro.tools.filesystem import search, MAX_TOTAL_BYTES
    import os
    files = []
    for i in range(5):
        name = f'part{i}.txt'
        (tmp_path / name).write_text('x'*230000)
        files.append(name)
    tools.permissions.approve_files('agent-a', files)
    allow(tools, 'project_search')
    definition = tools.registry.get('project_search', '1.0.0')
    context = tools.permissions.context(tools.permissions.authorize('agent-a', definition))
    read, sizes = os.read, []
    def measured(fd, size):
        result = read(fd, size)
        sizes.append(len(result))
        return result
    monkeypatch.setattr(os, 'read', measured)
    with pytest.raises(ToolFailure, match='resource limits'):
        search({'query':'absent','paths':[], 'max_results':20}, context)
    assert sum(sizes) <= MAX_TOTAL_BYTES
