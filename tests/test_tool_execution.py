"""Limits and fail-closed boundaries using actual disposable child processes."""
import asyncio
from dataclasses import replace
import sys
import threading

import pytest

from ai_agent_pro.tools.builtins import obj
from ai_agent_pro.tools.permissions import ToolBudget
from test_tools import tools, call, allow
from test_rag_retrieval import kb, indexed


def fake_worker(monkeypatch, script):
    """Only tests replace the fixed worker; no runtime custom-command interface."""
    original = asyncio.create_subprocess_exec
    children = []
    async def spawn(*args, **kwargs):
        assert args[1:5] == ('-I', '-B', '-m', 'ai_agent_pro.tools.worker')
        assert set(kwargs['env']) == {'PATH', 'LANG', 'PYTHONDONTWRITEBYTECODE'}
        process = await original(sys.executable, '-I', '-B', '-c', script, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    return children


def test_shared_call_and_result_budgets_are_enforced(tools):
    allow(tools, 'calculator')
    budget = ToolBudget(max_calls=1)
    assert call(tools, 'calculator', {'expression': '1'}, budget=budget).ok
    assert call(tools, 'calculator', {'expression': '2'}, budget=budget).error_code == 'budget_exhausted'
    tiny = ToolBudget(result_bytes=1)
    assert call(tools, 'calculator', {'expression': '1'}, budget=tiny).error_code == 'budget_exhausted'


@pytest.mark.parametrize('kwargs', [dict(max_calls=0), dict(max_calls=True), dict(seconds=0),
                                    dict(seconds=float('nan')), dict(result_bytes=0), dict(result_bytes=1048577)])
def test_budget_configuration_is_bounded(kwargs):
    with pytest.raises(ValueError):
        ToolBudget(**kwargs)


def test_real_process_deadline_kills_and_reaps_worker(tools, monkeypatch):
    children = fake_worker(monkeypatch, 'import time; time.sleep(30)')
    allow(tools, 'calculator')
    result = call(tools, 'calculator', {'expression': '1'}, budget=ToolBudget(seconds=.1))
    assert result.error_code == 'timeout' and result.elapsed_seconds < 2
    assert children and all(p.returncode is not None for p in children)


def test_cancel_event_before_start_and_during_execution(tools, monkeypatch):
    children = fake_worker(monkeypatch, 'import time; time.sleep(30)')
    allow(tools, 'calculator')
    cancel = threading.Event()
    cancel.set()
    assert call(tools, 'calculator', {'expression': '1'}, cancellation=cancel).error_code == 'cancelled'
    assert not children
    cancel.clear()
    async def scenario():
        task = asyncio.create_task(tools.execute('agent-a', 'calculator', '1.0.0', {'expression': '1'}, cancellation=cancel))
        while not children:
            await asyncio.sleep(.01)
        cancel.set()
        return await asyncio.wait_for(task, 2)
    result = asyncio.run(scenario())
    assert result.error_code == 'cancelled' and all(p.returncode is not None for p in children)


def test_coroutine_cancellation_propagates_after_reaping_child(tools, monkeypatch):
    children = fake_worker(monkeypatch, 'import time; time.sleep(30)')
    allow(tools, 'calculator')
    async def scenario():
        task = asyncio.create_task(tools.execute('agent-a', 'calculator', '1.0.0', {'expression': '1'}))
        while not children:
            await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert all(p.returncode is not None for p in children)


def test_stdout_is_bounded_and_worker_killed(tools, monkeypatch):
    children = fake_worker(monkeypatch, 'import sys,time; sys.stdout.write("x"*100000); sys.stdout.flush(); time.sleep(30)')
    allow(tools, 'calculator')
    result = call(tools, 'calculator', {'expression': '1'})
    assert result.error_code == 'result_too_large' and result.data is None
    assert children[0].returncode is not None


def test_memory_watchdog_is_enforced(tools, monkeypatch):
    from types import SimpleNamespace
    children = fake_worker(monkeypatch, 'import time; time.sleep(30)')
    monkeypatch.setattr('psutil.Process.memory_info', lambda self: SimpleNamespace(rss=257*1024**2))
    allow(tools, 'calculator')
    assert call(tools, 'calculator', {'expression': '1'}).error_code == 'resource_limit'
    assert children[0].returncode is not None


def test_parent_does_not_forward_credentials_or_pythonpath(tools, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-privateabcdefgh')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'private-aws-value')
    monkeypatch.setenv('PYTHONPATH', '/untrusted')
    fake_worker(monkeypatch, '''import json,os
value = int(any(k in os.environ for k in ('OPENAI_API_KEY','AWS_SECRET_ACCESS_KEY','PYTHONPATH')))
print(json.dumps({'ok':True,'data':{'value':value}}))''')
    allow(tools, 'calculator')
    assert call(tools, 'calculator', {'expression': '1'}).data == {'value': 0}


@pytest.mark.parametrize('payload', [
    'not json', '{"ok":true,"data":{"value":NaN}}',
    '{"ok":true,"data":{"value":1},"ok":false}',
    '{"ok":true,"data":{"value":1,"extra":"sk-privateabcdefgh"}}',
    '{"ok":false,"code":"sk-privateabcdefgh"}', '{"ok":false,"code":{}}',
    '{"ok":true,"data":{"value":true}}', '{"ok":true}',
])
def test_invalid_or_malicious_worker_output_is_not_released(tools, monkeypatch, payload):
    fake_worker(monkeypatch, 'import sys; sys.stdout.write(' + repr(payload) + ')')
    allow(tools, 'calculator')
    result = call(tools, 'calculator', {'expression': '1'})
    assert result.error_code == 'invalid_output' and result.data is None
    assert 'private' not in str(result)


def test_worker_errors_do_not_log_source_or_secret_tracebacks(tools, monkeypatch, capsys, caplog):
    fake_worker(monkeypatch, 'raise RuntimeError("sk-privateabcdefgh")')
    allow(tools, 'calculator')
    result = call(tools, 'calculator', {'expression': '1'})
    assert result.error_code == 'internal_error'
    captured = capsys.readouterr()
    assert 'private' not in str(result) + captured.out + captured.err + caplog.text


def test_registered_adapter_output_must_match_its_schema(tools):
    original = tools.registry.get('calculator', '1.0.0')
    tools.registry.register(replace(original, name='bad_adapter', output_schema=obj(required={'type':'boolean'})))
    allow(tools, 'bad_adapter')
    result = call(tools, 'bad_adapter', {'expression': '1'})
    assert result.error_code == 'invalid_output' and result.data is None


def test_real_result_size_limit_rejects_instead_of_leaking_partial_data(tools, tmp_path):
    (tmp_path / 'lines.txt').write_text(('x'*1000+'\n')*50)
    tools.permissions.approve_files('agent-a', ['lines.txt'])
    allow(tools, 'project_read')
    result = call(tools, 'project_read', {'path':'lines.txt','start_line':1,'line_count':50})
    assert result.error_code == 'result_too_large' and result.data is None


def test_grant_revocation_while_worker_runs_invalidates_result(tools, monkeypatch):
    original = tools._process
    async def revoke(*args):
        response = await original(*args)
        tools.permissions.revoke('agent-a', 'calculator')
        return response
    monkeypatch.setattr(tools, '_process', revoke)
    allow(tools, 'calculator')
    assert call(tools, 'calculator', {'expression':'1'}).error_code == 'permission_denied'


def test_knowledge_assignment_revocation_after_worker_is_rechecked(tools, repo, agent, kb, monkeypatch):
    indexed(kb)
    allow(tools, 'knowledge_search', agent.id)
    original = tools._process
    async def revoke(*args):
        response = await original(*args)
        kb.repository.assign(agent.id, [])
        return response
    monkeypatch.setattr(tools, '_process', revoke)
    result = call(tools, 'knowledge_search', {'query':'London','max_results':5}, agent=agent.id)
    assert result.error_code == 'permission_denied' and result.data is None


def test_concurrency_is_bounded(tools, monkeypatch):
    children = fake_worker(monkeypatch, 'import time; time.sleep(30)')
    allow(tools, 'calculator')
    async def scenario():
        one = asyncio.create_task(tools.execute('agent-a','calculator','1.0.0',{'expression':'1'}))
        two = asyncio.create_task(tools.execute('agent-a','calculator','1.0.0',{'expression':'2'}))
        while len(children) < 2:
            await asyncio.sleep(.01)
        three = await tools.execute('agent-a','calculator','1.0.0',{'expression':'3'})
        assert three.error_code == 'busy'
        one.cancel()
        two.cancel()
        await asyncio.gather(one, two, return_exceptions=True)
    asyncio.run(scenario())
    assert len(children) == 2 and all(p.returncode is not None for p in children)


def test_parent_redacts_valid_adapter_results_again(tools, monkeypatch, tmp_path):
    (tmp_path / 'safe.txt').write_text('safe')
    tools.permissions.approve_files('agent-a', ['safe.txt'])
    allow(tools, 'project_read')
    fake_worker(monkeypatch, '''import json
print(json.dumps({'ok':True,'data':{'path':'safe.txt','start_line':1,'lines':['sk-privateabcdefgh'],'truncated':False}}))''')
    result = call(tools, 'project_read', {'path':'safe.txt','start_line':1,'line_count':1})
    assert result.ok and result.data['lines'] == ['[REDACTED]']
    assert 'private' not in str(result)
