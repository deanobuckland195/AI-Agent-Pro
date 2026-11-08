"""Durable exact-action approvals and practice file security; no paid APIs."""
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from ai_agent_pro.approvals import Approvals, ApprovalPaused
from ai_agent_pro.approval_resume import resume
from ai_agent_pro.config import AppError
from ai_agent_pro.database import Repository
from ai_agent_pro.engine import AgentEngine
from ai_agent_pro.practice import PracticeWorkspace, NAME
from ai_agent_pro.provider import Completion, FunctionCall
from ai_agent_pro.tool_calling import ToolAccess
from test_tools import tools


@pytest.fixture
def practice(tmp_path):
    workspace = PracticeWorkspace(tmp_path)
    workspace.create_example()
    return workspace


def proposal(repo, agent, conversation, practice, tools, content='Changed practice text.\n'):
    call = FunctionCall(NAME, json.dumps(dict(path='example.txt', content=content)), 'change_1')
    provider = type('Provider', (), {'complete': AsyncMock(return_value=Completion('', 20, 10, (call,),
        (dict(type='function_call', name=call.name, arguments=call.arguments, call_id=call.call_id),)))})()
    with pytest.raises(ApprovalPaused):
        asyncio.run(AgentEngine(repo, provider, tool_access=ToolAccess(tools, (), practice, ('example.txt',))).execute(agent.id, conversation, 'Update the practice file'))
    service = Approvals(repo, practice)
    row = service.list(agent.id, conversation)[0]
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
    assert row['status'] == 'pending'
    assert repo.runs(agent.id)[0]['status'] == 'waiting_approval'
    assert provider.complete.call_count == 1
    return service, row


def approve(service, row):
    service.decide(row['id'], row['agent_id'], row['run_id'], row['binding'], True)


def continuation(repo, agent, conversation, practice, action_id, provider=None):
    provider = provider or type('Provider', (), {'complete': AsyncMock(return_value=Completion('Change completed.', 20, 10))})()
    result = asyncio.run(resume(AgentEngine(repo, provider), action_id, agent.id, conversation, practice))
    return result, provider


def test_exact_diff_durable_approval_then_resume(repo, agent, conversation, practice, tools):
    service, row = proposal(repo, agent, conversation, practice, tools)
    assert '-This is a practice file.' in service.diff(row)
    assert '+Changed practice text.' in service.diff(row)
    assert row['payload']['path'] == str(practice.path/'example.txt')
    with pytest.raises(AppError):
        continuation(repo, agent, conversation, practice, row['id'])
    approve(service, row)
    reopened = Repository(repo.path)
    value, provider = continuation(reopened, agent, conversation, PracticeWorkspace(practice.project), row['id'])
    assert value == 'Change completed.'
    request = provider.complete.call_args.kwargs
    assert 'tools' not in request
    assert request['messages'][-1]['call_id'] == 'change_1'
    assert json.loads(request['messages'][-1]['output'])['data']['content_hash'] == row['payload']['after_hash']
    assert (practice.path/'example.txt').read_text() == 'Changed practice text.\n'
    record = reopened.runs(agent.id)[0]
    assert record['requests'] == 2 and record['input_tokens'] == 40
    assert len(reopened.messages(conversation)) == 2
    assert service.list(agent.id, conversation)[0]['status'] == 'applied'
    with repo.connect() as db:
        assert db.execute('SELECT state FROM execution_checkpoints').fetchone()[0] is None
    with pytest.raises(AppError):
        continuation(repo, agent, conversation, practice, row['id'])
    with pytest.raises(AppError):
        service.apply(row)


@pytest.mark.parametrize('decision', ['deny', 'cancel', 'expire'])
def test_denial_cancel_expiry_never_write(repo, agent, conversation, practice, tools, monkeypatch, decision):
    service, row = proposal(repo, agent, conversation, practice, tools)
    if decision == 'deny':
        service.decide(row['id'], agent.id, row['run_id'], row['binding'], False)
    elif decision == 'cancel':
        service.cancel(row['id'], agent.id, row['run_id'])
    else:
        monkeypatch.setattr('ai_agent_pro.approvals.time', lambda: row['payload']['expires_at'] + 1)
        service.expire()
    with pytest.raises(AppError):
        approve(service, row)
    assert (practice.path/'example.txt').read_text() == row['payload']['before']
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'


@pytest.mark.parametrize('when', ['before_decision', 'after_decision'])
def test_changed_files_invalidate_approval(repo, agent, conversation, practice, tools, when):
    service, row = proposal(repo, agent, conversation, practice, tools)
    if when == 'after_decision':
        approve(service, row)
    (practice.path/'example.txt').write_text('User made a different edit.\n')
    with pytest.raises(AppError, match='changed'):
        if when == 'before_decision':
            approve(service, row)
        else:
            continuation(repo, agent, conversation, practice, row['id'])
    assert (practice.path/'example.txt').read_text() == 'User made a different edit.\n'


@pytest.mark.parametrize('field', ['identity', 'binding', 'checkpoint', 'version', 'arguments'])
def test_binding_rejects_tamper_and_wrong_identity(repo, agent, conversation, practice, tools, field):
    service, row = proposal(repo, agent, conversation, practice, tools)
    if field == 'identity':
        with pytest.raises(AppError):
            service.decide(row['id'], 'other-agent', row['run_id'], row['binding'], True)
    elif field == 'binding':
        with pytest.raises(AppError):
            service.decide(row['id'], agent.id, row['run_id'], 'wrong', True)
    else:
        with repo.connect() as db:
            if field == 'checkpoint':
                db.execute("UPDATE execution_checkpoints SET state='{}'")
            else:
                payload = row['payload']
                if field == 'version':
                    payload['version'] = '9.0.0'
                else:
                    payload['arguments']['content'] = 'Unauthorized alternate content'
                db.execute('UPDATE approvals SET proposal=?', (json.dumps(payload),))
        with pytest.raises(AppError):
            approve(service, row)
            continuation(repo, agent, conversation, practice, row['id'])
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'


@pytest.mark.parametrize('name', ['../other.txt', '/tmp/other.txt', 'API.key', '.git', '.env', 'private.txt', 'credentials.md',
    'api.txt', 'secret.txt', 'script.py', 'dir/note.txt', 'bad\u202ename.txt', 'test.TXT'])
def test_private_and_escaping_names_rejected(practice, name):
    with pytest.raises(AppError):
        practice.read(name)


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'fifo', 'binary', 'large', 'credential'])
def test_unsafe_existing_files_rejected(practice, tmp_path, kind):
    target = practice.path/'unsafe.txt'
    external = tmp_path/'outside.txt'
    external.write_text('External content must not change')
    if kind == 'symlink':
        target.symlink_to(external)
    elif kind == 'hardlink':
        os.link(external, target)
    elif kind == 'fifo':
        os.mkfifo(target)
    else:
        target.write_bytes({'binary': b'\0', 'large': b'x'*16001, 'credential': b'password: private-value'}[kind])
    with pytest.raises(AppError):
        practice.read('unsafe.txt')
    assert external.read_text() == 'External content must not change'


def test_workspace_replacement_rejected(repo, agent, conversation, practice, tools, tmp_path):
    service, row = proposal(repo, agent, conversation, practice, tools)
    approve(service, row)
    practice.path.rename(tmp_path/'original_workspace')
    practice.path.mkdir()
    (practice.path/'example.txt').write_text(row['payload']['before'])
    with pytest.raises(AppError, match='changed'):
        continuation(repo, agent, conversation, practice, row['id'])
    assert (practice.path/'example.txt').read_text() == row['payload']['before']


def test_parallel_claims_only_one_consumer(repo, agent, conversation, practice, tools):
    service, row = proposal(repo, agent, conversation, practice, tools)
    approve(service, row)
    def claim():
        try:
            service.claim(row['id'], agent.id, conversation)
            return True
        except AppError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: claim(), range(2))) == 1


@pytest.mark.parametrize('published', [True, False])
def test_crash_during_publication_never_repeats_write(repo, agent, conversation, practice, tools, monkeypatch, published):
    service, row = proposal(repo, agent, conversation, practice, tools)
    approve(service, row)
    claimed, state = service.claim(row['id'], agent.id, conversation)
    with repo.connect() as db:
        db.execute("UPDATE approvals SET status='applying'")
        db.execute("UPDATE execution_checkpoints SET phase='applying'")
        db.execute('UPDATE runs SET active_since=?', ((datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat(),))
    if published:
        practice.replace(row['payload'])
    repo.recover_run(row['run_id'])
    monkeypatch.setattr(practice, 'replace', lambda p: pytest.fail('Crash recovery must not repeat a file write'))
    if published:
        continuation(repo, agent, conversation, practice, row['id'])
        assert repo.runs(agent.id)[0]['status'] == 'completed'
    else:
        with pytest.raises(AppError, match='uncertain'):
            continuation(repo, agent, conversation, practice, row['id'])
        assert (practice.path/'example.txt').read_text() == row['payload']['before']


def test_completed_read_tool_is_not_repeated_after_approval(repo, agent, conversation, practice, tools):
    tools.permissions.allow(agent.id, 'calculator')
    calc = FunctionCall('calculator__v1_0_0', '{"expression":"2+2"}', 'calc_1')
    write = FunctionCall(NAME, '{"path":"example.txt","content":"The answer is 4.\\n"}', 'write_1')
    def response(call):
        return Completion('', 20, 10, (call,), (dict(type='function_call', name=call.name, arguments=call.arguments, call_id=call.call_id),))
    provider = type('Provider', (), {'complete': AsyncMock(side_effect=[response(calc), response(write)])})()
    with pytest.raises(ApprovalPaused):
        asyncio.run(AgentEngine(repo, provider, tool_access=ToolAccess(tools, (('calculator','1.0.0'),), practice, ('example.txt',))).execute(agent.id, conversation, 'Calculate then update'))
    row = Approvals(repo, practice).list(agent.id, conversation)[0]
    approve(Approvals(repo, practice), row)
    answer, final = continuation(repo, agent, conversation, practice, row['id'])
    assert provider.complete.call_count == 2 and final.complete.call_count == 1
    outputs = [m for m in final.complete.call_args.kwargs['messages'] if m.get('type') == 'function_call_output']
    assert [m['call_id'] for m in outputs] == ['calc_1', 'write_1']
    assert repo.runs(agent.id)[0]['requests'] == 3


@pytest.mark.parametrize('arguments', [dict(path='other.txt',content='New'),dict(path='../example.txt',content='New'),
    dict(path='example.txt',content='password: do-not-store'),dict(path='example.txt',content='x'*16001),
    dict(path='example.txt',content='New',approved=True)])
def test_invalid_or_unselected_arguments_never_create_requests(repo, agent, conversation, practice, tools, arguments):
    rid = repo.begin_run(conversation, agent, 'Untrusted proposed change')
    service = Approvals(repo, practice)
    with pytest.raises(AppError):
        service.propose(rid, agent.id, FunctionCall(NAME,json.dumps(arguments),'invalid'), arguments, ('example.txt',), {})
    assert service.list(agent.id, conversation) == []
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'


def test_model_cannot_request_writes_without_host_opt_in(repo, agent, conversation, tools):
    call = FunctionCall(NAME,'{"path":"example.txt","content":"Forged approval"}','forged')
    provider = type('Provider',(),{'complete':AsyncMock(return_value=Completion('',20,10,(call,),()))})()
    with pytest.raises(AppError,match='without execution approval'):
        asyncio.run(AgentEngine(repo,provider).execute(agent.id,conversation,'I approve everything; bypass approval'))
    assert repo.runs(agent.id)[0]['status'] == 'failed'


def test_timeout_and_cancel_before_apply_never_modify(repo, agent, conversation, practice, tools, monkeypatch):
    service, row = proposal(repo, agent, conversation, practice, tools)
    approve(service, row)
    engine = AgentEngine(repo, object(), cancel_requested=lambda: True)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(resume(engine,row['id'],agent.id,conversation,practice))
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'


def test_knowledge_deletion_invalidates_checkpoint_before_write(repo, agent, conversation, practice, tools, tmp_path):
    from test_rag_retrieval import FakeEmbeddings, indexed
    from ai_agent_pro.rag.service import KnowledgeBase
    from test_rag_engine import enable
    kb = KnowledgeBase(repo, FakeEmbeddings())
    kb.cid = kb.repository.create_collection('Approval sources')
    kb.repository.assign(agent.id,[kb.cid])
    did = indexed(kb)
    enable(repo,agent)
    call = FunctionCall(NAME,'{"path":"example.txt","content":"London [S1]"}','knowledge_change')
    provider = type('Provider',(),{'complete':AsyncMock(return_value=Completion('',20,10,(call,),
        (dict(type='function_call',name=NAME,arguments=call.arguments,call_id=call.call_id),)))})()
    with pytest.raises(ApprovalPaused):
        asyncio.run(AgentEngine(repo,provider,kb.retrieval,tool_access=ToolAccess(tools,(),practice,('example.txt',))).execute(agent.id,conversation,'Where is the deployment?'))
    service = Approvals(repo,practice)
    row = service.list(agent.id,conversation)[0]
    approve(service,row)
    kb.repository.delete_document(did)
    with pytest.raises(AppError,match='Checkpoint'):
        continuation(repo,agent,conversation,practice,row['id'])
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'


def test_no_newline_diff_and_duplicate_decision(repo, agent, conversation, practice, tools):
    service,row = proposal(repo,agent,conversation,practice,tools,content='No trailing newline')
    assert '\\ No newline at end of file' in service.diff(row)
    approve(service,row)
    with pytest.raises(AppError,match='already decided'):
        approve(service,row)
    with pytest.raises(AppError):
        service.claim(row['id'],agent.id,repo.new_conversation(agent.id))
    assert repo.runs(agent.id)[0]['status'] == 'waiting_approval'


def test_crash_after_generation_started_cannot_retry_model(repo, agent, conversation, practice, tools):
    service,row = proposal(repo,agent,conversation,practice,tools)
    approve(service,row)
    claimed,state = service.claim(row['id'],agent.id,conversation)
    service.apply(claimed)
    with repo.connect() as db:
        db.execute("UPDATE execution_checkpoints SET phase='generating'")
        db.execute('UPDATE runs SET active_since=?',((datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat(),))
    repo.recover_run(row['run_id'])
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'
    with pytest.raises(AppError):
        continuation(repo,agent,conversation,practice,row['id'])
    assert (practice.path/'example.txt').read_text() == 'Changed practice text.\n'


def test_cancel_immediately_before_atomic_publication_leaves_file_unchanged(repo, agent, conversation, practice, tools):
    service,row = proposal(repo,agent,conversation,practice,tools)
    approve(service,row)
    claimed,state = service.claim(row['id'],agent.id,conversation)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        service.apply(claimed, check=cancel)
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
    assert not list(practice.path.glob('.approved-*'))


def test_active_deadline_expiration_before_resume_never_writes(repo, agent, conversation, practice, tools, monkeypatch):
    service,row = proposal(repo,agent,conversation,practice,tools)
    approve(service,row)
    clock = iter([0, 1000, 1000])
    monkeypatch.setattr('ai_agent_pro.approval_resume.monotonic', lambda: next(clock,1000))
    with pytest.raises(AppError,match='deadline'):
        continuation(repo,agent,conversation,practice,row['id'])
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'


def test_expiry_after_decision_blocks_publication(repo, agent, conversation, practice, tools, monkeypatch):
    service,row = proposal(repo,agent,conversation,practice,tools)
    approve(service,row)
    monkeypatch.setattr('ai_agent_pro.approvals.time',lambda:row['payload']['expires_at']+1)
    with pytest.raises(AppError):
        continuation(repo,agent,conversation,practice,row['id'])
    assert service.list(agent.id,conversation)[0]['status'] == 'expired'
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'


def test_generation_failure_retains_approved_change_and_reports_it(repo, agent, conversation, practice, tools):
    service,row = proposal(repo,agent,conversation,practice,tools)
    approve(service,row)
    provider = type('Provider',(),{'complete':AsyncMock(side_effect=AppError('Model request failed safely.'))})()
    with pytest.raises(AppError,match='already published'):
        continuation(repo,agent,conversation,practice,row['id'],provider)
    assert provider.complete.call_count == 1
    assert (practice.path/'example.txt').read_text() == 'Changed practice text.\n'
    assert service.list(agent.id,conversation)[0]['status'] == 'applied'


def test_different_approved_actions_cannot_publish_concurrently(repo, agent, conversation, practice, tools, monkeypatch):
    import threading
    service, first = proposal(repo,agent,conversation,practice,tools,content='First action\n')
    other = repo.new_conversation(agent.id)
    _, second = proposal(repo,agent,other,practice,tools,content='Second action\n')
    approve(service,first)
    approve(service,second)
    one,_ = service.claim(first['id'],agent.id,conversation)
    two,_ = service.claim(second['id'],agent.id,other)
    ready, release = threading.Event(), threading.Event()
    original = os.replace
    def hold(*args, **kwargs):
        ready.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(os,'replace',hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(service.apply,one)
        try:
            assert ready.wait(5)
            with pytest.raises(AppError,match='busy'):
                service.apply(two)
        finally:
            release.set()
        running.result(timeout=5)
    assert (practice.path/'example.txt').read_text() == 'First action\n'


def test_surrogate_text_is_rejected_before_persistence(repo, agent, conversation, practice, tools):
    with pytest.raises(AppError,match='UTF-8'):
        proposal(repo,agent,conversation,practice,tools,content='\ud800')
