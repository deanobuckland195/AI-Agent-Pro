"""Six-agent proposals, static QA, immutable history and offline safety."""
import asyncio
from dataclasses import replace
import json

import pytest

from ai_agent_pro.approvals import Approvals, ApprovalPaused
from ai_agent_pro.config import Agent, AppError, Settings
from ai_agent_pro.practice import PracticeWorkspace, NAME
from ai_agent_pro.tool_calling import ToolAccess
from ai_agent_pro.workflow_contracts import SOFTWARE_ROLES, QA_PREFIX, Limits, software_workflow
from ai_agent_pro.workflow_engine import WorkflowEngine, StepAccess
from ai_agent_pro.workflow_store import WorkflowStore
from test_tools import tools
from test_workflows import result, called, provider, execute, prepare
from test_ui import launch, button
from test_workflow_ui import select


def prepare_six(repo, limits=None):
    agents = [Agent.create(role, '', 'Work only within your assigned role.') for role in SOFTWARE_ROLES]
    for agent in agents:
        repo.save_agent(agent)
    value = software_workflow([a.id for a in agents], limits)
    store = WorkflowStore(repo)
    ident, version = store.save(value)
    return store, store.start(ident, version, 'Propose an add(a,b) function and tests.'), agents, value


def six_results():
    return [result('Acceptance criteria: add two integers.'), result('No external evidence used.'),
            result('A pure function with unit tests.'),
            result('Proposed code: def add(a,b): return a+b\nProposed tests: assert add(1,2)==3'),
            result(QA_PREFIX+'\nStatic finding: missing negative-number test. Proposed test: assert add(-1,2)==1'),
            result('Review of QA findings: add the proposed negative-number test. Execution is unverified.')]


def test_six_stages_receive_developer_code_and_qa_findings(repo):
    store, wid, agents, value = prepare_six(repo)
    assert value['limits'] == dict(max_requests=8, token_budget=30000, timeout_seconds=300, max_tool_calls=0)
    backend = provider(*six_results())
    execute(repo, backend, wid, on_text=lambda text:None)
    row = store.get(wid)
    assert row['status'] == 'completed' and row['cursor'] == 6 and row['requests'] == 6
    assert [s['role'] for s in value['steps']] == list(SOFTWARE_ROLES)
    assert len({s['conversation_id'] for s in store.steps(wid)}) == 6
    qa = backend.complete.call_args_list[4].kwargs
    reviewer = backend.complete.call_args_list[5].kwargs
    assert 'Proposed code: def add' in qa['messages'][0]['content']
    assert 'Proposed tests: assert add' in qa['messages'][0]['content']
    assert 'Tests have NOT been' in qa['instructions']
    assert 'on_delta' not in qa  # QA claims cannot stream before host validation.
    assert 'missing negative-number test' in reviewer['messages'][0]['content']
    assert 'Test execution: NOT RUN' in reviewer['messages'][0]['content']
    assert all(c.kwargs['max_output_tokens'] <= 1200 and 'tools' not in c.kwargs
               for c in backend.complete.call_args_list)


def test_new_version_preserves_five_stage_records_and_agent_settings(repo):
    store, old_wid, old_agents, old_value = prepare(repo, 5)
    old = store.get(old_wid)
    before_definitions, before_steps = store.definitions(), store.steps(old_wid)
    qa = Agent.create('QA Testing Engineer', '', 'Static QA only.', Settings(max_output_tokens=2400))
    repo.save_agent(qa)
    value = software_workflow([a.id for a in old_agents[:4]]+[qa.id,old_agents[4].id])
    ident, version = store.save(value, old['definition_id'], old['version']+1)
    new_wid = store.start(ident, version, 'New goal')
    assert store.get(old_wid) == old
    assert store.steps(old_wid) == before_steps
    assert all(d in store.definitions() for d in before_definitions)
    assert store.get(new_wid)['data']['agents'][4]['settings']['max_output_tokens'] == 1200
    assert repo.agent(qa.id).settings.max_output_tokens == 2400
    execute(repo, provider(*(result() for _ in range(5))), old_wid)
    execute(repo, provider(*six_results()), new_wid)
    assert store.get(old_wid)['cursor'] == 5 and store.get(new_wid)['cursor'] == 6
    assert json.loads(before_definitions[0]['definition']) == old_value
    with repo.connect() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 5
        assert db.execute('PRAGMA foreign_key_check').fetchall() == []


@pytest.mark.parametrize('report', ['All tests passed.', QA_PREFIX+'\nAll tests passed.',
    QA_PREFIX+'\npytest: 12 passed', QA_PREFIX+'\nExecuted the tests successfully.',
    QA_PREFIX+'\nThe test suite passed.', QA_PREFIX+'\nTests: **passed**.',
    QA_PREFIX+'\nTests have been executed.'])
def test_qa_unverified_success_or_missing_status_stops_reviewer(repo, report):
    store, wid, agents, value = prepare_six(repo)
    responses = six_results()
    responses[4] = result(report)
    backend = provider(*responses)
    with pytest.raises(AppError, match='QA handoff rejected'):
        execute(repo, backend, wid)
    assert backend.complete.call_count == 5
    assert store.steps(wid)[4]['output'] is None
    assert store.steps(wid)[5]['run_id'] is None
    assert store.get(wid)['status'] == 'failed'
    step = store.steps(wid)[4]
    with repo.connect() as db:
        assert db.execute('SELECT status FROM runs WHERE id=?',(step['run_id'],)).fetchone()[0] == 'failed'
        assert db.execute("SELECT count(*) FROM messages WHERE run_id=? AND role='assistant'",(step['run_id'],)).fetchone()[0] == 0


def test_six_stage_template_rejects_reused_architect_as_qa(repo):
    agent = Agent.create('Renamed architect', '', 'Instructions')
    repo.save_agent(agent)
    with pytest.raises(AppError, match='six distinct agents'):
        software_workflow([agent.id]*6)


@pytest.mark.parametrize('limits,message,expected', [
    (Limits(max_requests=4, token_budget=30000, max_tool_calls=0), '4/4 requests used', 4),
    (Limits(max_requests=8, token_budget=4096, max_tool_calls=0), 'shortfall', 0)])
def test_six_stage_budget_failure_is_clear_and_does_not_send_rejected_request(repo, limits, message, expected):
    store, wid, agents, value = prepare_six(repo, limits)
    # Force the initial reservation beyond the small budget, without sending data externally.
    if expected == 0:
        agent = agents[0]
        repo.save_agent(replace(agent, instructions='Bounded instructions. '*180))
        wid = store.start(store.get(wid)['definition_id'],1,'New goal')
    backend = provider(*six_results())
    with pytest.raises(AppError, match=message) as caught:
        execute(repo, backend, wid)
    assert 'No request was sent' in str(caught.value)
    assert backend.complete.call_count == expected
    assert store.get(wid)['requests'] == expected
    assert len([s for s in store.steps(wid) if s['status']=='completed']) == expected


def test_zero_tool_allowance_rejects_selected_qa_tool_before_model(repo, tools):
    store, wid, agents, value = prepare_six(repo)
    tools.permissions.allow(agents[4].id, 'calculator')
    access = {4: StepAccess(agents[4].id, ToolAccess(tools,(('calculator','1.0.0'),)))}
    backend = provider(*six_results())
    with pytest.raises(AppError, match='tool-call budget'):
        execute(repo, backend, wid, access)
    assert backend.complete.call_count == 4
    assert store.get(wid)['requests'] == 4
    assert store.steps(wid)[5]['run_id'] is None


def test_qa_grants_are_not_inherited_by_reviewer(repo, tools):
    store, wid, agents, value = prepare_six(repo, Limits(max_requests=8, token_budget=30000, max_tool_calls=1))
    tools.permissions.allow(agents[4].id, 'calculator')
    access = {4: StepAccess(agents[4].id, ToolAccess(tools,(('calculator','1.0.0'),)))}
    responses = six_results()
    backend = provider(*responses[:4], called(), *responses[4:])
    execute(repo, backend, wid, access)
    assert backend.complete.call_args_list[4].kwargs['tools']
    assert 'tools' not in backend.complete.call_args_list[-1].kwargs
    assert store.get(wid)['tool_calls'] == 1 and store.get(wid)['requests'] == 7


@pytest.mark.parametrize('valid_qa', [True, False])
def test_qa_file_change_still_requires_exact_human_approval(repo, tools, tmp_path, valid_qa):
    store, wid, agents, value = prepare_six(repo, Limits(max_requests=8, token_budget=30000, max_tool_calls=1))
    practice = PracticeWorkspace(tmp_path)
    practice.create_example()
    access = {4: StepAccess(agents[4].id, ToolAccess(tools, (), practice, ('example.txt',)))}
    backend = provider(*six_results()[:4], called(NAME, '{"path":"example.txt","content":"Approved QA change"}'))
    with pytest.raises(ApprovalPaused):
        execute(repo, backend, wid, access, workspace=practice)
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
    assert store.steps(wid)[5]['run_id'] is None
    step = store.steps(wid)[4]
    approvals = Approvals(repo, practice)
    action = approvals.list(agents[4].id, step['conversation_id'])[0]
    approvals.decide(action['id'], agents[4].id, action['run_id'], action['binding'], True)
    final = provider(six_results()[4] if valid_qa else result(QA_PREFIX+'\nAll tests passed.'))
    if not valid_qa:
        with pytest.raises(AppError, match='QA handoff rejected'):
            asyncio.run(WorkflowEngine(repo, final, workspace=practice).run(wid, approval_id=action['id']))
        assert store.get(wid)['status'] == 'failed'
        assert store.steps(wid)[5]['run_id'] is None
        assert (practice.path/'example.txt').read_text() == 'Approved QA change'
        with repo.connect() as db:
            assert db.execute("SELECT count(*) FROM messages WHERE run_id=? AND role='assistant'",(step['run_id'],)).fetchone()[0] == 0
        return
    asyncio.run(WorkflowEngine(repo, final, workspace=practice).run(wid, approval_id=action['id']))
    assert store.get(wid)['status'] == 'ready' and store.get(wid)['cursor'] == 5
    assert store.steps(wid)[5]['run_id'] is None
    execute(repo, provider(six_results()[5]), wid)
    assert store.get(wid)['status'] == 'completed'
    assert (practice.path/'example.txt').read_text() == 'Approved QA change'


def test_ui_six_stage_version_defaults_inspection_and_old_definition_preserved(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    store, old_wid, agents, old_value = prepare(repo,5)
    qa = Agent.create('QA Testing Engineer','','Static QA only.')
    repo.save_agent(qa)
    before, old = store.definitions(), store.get(old_wid)
    assignments = agents[:4]+[qa, agents[4]]
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda:'offline-test-key')
    backend = provider(*six_results())
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', backend.complete)
    at.sidebar.radio[0].set_value('Workflows').run()
    for role, agent in zip(SOFTWARE_ROLES,assignments):
        select(at,role+' agent').set_value(agent.id)
    select(at,'Definition to version').set_value(old['definition_id']).run()
    assert next(w for w in at.number_input if w.label=='Workflow version').value == 2
    for label, expected in [('Workflow model-request limit',8),('Workflow token budget',30000),
                            ('Workflow active-time limit (seconds)',300),('Workflow tool-call limit',0)]:
        assert next(w for w in at.number_input if w.label==label).value == expected
    button(at,'Save workflow version').click().run()
    assert all(d in store.definitions() for d in before)
    assert store.get(old_wid) == old
    qa_tools = next(w for w in at.multiselect if w.label=='Tools for step 5')
    qa_tools.select('calculator').run()
    assert button(at,'Start workflow').disabled
    assert any('tool-call limit is zero' in w.value for w in at.warning)
    next(w for w in at.multiselect if w.label=='Tools for step 5').unselect('calculator').run()
    next(w for w in at.text_area if w.label=='Workflow goal').set_value('Propose a harmless implementation')
    at.run()
    button(at,'Start workflow').click().run()
    assert at.session_state['_workflow_job'].wait(10)
    at.run()
    assert not at.exception
    assert backend.complete.call_count == 6
    assert len([b for b in at.button if b.label.startswith('Inspect step ')]) == 6
    button(at,'Inspect step 5').click().run()
    assert any('Test execution: NOT RUN' in i.value for i in at.info)
    assert any(t.value=='QA Testing Engineer' for t in at.text)
    assert store.runs()[0]['version'] == 2
