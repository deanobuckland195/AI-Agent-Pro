"""Visual projections and inspectors cannot execute work or grant permissions."""
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from ai_agent_pro.config import Agent
from ai_agent_pro.presentation.workflow_visual import nodes_for,elapsed_display,status_label,STYLE
from ai_agent_pro.workflow_store import WorkflowStore
from test_ui import launch,button
from test_workflows import prepare,result,execute,provider


@pytest.mark.parametrize('status', ['pending','running','completed','failed','cancelled','waiting_approval','interrupted'])
def test_visual_status_has_explicit_text_and_preserves_recorded_state(repo,status):
    store,wid,agents,value=prepare(repo)
    steps=store.steps(wid)
    steps[0]['status']=status
    row=store.get(wid)
    nodes=nodes_for(value,agents,row,steps)
    assert nodes[0]['status']==status
    assert len(status_label(status).split(' ',1)[1])>2
    assert nodes[1]['status']=='pending'
    assert store.get(wid)['status']=='ready'


def test_names_are_frozen_and_interruption_is_not_shown_as_running(repo):
    store,wid,agents,value=prepare(repo)
    current=[replace(a,name='Renamed later') for a in agents]
    row=store.get(wid)
    row['status']='interrupted'
    steps=store.steps(wid);steps[0]['status']='running'
    nodes=nodes_for(value,current,row,steps)
    assert nodes[0]['name']==agents[0].name
    assert nodes[0]['status']=='interrupted'
    assert nodes[1]['status']=='pending'


@pytest.mark.parametrize('state', ['ready','waiting_approval','completed','failed','cancelled'])
def test_waits_and_terminal_states_do_not_invent_elapsed_time(state):
    row=dict(status=state,active_seconds=18.45,lease_until=20)
    assert elapsed_display(row,dict(timeout_seconds=180),current=1000)==18.45


def test_live_elapsed_is_bounded_display_only():
    row=dict(status='running',active_seconds=4,lease_until=290)
    assert elapsed_display(row,dict(timeout_seconds=180),current=110)==10
    assert elapsed_display(row,dict(timeout_seconds=180),current=1000)==180
    assert row['active_seconds']==4


def test_inspection_changes_only_session_selection_and_renders_handoff(tmp_path,monkeypatch):
    at,repo=launch(tmp_path,monkeypatch)
    store,wid,agents,value=prepare(repo,5)
    backend=provider(*(result('Deliverable '+str(i)) for i in range(5)))
    execute(repo,backend,wid)
    before=store.get(wid)
    recorded=store.steps(wid)
    blocked=AsyncMock(side_effect=AssertionError('Inspecting must never call a model'))
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete',blocked)
    monkeypatch.setattr('ai_agent_pro.ui.api_key',lambda:'offline-test-key')
    at.sidebar.radio[0].set_value('Workflows').run()
    next(s for s in at.selectbox if s.label=='Workflow execution').set_value(wid).run()
    button(at,'Inspect step 3').click().run()
    assert not at.exception
    assert at.get('progress')[0].proto.value==100
    assert any(t.value=='Summary' for t in at.text)
    assert any(m.value=='Deliverable 2' for m in at.markdown)
    assert any('Step 3 inspection'==h.value for h in at.subheader)
    assert store.get(wid)==before and store.steps(wid)==recorded
    blocked.assert_not_called()
    assert not any(b.label in ('Start workflow','Continue workflow','Cancel workflow','Approve file change') for b in at.button)


def test_untrusted_node_text_is_never_interpolated_into_html(tmp_path,monkeypatch):
    at,repo=launch(tmp_path,monkeypatch)
    payload='<img src="https://untrusted.invalid/pixel" onerror="alert(1)">'
    agent=Agent.create(payload,'','Helpful')
    repo.save_agent(agent)
    from ai_agent_pro.workflow_contracts import software_workflow
    store=WorkflowStore(repo)
    value=software_workflow([agent.id]*5)
    value['steps'][0]['role']=payload
    store.save(value)
    monkeypatch.setattr('ai_agent_pro.ui.api_key',lambda:'')
    at.sidebar.radio[0].set_value('Workflows').run()
    assert not at.exception
    assert any(t.value==payload for t in at.text)
    assert all(payload not in m.value for m in at.markdown)
    assert all(payload not in h.value for h in at.subheader)
    assert button(at,'Start workflow').disabled
    assert 'prefers-reduced-motion' in STYLE
    assert ':focus-visible' in STYLE


def test_unknown_status_is_not_presented_as_completed(repo):
    store,wid,agents,value=prepare(repo)
    steps=store.steps(wid);steps[0]['status']='untrusted <script>'
    node=nodes_for(value,agents,store.get(wid),steps)[0]
    assert node['status']=='unknown' and status_label(node['status'])=='? Unknown'


def test_inspecting_pending_nodes_cannot_enable_unauthorised_continue(tmp_path,monkeypatch):
    at,repo=launch(tmp_path,monkeypatch)
    store,wid,agents,value=prepare(repo,5)
    before=store.get(wid)
    blocked=AsyncMock(side_effect=AssertionError('Node navigation cannot start execution'))
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete',blocked)
    monkeypatch.setattr('ai_agent_pro.ui.api_key',lambda:'')
    at.sidebar.radio[0].set_value('Workflows').run()
    next(s for s in at.selectbox if s.label=='Workflow execution').set_value(wid).run()
    assert button(at,'Continue workflow').disabled
    button(at,'Inspect step 4').click().run()
    assert not at.exception
    assert button(at,'Continue workflow').disabled
    assert all(not m.value for m in at.multiselect)
    assert store.get(wid)==before and before['requests']==0
    blocked.assert_not_called()
