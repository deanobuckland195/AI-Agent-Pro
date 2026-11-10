"""Real Streamlit workflow authoring, execution and progress controls."""
from unittest.mock import AsyncMock

from ai_agent_pro.config import Agent
from ai_agent_pro.workflow_store import WorkflowStore
from test_ui import launch,button
from test_workflows import result, prepare


def select(at,label):
    return next(w for w in at.selectbox if w.label==label)


def test_ui_assign_existing_agents_and_execute_five_steps(tmp_path,monkeypatch):
    at,repo=launch(tmp_path,monkeypatch)
    agents=[Agent.create('Role '+str(i),'','Helpful specialist') for i in range(5)]
    for agent in agents:repo.save_agent(agent)
    monkeypatch.setattr('ai_agent_pro.ui.api_key',lambda:'offline-test-key')
    model=AsyncMock(side_effect=[result('Output '+str(i)) for i in range(5)])
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete',model)
    at.sidebar.radio[0].set_value('Workflows').run()
    select(at,'Workflow template').set_value('Original five-stage software engineering').run()
    assert button(at,'Save workflow version').disabled
    for i,role in enumerate(('Requirements Analyst','Researcher','Architect','Developer','Reviewer')):
        select(at,role+' agent').set_value(agents[i].id)
    at.run()
    button(at,'Save workflow version').click().run()
    assert len(WorkflowStore(repo).definitions())==1
    assert all(not m.value for m in at.multiselect)
    next(w for w in at.text_area if w.label=='Workflow goal').set_value('Design a harmless example')
    at.run()
    button(at,'Start workflow').click().run()
    assert at.session_state['_workflow_job'].wait(10)
    at.run()
    assert not at.exception
    row=WorkflowStore(repo).runs()[0]
    assert row['status']=='completed' and row['requests']==5
    assert model.call_count==5
    assert any(t.value=='Workflow status: completed' for t in at.text)
    assert any(t.value=='Reviewer' for t in at.text)
    assert len([t for t in at.text if t.value=='✓ Completed'])>=5
    assert len([b for b in at.button if b.label.startswith('Inspect step ')])==5


def test_ui_requires_key_and_keeps_step_permissions_independent(tmp_path,monkeypatch):
    at,repo=launch(tmp_path,monkeypatch)
    store,wid,agents,value=prepare(repo)
    monkeypatch.setattr('ai_agent_pro.ui.api_key',lambda:'')
    at.sidebar.radio[0].set_value('Workflows').run()
    at.multiselect[0].select('calculator').run()
    assert at.multiselect[2].value==[] # second step's tools, after first practice selector
    assert button(at,'Start workflow').disabled
    assert not at.exception
