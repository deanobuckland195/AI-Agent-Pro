"""Dashboard charts, filters, recovery states and library interactions."""
from datetime import datetime, timedelta, timezone
from ai_agent_pro.config import Agent
from ai_agent_pro.presentation.components import recovery_wait
from ai_agent_pro.rag.repository import KnowledgeRepository
from ai_agent_pro.rag.storage import DocumentStorage
from test_ui import launch, button
from test_analytics import record


def test_dashboard_uses_recorded_metrics_and_charts(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Measured assistant', '', 'Helpful')
    repo.save_agent(agent)
    record(repo, agent, elapsed=2, steps=[{'input_tokens': 10, 'output_tokens': 3}])
    record(repo, agent, 'failed', elapsed=4, error='OpenAI request timed out.')
    at.run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics['Recorded tasks'] == '2'
    assert metrics['Completion rate'] == '50%'
    assert metrics['Known generation tokens'] == '13'
    assert metrics['Median task time'] == '3.00s'
    assert len(at.get('vega_lite_chart')) >= 4
    assert not at.exception
    button(at, 'Inspect task').click().run()
    assert at.sidebar.radio[0].value == 'Activity'
    assert any(s.label == 'Inspect task' for s in at.selectbox)
    assert not at.exception
    button(at, 'Open conversation').click().run()
    assert at.sidebar.radio[0].value == 'Chat'
    assert at.sidebar.selectbox[0].value == agent.id


def test_activity_filters_refresh_and_recovery_eligibility(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Assistant', '', 'Helpful')
    repo.save_agent(agent)
    rid = record(repo, agent, 'running')
    at.sidebar.radio[0].set_value('Activity').run()
    assert button(at, 'Recover interrupted task').disabled
    with repo.connect() as db:
        db.execute('UPDATE runs SET started_at=? WHERE id=?',
                   ((datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(), rid))
    button(at, 'Refresh records').click().run()
    assert not button(at, 'Recover interrupted task').disabled
    button(at, 'Recover interrupted task').click().run()
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'
    next(s for s in at.selectbox if s.label == 'Task status').set_value('completed').run()
    assert any('No recorded tasks match' in x.value for x in at.info)


def test_library_search_filters_duplicate_upload_and_refresh(tmp_path, monkeypatch):
    from types import SimpleNamespace
    at, repo = launch(tmp_path, monkeypatch)
    knowledge = KnowledgeRepository(repo)
    cid = knowledge.create_collection('Engineering')
    storage = DocumentStorage(knowledge)
    storage.upload(cid, 'first.txt', b'hello')
    storage.upload(cid, 'second.txt', b'world')
    monkeypatch.setattr('streamlit.file_uploader', lambda *args, **kwargs:
                        SimpleNamespace(name='first.txt', getvalue=lambda: b'hello'))
    at.sidebar.radio[0].set_value('Knowledge base').run()
    next(x for x in at.text_input if x.label == 'Search documents').set_value('second').run()
    detail = next(s for s in at.selectbox if s.label == 'Document details')
    assert len(detail.options) == 1 and detail.options[0] == 'second.txt'
    button(at, 'Upload document').click().run()
    assert len(knowledge.documents(cid)) == 2
    assert any('Already in this collection' in x.value for x in at.success)
    next(x for x in at.selectbox if x.label == 'File type').set_value('pdf').run()
    assert any('No documents match' in x.value for x in at.info)
    button(at, 'Refresh library').click().run()
    assert not at.exception


def test_library_recovery_disabled_until_deadline(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    knowledge = KnowledgeRepository(repo)
    cid = knowledge.create_collection('Engineering')
    did = DocumentStorage(knowledge).upload(cid, 'notes.txt', b'hello')
    jid = knowledge.begin_index(did)
    at.sidebar.radio[0].set_value('Knowledge base').run()
    assert button(at, 'Recover interrupted indexing').disabled
    with repo.connect() as db:
        db.execute('UPDATE index_jobs SET started_at=? WHERE id=?',
                   ((datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(), jid))
    button(at, 'Refresh library').click().run()
    assert not button(at, 'Recover interrupted indexing').disabled
    button(at, 'Recover interrupted indexing').click().run()
    assert knowledge.jobs(did)[0]['status'] == 'interrupted'


def test_filters_and_conversation_selection_survive_navigation(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Assistant', '', 'Helpful')
    repo.save_agent(agent)
    first = repo.new_conversation(agent.id, 'First')
    repo.new_conversation(agent.id, 'Second')
    at.sidebar.radio[0].set_value('Chat').run()
    next(s for s in at.selectbox if s.label == 'Conversation').set_value(first).run()
    at.sidebar.radio[0].set_value('Agents').run()
    next(x for x in at.text_input if x.label == 'Search agents').set_value('Assistant').run()
    at.sidebar.radio[0].set_value('Chat').run()
    assert next(s for s in at.selectbox if s.label == 'Conversation').value == first
    at.sidebar.radio[0].set_value('Agents').run()
    assert next(x for x in at.text_input if x.label == 'Search agents').value == 'Assistant'


def test_recovery_wait_uses_real_deadline():
    current = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    assert recovery_wait(current.isoformat(), 60, current) == 70
    assert recovery_wait((current - timedelta(seconds=70)).isoformat(), 60, current) == 0
