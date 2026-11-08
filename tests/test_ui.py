from unittest.mock import AsyncMock
from pathlib import Path
from streamlit.testing.v1 import AppTest
from ai_agent_pro.config import Agent
from ai_agent_pro.database import Repository
from ai_agent_pro.provider import Completion


def launch(tmp_path, monkeypatch):
    path = tmp_path / 'ui.sqlite3'
    monkeypatch.setenv('AI_AGENT_PRO_DB', str(path))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    return AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=15).run(), Repository(path)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def finish_chat(at):
    assert at.session_state['_chat_job'].wait(5)
    at.run()
    assert not at.exception


def test_empty_startup(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    assert not at.exception
    assert 'AI Agent Pro' in at.title[0].value
    assert not repo.agents()


def test_agent_form_create_edit_delete(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    at.sidebar.radio[0].set_value('Agents').run()
    next(x for x in at.text_input if x.label == 'Agent name').set_value('Portfolio assistant')
    button(at, 'Create agent').click().run()
    assert not at.exception
    agent = repo.agents()[0]
    assert agent.name == 'Portfolio assistant'
    next(r for r in at.radio if r.label == 'Agent management').set_value('Edit selected agent').run()
    next(x for x in at.text_input if x.label == 'Agent name').set_value('Writing assistant')
    at.number_input[0].set_value(2)
    button(at, 'Save agent').click().run()
    assert not at.exception
    assert repo.agent(agent.id).settings.max_steps == 2
    assert repo.agent(agent.id).name == 'Writing assistant'
    at.checkbox[0].check().run()
    button(at, 'Delete agent').click().run()
    assert not at.exception
    assert repo.agents() == []


def test_chat_persistence_new_and_clear(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Assistant', 'Helpful', 'Be helpful')
    repo.save_agent(agent)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-private-key')
    completion = AsyncMock(return_value=Completion('<script>unsafe()</script>', 10, 5))
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', completion)
    at.sidebar.radio[0].set_value('Chat').run()
    assert not at.chat_input[0].disabled
    at.chat_input[0].set_value('Hello assistant').run()
    finish_chat(at)
    assert not at.exception
    cid = repo.conversations(agent.id)[0]['id']
    assert len(repo.messages(cid)) == 2
    rendered = next(m for m in at.markdown if m.value == '<script>unsafe()</script>')
    assert not rendered.proto.allow_html
    at.run()
    at.checkbox[0].check().run()
    button(at, 'Clear conversation').click().run()
    assert not at.exception
    assert repo.messages(cid) == []
    assert not any('Completed' in message.value for message in at.success)
    button(at, 'New conversation').click().run()
    assert len(repo.conversations(agent.id)) == 2
    at.sidebar.radio[0].set_value('Activity').run()
    assert not at.exception
    at.sidebar.radio[0].set_value('Help').run()
    assert not at.exception


def test_no_key_disables_chat(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    repo.save_agent(Agent.create('Assistant', '', 'Helpful'))
    # Isolate from any private secrets configured on the workstation.
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: '')
    at.sidebar.radio[0].set_value('Chat').run()
    assert not at.exception
    assert at.chat_input[0].disabled


def test_knowledge_base_full_document_workflow(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_rag_retrieval import FakeEmbeddings
    from ai_agent_pro.rag.repository import KnowledgeRepository
    embeddings = FakeEmbeddings()
    monkeypatch.setattr('ai_agent_pro.rag.ui.OpenAIEmbeddings', lambda key: embeddings)
    monkeypatch.setattr('ai_agent_pro.presentation.workspace.OpenAIEmbeddings', lambda key: embeddings)
    monkeypatch.setattr('streamlit.file_uploader', lambda *args, **kwargs:
                        SimpleNamespace(name='deployment.txt', getvalue=lambda: b'The deployment region is London.'))
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Document assistant', '', 'Answer accurately.')
    repo.save_agent(agent)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-private-key')
    at.run()
    at.sidebar.radio[0].set_value('Knowledge base').run()
    at.text_input[0].set_value('Engineering')
    button(at, 'Create collection').click().run()
    assert not at.exception
    knowledge = KnowledgeRepository(repo)
    cid = knowledge.collections()[0]['id']
    at.multiselect[0].set_value([cid])
    button(at, 'Save collection assignments').click().run()
    assert knowledge.assigned(agent.id) == [cid]
    button(at, 'Upload document').click().run()
    assert not at.exception
    did = knowledge.documents(cid)[0]['id']
    button(at, 'Index document').click().run()
    assert not at.exception
    assert knowledge.document(did)['active_generation']
    first = knowledge.document(did)['active_generation']
    calls = embeddings.calls
    button(at, 'Re-index document').click().run()
    assert not at.exception
    assert knowledge.document(did)['active_generation'] != first
    assert embeddings.calls == calls
    at.sidebar.radio[0].set_value('Agents').run()
    next(r for r in at.radio if r.label == 'Agent management').set_value('Edit selected agent').run()
    at.toggle[0].set_value(True)
    button(at, 'Save agent').click().run()
    assert repo.agent(agent.id).settings.rag_enabled
    completion = AsyncMock(return_value=Completion('London [S1]', 20, 5))
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', completion)
    at.sidebar.radio[0].set_value('Chat').run()
    at.chat_input[0].set_value('Where is the deployment?').run()
    finish_chat(at)
    assert not at.exception
    assert any('[S1] deployment.txt' in t.value for t in at.text)
    run_id = repo.runs(agent.id)[0]['id']
    assert knowledge.sources(run_id, agent.id)[0]['available']
    at.sidebar.radio[0].set_value('Activity').run()
    assert not at.exception
    assert any('Query embedding usage' in str(x.value) for x in at.markdown)
    at.sidebar.radio[0].set_value('Knowledge base').run()
    next(c for c in at.checkbox if c.label.startswith('Delete original')).check().run()
    button(at, 'Delete document').click().run()
    assert not at.exception
    assert knowledge.documents(cid) == []
    assert knowledge.sources(run_id, agent.id)[0]['excerpt'] is None
    next(c for c in at.checkbox if c.label.startswith('Delete this collection')).check().run()
    button(at, 'Delete collection').click().run()
    assert not at.exception
    assert knowledge.collections() == []


def test_knowledge_upload_validation_and_missing_key(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ai_agent_pro.rag.repository import KnowledgeRepository
    at, repo = launch(tmp_path, monkeypatch)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: '')
    knowledge = KnowledgeRepository(repo)
    cid = knowledge.create_collection('Documents')
    monkeypatch.setattr('streamlit.file_uploader', lambda *args, **kwargs:
                        SimpleNamespace(name='../escape.txt', getvalue=lambda: b'hello'))
    at.sidebar.radio[0].set_value('Knowledge base').run()
    button(at, 'Upload document').click().run()
    assert not at.exception
    assert any('without directories' in error.value for error in at.error)
    assert knowledge.documents(cid) == []
    monkeypatch.setattr('streamlit.file_uploader', lambda *args, **kwargs:
                        SimpleNamespace(name='notes.txt', getvalue=lambda: b'hello'))
    at.run()
    button(at, 'Upload document').click().run()
    assert not at.exception
    assert button(at, 'Index document').disabled


def test_storage_errors_do_not_expose_raw_exception(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sqlite3
    from ai_agent_pro.rag.repository import KnowledgeRepository
    at, repo = launch(tmp_path, monkeypatch)
    KnowledgeRepository(repo).create_collection('Documents')
    monkeypatch.setattr('streamlit.file_uploader', lambda *args, **kwargs:
                        SimpleNamespace(name='notes.txt', getvalue=lambda: b'hello'))
    def failure(*args, **kwargs):
        raise sqlite3.OperationalError('private disk error sk-privateabcdefgh')
    monkeypatch.setattr('ai_agent_pro.rag.storage.DocumentStorage.upload', failure)
    at.sidebar.radio[0].set_value('Knowledge base').run()
    button(at, 'Upload document').click().run()
    assert not at.exception
    assert any('Local database operation failed' in error.value for error in at.error)
    assert all('sk-private' not in error.value for error in at.error)
