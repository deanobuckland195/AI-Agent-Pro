"""Application shell, real navigation shortcuts and agent cards."""
from ai_agent_pro.config import Agent
from ai_agent_pro.presentation.components import ROUTES
from test_ui import launch, button


def test_every_screen_without_data(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    assert at.sidebar.radio[0].value == 'Overview'
    for route in ROUTES:
        at.sidebar.radio[0].set_value(route).run()
        assert not at.exception, route


def test_overview_shortcuts(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    button(at, 'Create an agent').click().run()
    assert at.sidebar.radio[0].value == 'Agents'
    assert next(r for r in at.radio if r.label == 'Agent management').value == 'Create agent'
    at.sidebar.radio[0].set_value('Overview').run()
    button(at, 'Open document library').click().run()
    assert at.sidebar.radio[0].value == 'Knowledge base'


def test_agent_card_actions_and_search(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    first = Agent.create('Build assistant', 'Tests and builds', 'Be precise')
    second = Agent.create('Document assistant', 'Source research', 'Use evidence')
    repo.save_agent(first)
    repo.save_agent(second)
    at.sidebar.radio[0].set_value('Agents').run()
    search = next(x for x in at.text_input if x.label == 'Search agents')
    search.set_value('Source research').run()
    assert len([b for b in at.button if b.label == 'Open chat']) == 1
    button(at, 'Open chat').click().run()
    assert at.sidebar.radio[0].value == 'Chat'
    assert at.sidebar.selectbox[0].value == second.id
    button(at, 'Agent settings').click().run()
    assert at.sidebar.radio[0].value == 'Agents'
    assert next(x for x in at.text_input if x.label == 'Agent name').value == second.name
    next(b for b in at.button if b.key == f'collections_{second.id}').click().run()
    assert at.sidebar.radio[0].value == 'Knowledge base'
    assert at.sidebar.selectbox[0].value == second.id


def test_missing_credentials_and_delete_confirmation(tmp_path, monkeypatch):
    at, repo = launch(tmp_path, monkeypatch)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: '')
    agent = Agent.create('Assistant', '', 'Helpful')
    repo.save_agent(agent)
    at.sidebar.radio[0].set_value('Agents').run()
    button(at, 'Edit agent').click().run()
    assert button(at, 'Delete agent').disabled
    at.sidebar.radio[0].set_value('Chat').run()
    assert at.chat_input[0].disabled
    button(at, 'View collections').click().run()
    assert at.sidebar.radio[0].value == 'Knowledge base'
