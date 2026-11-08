"""Actual Streamlit chat interactions with controlled asynchronous generation."""
import asyncio
import threading
from unittest.mock import AsyncMock

from ai_agent_pro.config import Agent
from test_ui import launch, button, finish_chat


def start_slow_chat(tmp_path, monkeypatch):
    arrived = threading.Event()
    async def slow(on_delta, **kwargs):
        on_delta('First progressive text ')
        arrived.set()
        await asyncio.Event().wait()
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', AsyncMock(side_effect=slow))
    at, repo = launch(tmp_path, monkeypatch)
    repo.save_agent(Agent.create('Streaming assistant', '', 'Be helpful'))
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    at.sidebar.radio[0].set_value('Chat').run()
    at.chat_input[0].set_value('Hello').run()
    assert arrived.wait(3)
    at.run()
    return at, repo


def test_progressive_ui_stop_and_duplicate_submission_guard(tmp_path, monkeypatch):
    at, repo = start_slow_chat(tmp_path, monkeypatch)
    assert at.chat_input[0].disabled
    assert button(at, 'New conversation').disabled
    assert any('Generating' in info.value for info in at.info)
    assert any(m.value == 'First progressive text' for m in at.markdown)
    assert not button(at, 'Stop generation').disabled
    job = at.session_state['_chat_job']
    at.run()
    assert at.session_state['_chat_job'] is job
    assert len(repo.runs(repo.agents()[0].id)) == 1
    button(at, 'Stop generation').click().run()
    finish_chat(at)
    assert any('Cancelled' in info.value for info in at.info)
    assert not at.chat_input[0].disabled
    assert not any('First progressive' in m.value for m in at.markdown)
    cid = repo.conversations(repo.agents()[0].id)[0]['id']
    assert [m['role'] for m in repo.messages(cid)] == ['user']


def test_navigation_away_cancels_owned_request(tmp_path, monkeypatch):
    at, repo = start_slow_chat(tmp_path, monkeypatch)
    job = at.session_state['_chat_job']
    at.sidebar.radio[0].set_value('Overview').run()
    assert job.wait()
    assert job.snapshot().state == 'Cancelled'
    assert repo.runs(repo.agents()[0].id)[0]['status'] == 'interrupted'
    assert not at.exception
