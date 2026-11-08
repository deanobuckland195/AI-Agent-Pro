import pytest
import os
from pathlib import Path
from ai_agent_pro.config import Agent
from ai_agent_pro.database import Repository


def pytest_configure(config):
    # Pytest's import-rewriting hook is installed before Streamlit loads anyio.
    from support.sitecustomize import isolate
    isolate()
    os.environ['PYTHONPATH'] = str(Path(__file__).parent / 'support')
    import streamlit as st
    st.config.set_option('secrets.files', [])
    st.secrets._secrets = {}


@pytest.fixture(autouse=True)
def cleanup_chat_workers(monkeypatch):
    # Join before mocks are restored, even if an assertion fails mid-generation.
    from ai_agent_pro.execution import ChatJob
    jobs = []
    original = ChatJob.start
    def start(job):
        jobs.append(job)
        return original(job)
    monkeypatch.setattr(ChatJob, 'start', start)
    yield
    for job in jobs:
        job.cancel()
        assert job.wait(5), 'Chat worker did not stop during test cleanup'


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'test.sqlite3')


@pytest.fixture
def agent(repo):
    value = Agent.create('Research assistant', 'A general assistant', 'Be precise and helpful.')
    repo.save_agent(value)
    return value


@pytest.fixture
def conversation(repo, agent):
    return repo.new_conversation(agent.id)
