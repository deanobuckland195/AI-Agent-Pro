"""Release checks use only synthetic configuration and disposable databases."""
import os
import socket
import sqlite3
import subprocess
import sys

import pytest

from ai_agent_pro.database import Repository
from ai_agent_pro.rag.repository import KnowledgeRepository
from ai_agent_pro.rag.storage import DocumentStorage
from ai_agent_pro.workflow_contracts import software_workflow
from ai_agent_pro.workflow_store import WorkflowStore
from support.sitecustomize import ExternalNetworkBlocked


@pytest.mark.parametrize('host', ['example.invalid', '203.0.113.1', '2001:db8::1'])
def test_external_network_is_blocked_before_connection(host):
    with pytest.raises(ExternalNetworkBlocked):
        socket.getaddrinfo(host, 443)
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    if host != 'example.invalid':
        with socket.socket(family) as client:
            with pytest.raises(ExternalNetworkBlocked):
                client.connect((host, 443))
            with pytest.raises(ExternalNetworkBlocked):
                client.connect_ex((host, 443))


def test_private_streamlit_files_are_not_consulted(monkeypatch):
    import streamlit as st
    from streamlit.runtime.secrets import Secrets
    from ai_agent_pro.ui import api_key
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail('A test attempted to load a private secrets file.')
    # Secrets rejects instance method assignment; patch the class so pytest can
    # restore the loader while still failing before any secrets path is read.
    monkeypatch.setattr(Secrets, '_parse_file_path', forbidden)
    monkeypatch.setattr(st.secrets, '_secrets', None)
    assert st.config.get_option('secrets.files') == []
    assert api_key() == ''


def test_python_server_children_inherit_isolation():
    # A synthetic canary demonstrates scrubbing without consulting real values.
    env = dict(os.environ, OPENAI_API_KEY='synthetic-release-canary')
    code = '''import os,socket
assert os.environ['OPENAI_API_KEY'] == ''
assert os.environ['STREAMLIT_SECRETS_FILES'].endswith('empty-secrets.toml')
try:
    socket.getaddrinfo('example.invalid',443)
except RuntimeError:
    print('isolated')
else:
    raise AssertionError('External DNS was permitted')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout.strip() == 'isolated'


def test_sqlite_backup_restore_preserves_relations_and_is_independent(repo, agent, conversation, tmp_path):
    rid = repo.begin_run(conversation, agent, 'Synthetic backup question')
    repo.finish_run(rid, 'completed', 1, 10, 5, 0.1, [], answer='Synthetic backup answer')
    knowledge = KnowledgeRepository(repo)
    collection = knowledge.create_collection('Synthetic collection')
    knowledge.assign(agent.id, [collection])
    document = DocumentStorage(knowledge).upload(collection, 'sample.txt', b'Synthetic document')
    workflows = WorkflowStore(repo)
    definition = workflows.save(software_workflow([agent.id] * 5))
    wid = workflows.start(*definition, 'Synthetic backup workflow')
    backup = tmp_path / 'backup.sqlite3'
    restored_path = tmp_path / 'restored.sqlite3'
    with repo.connect() as source, sqlite3.connect(backup) as destination:
        source.backup(destination)
    with sqlite3.connect(backup) as source, sqlite3.connect(restored_path) as destination:
        source.backup(destination)
    restored = Repository(restored_path)
    with restored.connect() as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert db.execute('PRAGMA foreign_key_check').fetchall() == []
        assert db.execute('PRAGMA user_version').fetchone()[0] == 5
    assert restored.agent(agent.id) == agent
    assert restored.messages(conversation) == repo.messages(conversation)
    assert restored.runs(agent.id) == repo.runs(agent.id)
    assert KnowledgeRepository(restored).assigned(agent.id) == [collection]
    assert KnowledgeRepository(restored).document(document)['original'] == b'Synthetic document'
    assert WorkflowStore(restored).get(wid)['data'] == workflows.get(wid)['data']
    assert WorkflowStore(restored).steps(wid) == workflows.steps(wid)
    KnowledgeRepository(restored).delete_document(document)
    assert knowledge.document(document)['original'] == b'Synthetic document'
