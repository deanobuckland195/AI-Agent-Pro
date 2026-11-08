from dataclasses import replace
from datetime import datetime, timedelta, timezone
import pytest
from ai_agent_pro.config import AppError
from ai_agent_pro.database import Repository


def finish(repo, rid, answer='Result'):
    repo.finish_run(rid, 'completed', 1, 10, 20, .1, [], answer=answer)


def test_create_update_and_reload(repo, agent):
    changed = replace(agent, name="O'Reilly", settings=replace(agent.settings, max_steps=3))
    repo.save_agent(changed)
    assert Repository(repo.path).agent(agent.id) == changed


def test_messages_and_runs_survive_reopen(repo, agent, conversation):
    rid = repo.begin_run(conversation, agent, 'hello')
    finish(repo, rid)
    reopened = Repository(repo.path)
    assert [m['content'] for m in reopened.messages(conversation)] == ['hello', 'Result']
    assert reopened.runs(agent.id)[0]['input_tokens'] == 10
    assert reopened.runs(agent.id)[0]['status'] == 'completed'


def test_conversation_ownership(repo, agent, conversation):
    with pytest.raises(AppError):
        repo.require_conversation(conversation, 'other-agent')


def test_sql_values_and_cascades(repo, agent, conversation):
    rid = repo.begin_run(conversation, agent, "'; DROP TABLE agents; --")
    finish(repo, rid)
    assert len(repo.agents()) == 1
    repo.delete_agent(agent.id)
    assert repo.messages(conversation) == []
    assert repo.conversations(agent.id) == []


def test_clear_removes_messages_and_runs(repo, agent, conversation):
    rid = repo.begin_run(conversation, agent, 'hello')
    finish(repo, rid)
    repo.clear_conversation(conversation)
    assert repo.messages(conversation) == []
    assert repo.runs(agent.id) == []
    assert repo.conversations(agent.id)[0]['title'] == 'New conversation'


def test_active_run_lock(repo, agent, conversation):
    repo.begin_run(conversation, agent, 'hello')
    for operation in (lambda: repo.begin_run(conversation, agent, 'again'),
                      lambda: repo.clear_conversation(conversation),
                      lambda: repo.delete_agent(agent.id)):
        with pytest.raises(AppError):
            operation()
    assert len(repo.messages(conversation)) == 1


def test_recover_requires_expired_deadline(repo, agent, conversation):
    rid = repo.begin_run(conversation, agent, 'hello')
    with pytest.raises(AppError):
        repo.recover_run(rid)
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with repo.connect() as db:
        db.execute('UPDATE runs SET started_at=? WHERE id=?', (old, rid))
    repo.recover_run(rid)
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'
    with pytest.raises(AppError):
        finish(repo, rid)


def test_secrets_not_persisted(repo, agent, conversation):
    repo.save_agent(replace(agent, instructions='Use sk-testabcdefgh carefully'))
    rid = repo.begin_run(conversation, agent, 'password=hunter2')
    finish(repo, rid, 'sk-testabcdefgh')
    assert 'hunter2' not in str(repo.messages(conversation))
    assert 'sk-testabcdefgh' not in repo.agent(agent.id).instructions
