"""Functional Streamlit decisions, pause controls and reload durability."""
import json
from unittest.mock import AsyncMock

import pytest

from ai_agent_pro.config import Agent
from ai_agent_pro.practice import PracticeWorkspace, NAME
from ai_agent_pro.provider import Completion, FunctionCall
from test_ui import launch, finish_chat, button
from test_tools import tools


@pytest.mark.parametrize('decision', ['Approve file change', 'Deny file change', 'Cancel paused execution'])
def test_ui_diff_and_explicit_decision(tmp_path, monkeypatch, tools, decision):
    at, repo = launch(tmp_path, monkeypatch)
    agent = Agent.create('Practice editor', '', 'Propose only requested changes')
    repo.save_agent(agent)
    practice = PracticeWorkspace(tmp_path)
    practice.create_example()
    monkeypatch.setattr('ai_agent_pro.presentation.workspace.practice_workspace', lambda: practice)
    monkeypatch.setattr('ai_agent_pro.presentation.workspace.tool_executor', lambda path: tools)
    monkeypatch.setattr('ai_agent_pro.ui.api_key', lambda: 'offline-test-key')
    call = FunctionCall(NAME, json.dumps(dict(path='example.txt', content='Approved new text.\n')), 'write_ui')
    model = AsyncMock(side_effect=[Completion('',20,10,(call,),
        (dict(type='function_call', name=call.name, arguments=call.arguments, call_id=call.call_id),)),
        Completion('Change completed.',20,10)])
    monkeypatch.setattr('ai_agent_pro.provider.OpenAIProvider.complete', model)
    at.sidebar.radio[0].set_value('Chat').run()
    next(c for c in at.checkbox if c.label == 'Allow practice file change proposals').check().run()
    next(m for m in at.multiselect if m.label == 'Explicitly approved practice files').select('example.txt').run()
    at.chat_input[0].set_value('Replace the practice file; this message is not an approval.').run()
    finish_chat(at)
    assert repo.runs(agent.id)[0]['status'] == 'waiting_approval'
    assert model.call_count == 1
    assert at.chat_input[0].disabled
    assert any(str(practice.path/'example.txt') in t.value for t in at.text)
    assert any('-This is a practice file.' in c.value and '+Approved new text.' in c.value for c in at.code)
    assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
    # A rerender must reconstruct the pending proposal from SQLite.
    at.run()
    button(at, decision).click().run()
    if decision == 'Approve file change':
        finish_chat(at)
        assert (practice.path/'example.txt').read_text() == 'Approved new text.\n'
        assert model.call_count == 2
        assert repo.runs(agent.id)[0]['requests'] == 2
    else:
        assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
        assert model.call_count == 1
    assert not at.exception
