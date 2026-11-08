from dataclasses import replace
import logging
import pytest
from ai_agent_pro.config import Agent, AppError, Settings
from ai_agent_pro.safety import RedactingFilter, instructions_for, redact, validate_text


@pytest.mark.parametrize('field,value', [('model', 'invented-model'), ('max_steps', 4),
    ('max_steps', True), ('max_output_tokens', 0), ('token_budget', 100001),
    ('timeout_seconds', 181), ('history_messages', -1)])
def test_invalid_settings(field, value):
    with pytest.raises(AppError):
        replace(Settings(), **{field: value}).validate()


@pytest.mark.parametrize('text', ['', '   ', 'a' * 81, 'bad\x00name'])
def test_invalid_agent_names(text):
    with pytest.raises(AppError):
        Agent.create(text, '', 'Helpful')


def test_unicode_and_optional_description():
    agent = Agent.create('Assistant 🌍', '', 'Explain clearly')
    assert agent.settings.validate() == Settings()
    assert validate_text('hello 🌍', 'Message', 10) == 'hello 🌍'


@pytest.mark.parametrize('text', ['sk-proj-abcdefghijk', 'password=hunter2',
                                 'api_key: supersecret', 'Bearer abcdefghi'])
def test_redacts_common_secrets(text):
    assert redact(text) == '[REDACTED]'


def test_logging_redaction():
    record = logging.LogRecord('test', logging.WARNING, '', 0, 'secret %s', ('sk-testabcdefgh',), None)
    assert RedactingFilter().filter(record)
    assert record.getMessage() == 'secret [REDACTED]'


def test_boundaries(agent):
    instructions = instructions_for(agent, 2, 2)
    assert 'untrusted data' in instructions
    assert 'no tools' in instructions.lower()
    assert 'Review the previous draft' in instructions
