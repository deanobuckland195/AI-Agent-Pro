"""Worker lifecycle, session isolation, cancellation races and preview cleanup."""
import asyncio
import threading
from unittest.mock import AsyncMock

from ai_agent_pro.execution import ChatJob
from test_streaming import fake_stream


def test_worker_completion_and_preview_are_session_owned(repo, agent, conversation):
    job = ChatJob(repo, fake_stream('Safe answer '), agent.id, conversation, 'Hi').start()
    assert job.wait()
    assert job.snapshot().state == 'Completed'
    assert job.snapshot().text == 'Safe answer '
    assert not job.cancel()
    assert len(repo.messages(conversation)) == 2


def test_worker_stop_while_waiting_closes_provider_and_discards_preview(repo, agent, conversation):
    arrived, closed = threading.Event(), threading.Event()
    async def complete(on_delta, **kwargs):
        try:
            on_delta('A provisional answer ')
            arrived.set()
            await asyncio.Event().wait()
        finally:
            closed.set()
    provider = type('Fake', (), {'complete': AsyncMock(side_effect=complete)})()
    job = ChatJob(repo, provider, agent.id, conversation, 'Hi').start()
    assert arrived.wait(3)
    assert job.snapshot().text == 'A provisional answer '
    assert job.cancel()
    assert not job.cancel()  # Do not cancel the cleanup itself on a second click.
    assert job.wait() and closed.is_set()
    assert job.snapshot().state == 'Cancelled' and job.snapshot().text == ''
    assert repo.runs(agent.id)[0]['status'] == 'interrupted'
    assert len(repo.messages(conversation)) == 1


def test_worker_cancel_before_start_and_abandoned_session(repo, agent, conversation):
    provider = fake_stream()
    job = ChatJob(repo, provider, agent.id, conversation, 'Hi')
    assert job.cancel()
    job.start()
    assert job.wait() and job.snapshot().state == 'Cancelled'
    provider.complete.assert_not_called()

    async def stalled(on_delta, **kwargs):
        await asyncio.Event().wait()
    provider.complete.side_effect = stalled
    job = ChatJob(repo, provider, agent.id, conversation, 'Retry', lease_seconds=.05).start()
    assert job.wait() and job.snapshot().state == 'Cancelled'
    assert all(r['status'] == 'interrupted' for r in repo.runs(agent.id))


def test_worker_error_removes_preview_and_never_leaks_raw_exception(repo, agent, conversation):
    job = ChatJob(repo, fake_stream(fail=RuntimeError('sk-privateabcdefgh')),
                  agent.id, conversation, 'Hi').start()
    assert job.wait()
    snapshot = job.snapshot()
    assert snapshot.state == 'Failed' and snapshot.text == ''
    assert 'private' not in snapshot.error
    assert len(repo.messages(conversation)) == 1


def test_existing_run_lock_blocks_duplicate_worker(repo, agent, conversation):
    repo.begin_run(conversation, agent, 'Existing request')
    provider = fake_stream()
    job = ChatJob(repo, provider, agent.id, conversation, 'Duplicate').start()
    assert job.wait() and job.snapshot().state == 'Failed'
    provider.complete.assert_not_called()
    assert len(repo.messages(conversation)) == 1


def test_late_stop_during_atomic_save_does_not_claim_cancellation(repo, agent, conversation, monkeypatch):
    saving, release = threading.Event(), threading.Event()
    original = repo.finish_run
    def finish(*args, **kwargs):
        saving.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(repo, 'finish_run', finish)
    job = ChatJob(repo, fake_stream(), agent.id, conversation, 'Hi').start()
    assert saving.wait(3)
    assert job.snapshot().state == 'Saving'
    assert not job.cancel()
    release.set()
    assert job.wait() and job.snapshot().state == 'Completed'
    assert len(repo.messages(conversation)) == 2
