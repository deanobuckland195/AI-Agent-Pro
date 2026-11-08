"""Regressions for the demonstrated SQLite pathname-swap boundary failure."""
import asyncio
from pathlib import Path
import shutil
import sqlite3

import pytest

from ai_agent_pro.tools import knowledge, sqlite_snapshot
from ai_agent_pro.tools.contracts import ToolFailure
from ai_agent_pro.tools.permissions import ToolPermissions
from test_tools import tools, call, allow
from test_rag_retrieval import kb, indexed


@pytest.fixture
def scoped(tools, repo, agent, kb):
    indexed(kb)
    allow(tools, 'knowledge_search', agent.id)
    grant = tools.permissions.authorize(agent.id, tools.registry.get('knowledge_search', '1.0.0'))
    return tools.permissions.context(grant)


def outside_copy(repo, tmp_path):
    # A disposable sibling outside the authority's project root.
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    path = outside / 'canary.sqlite3'
    shutil.copyfile(repo.path, path)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE document_chunks SET text=?', ('London outside canary',))
    return path


def test_original_sqlite_connect_swap_cannot_redirect_open(scoped, repo, tmp_path, monkeypatch):
    outside = outside_copy(repo, tmp_path)
    original = sqlite3.connect
    saved = tmp_path / 'saved.sqlite3'
    opened = []
    def swap_at_sqlite_open(database, *args, **kwargs):
        opened.append(database)
        if database == ':memory:':
            return original(database, *args, **kwargs)
        # This reproduces the original failure: swap after path checks, open
        # outside data, restore the exact original inode before the final check.
        repo.path.rename(saved)
        repo.path.symlink_to(outside)
        try:
            return original(database, *args, **kwargs)
        finally:
            repo.path.unlink()
            saved.rename(repo.path)
    monkeypatch.setattr(sqlite3, 'connect', swap_at_sqlite_open)
    result = knowledge.run({'query':'London', 'max_results':5}, scoped)
    knowledge.revalidate(result, scoped)
    assert opened and all(path == ':memory:' for path in opened)
    assert result['sources'] and all('outside canary' not in s['excerpt'] for s in result['sources'])


@pytest.mark.parametrize('kind', ['symlink', 'regular', 'hardlink'])
def test_database_replaced_after_binding_is_denied(scoped, repo, tmp_path, kind):
    outside = outside_copy(repo, tmp_path)
    saved = tmp_path / 'saved.sqlite3'
    repo.path.rename(saved)
    if kind == 'symlink':
        repo.path.symlink_to(outside)
    elif kind == 'hardlink':
        repo.path.hardlink_to(outside)
    else:
        shutil.copyfile(outside, repo.path)
    try:
        with pytest.raises(ToolFailure, match='not approved'):
            knowledge.run({'query':'London', 'max_results':5}, scoped)
    finally:
        repo.path.unlink()
        saved.rename(repo.path)


def test_swap_immediately_before_descriptor_open_is_denied(scoped, repo, tmp_path, monkeypatch):
    outside = outside_copy(repo, tmp_path)
    saved = tmp_path / 'saved.sqlite3'
    original = sqlite_snapshot.os.open
    def race(path, flags, *args, **kwargs):
        if path == repo.path.name and 'dir_fd' in kwargs:
            repo.path.rename(saved)
            repo.path.symlink_to(outside)
            try:
                return original(path, flags, *args, **kwargs)
            finally:
                repo.path.unlink()
                saved.rename(repo.path)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(sqlite_snapshot.os, 'open', race)
    monkeypatch.setattr(sqlite_snapshot.os, 'supports_dir_fd', sqlite_snapshot.os.supports_dir_fd | {race})
    with pytest.raises((ToolFailure, OSError)):
        sqlite_snapshot._copy(scoped)


def test_open_descriptor_never_reads_swapped_target(scoped, repo, tmp_path, monkeypatch):
    outside = outside_copy(repo, tmp_path)
    saved = tmp_path / 'saved.sqlite3'
    original = sqlite_snapshot.os.read
    swapped = False
    def race(fd, count):
        nonlocal swapped
        if not swapped:
            swapped = True
            repo.path.rename(saved)
            repo.path.symlink_to(outside)
        value = original(fd, count)
        assert b'outside canary' not in value
        return value
    monkeypatch.setattr(sqlite_snapshot.os, 'read', race)
    try:
        with pytest.raises(ToolFailure):
            sqlite_snapshot._copy(scoped)
    finally:
        if repo.path.is_symlink():
            repo.path.unlink()
            saved.rename(repo.path)
    assert swapped


def test_parent_directory_symlink_swap_after_binding_is_denied(tmp_path, repo, agent, kb):
    indexed(kb)
    folder = tmp_path / 'data'
    folder.mkdir()
    target = folder / 'knowledge.sqlite3'
    shutil.copyfile(repo.path, target)
    permissions = ToolPermissions(tmp_path, target)
    permissions.allow(agent.id, 'knowledge_search')
    from ai_agent_pro.tools.executor import ToolExecutor
    tools = ToolExecutor(permissions)
    folder.rename(tmp_path / 'original-data')
    folder.symlink_to(tmp_path / 'original-data', target_is_directory=True)
    result = call(tools, 'knowledge_search', {'query':'London','max_results':5}, agent=agent.id)
    assert result.error_code == 'permission_denied' and result.data is None


@pytest.mark.parametrize('suffix', ['-wal', '-shm', '-journal'])
def test_sidecars_including_symlinks_are_never_opened(scoped, repo, tmp_path, suffix):
    outside = tmp_path / 'outside-private.txt'
    outside.write_bytes(b'private canary')
    sidecar = Path(str(repo.path) + suffix)
    sidecar.symlink_to(outside)
    with pytest.raises(ToolFailure):
        knowledge.run({'query':'London','max_results':5}, scoped)
    assert outside.read_bytes() == b'private canary'


def test_wal_database_fails_closed_even_with_no_sidecars(scoped, repo):
    with sqlite3.connect(repo.path) as db:
        assert db.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
    with pytest.raises(ToolFailure):
        knowledge.run({'query':'London','max_results':5}, scoped)


def test_active_writer_blocks_snapshot_without_releasing_app_locks(scoped, repo):
    with sqlite3.connect(repo.path, timeout=.05) as writer:
        writer.execute('BEGIN IMMEDIATE')
        with pytest.raises(ToolFailure):
            knowledge.run({'query':'London','max_results':5}, scoped)
        with sqlite3.connect(repo.path, timeout=.05) as contender:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                contender.execute('BEGIN IMMEDIATE')
    assert knowledge.run({'query':'London','max_results':5}, scoped)['sources']


def test_snapshot_is_read_only_and_cannot_attach_other_files(scoped, repo, tmp_path):
    before = repo.path.read_bytes()
    with knowledge.connection(scoped) as db:
        assert db.execute('SELECT id FROM agents').fetchone()
        for statement in ['DELETE FROM agents', 'PRAGMA query_only=OFF',
                          "ATTACH ':memory:' AS other", "SELECT load_extension('unsafe')"]:
            with pytest.raises(sqlite3.DatabaseError):
                db.execute(statement)
    assert repo.path.read_bytes() == before


def test_missing_binding_and_unsupported_deserialize_fail_closed(scoped, monkeypatch):
    context = dict(scoped, database_identity=None)
    with pytest.raises(ToolFailure):
        knowledge.run({'query':'London','max_results':5}, context)
    opened = []
    class NoDeserialize:
        def close(self):
            opened.append('closed')
    def unsupported(path, *args, **kwargs):
        assert path == ':memory:'
        opened.append(path)
        return NoDeserialize()
    monkeypatch.setattr(sqlite3, 'connect', unsupported)
    with pytest.raises(ToolFailure):
        knowledge.run({'query':'London','max_results':5}, scoped)
    assert opened == [':memory:', 'closed']


def test_unavailable_no_follow_support_fails_closed(scoped, monkeypatch):
    monkeypatch.setattr(sqlite_snapshot.os, 'supports_dir_fd', set())
    with pytest.raises(ToolFailure):
        sqlite_snapshot.pin_database(scoped)


def test_oversized_snapshot_rejected_before_copy(scoped, repo):
    with repo.path.open('r+b') as file:
        file.truncate(sqlite_snapshot.MAX_DATABASE_BYTES + 1)
    with pytest.raises(ToolFailure) as caught:
        knowledge.run({'query':'London','max_results':5}, scoped)
    assert caught.value.code == 'resource_limit'


def test_identity_is_stable_for_legitimate_updates_but_not_replacement(scoped, repo, agent, kb, tmp_path):
    with sqlite3.connect(repo.path) as db:
        db.execute('UPDATE document_chunks SET text=?', ('London updated authorised text',))
    assert 'updated authorised' in knowledge.run({'query':'London','max_results':5}, scoped)['sources'][0]['excerpt']
    replacement = tmp_path / 'replacement.sqlite3'
    shutil.copyfile(repo.path, replacement)
    replacement.replace(repo.path)
    with pytest.raises(ToolFailure):
        knowledge.run({'query':'London','max_results':5}, scoped)
    fresh = ToolPermissions(tmp_path, repo.path)
    fresh.allow(agent.id, 'knowledge_search')
    from ai_agent_pro.tools.executor import ToolExecutor
    result = call(ToolExecutor(fresh), 'knowledge_search', {'query':'London','max_results':5}, agent=agent.id)
    assert result.ok


def test_directory_swap_at_descriptor_open_fails_closed(scoped, repo, tmp_path, monkeypatch):
    folder = tmp_path / 'nested'
    folder.mkdir()
    target = folder / 'database.sqlite3'
    shutil.copyfile(repo.path, target)
    permissions = ToolPermissions(tmp_path, target)
    context = dict(scoped, database=str(target), database_identity=permissions.database_identity)
    saved = tmp_path / 'saved-dir'
    original = sqlite_snapshot.os.open
    def race(path, flags, *args, **kwargs):
        if path == 'nested' and 'dir_fd' in kwargs:
            folder.rename(saved)
            folder.symlink_to(saved, target_is_directory=True)
            try:
                return original(path, flags, *args, **kwargs)
            finally:
                folder.unlink()
                saved.rename(folder)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(sqlite_snapshot.os, 'open', race)
    monkeypatch.setattr(sqlite_snapshot.os, 'supports_dir_fd', sqlite_snapshot.os.supports_dir_fd | {race})
    with pytest.raises(ToolFailure):
        sqlite_snapshot._copy(context)


def test_cancelling_worker_stops_snapshot_descendants(tools, monkeypatch, tmp_path):
    import psutil
    import time
    from test_tool_execution import fake_worker
    # Emulate a worker waiting on its fixed reader, not model-executable code.
    pidfile = tmp_path / 'reader.pid'
    script = '''import subprocess,sys,time
from pathlib import Path
reader=subprocess.Popen([sys.executable,'-I','-B','-c','import time; time.sleep(30)'])
Path(''' + repr(str(pidfile)) + ''').write_text(str(reader.pid))
time.sleep(30)
'''
    children = fake_worker(monkeypatch, script)
    allow(tools, 'calculator')
    observed = []
    async def scenario():
        task = asyncio.create_task(tools.execute('agent-a','calculator','1.0.0',{'expression':'1'}))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if pidfile.exists():
                observed.append(psutil.Process(int(pidfile.read_text())))
                break
            await asyncio.sleep(.01)
        assert observed, 'Reader descendant did not start'
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert children[0].returncode is not None
    for child in observed:
        try:
            child.wait(timeout=2)
        except psutil.TimeoutExpired:
            assert child.status() == psutil.STATUS_ZOMBIE  # Exited, awaiting OS reaping.


def test_closed_lease_denies_knowledge_without_starting_worker(scoped, tools, agent, monkeypatch):
    tools.permissions.close()
    async def forbidden(*args):
        pytest.fail('Closed knowledge binding must fail before execution')
    monkeypatch.setattr(tools, '_process', forbidden)
    result = call(tools, 'knowledge_search', {'query':'London','max_results':5}, agent=agent.id)
    assert result.error_code == 'permission_denied' and result.data is None


def test_lease_holds_original_inode_until_authority_closes(scoped, tools, repo, tmp_path):
    holder = tools.permissions._database_lease
    original_identity = list(scoped['database_identity'])
    repo.path.unlink()
    # The unlinked authorised inode is still held open in the isolated holder.
    assert holder.alive()
    with sqlite3.connect(repo.path) as db:
        db.execute('CREATE TABLE replacement(value TEXT)')
    replacement = repo.path.stat()
    assert [replacement.st_dev, replacement.st_ino] != original_identity
    with pytest.raises(ToolFailure):
        knowledge.run({'query':'London','max_results':5}, scoped)
    tools.permissions.close()
    assert not holder.alive()


def test_lease_death_during_read_invalidates_result(scoped, tools, agent, monkeypatch):
    original = tools._process
    async def terminate(*args):
        response = await original(*args)
        tools.permissions._database_lease.close()
        return response
    monkeypatch.setattr(tools, '_process', terminate)
    result = call(tools, 'knowledge_search', {'query':'London','max_results':5}, agent=agent.id)
    assert result.error_code == 'permission_denied' and result.data is None


def test_reader_startup_failure_has_no_unsafe_fallback(scoped, repo, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('private startup details')
    monkeypatch.setattr(sqlite_snapshot.subprocess, 'Popen', unavailable)
    with pytest.raises(ToolFailure) as caught:
        knowledge.run({'query':'London','max_results':5}, scoped)
    assert 'private' not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        ToolPermissions(tmp_path, repo.path)
    assert 'private' not in str(caught.value)
