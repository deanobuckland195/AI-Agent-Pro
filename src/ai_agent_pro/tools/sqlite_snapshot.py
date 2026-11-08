"""Pinned, descriptor-only SQLite snapshots; never pass live filenames to SQLite.

The isolated reader holds SQLite's rollback-journal lock bytes while copying.
POSIX record locks belong to a process: opening/closing a raw database descriptor
in the application process could release its existing SQLite locks, so only the
reader subprocess opens the database itself. The host only stats its directory
entry to bind identity. WAL/hot-journal databases fail closed.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import weakref

from ai_agent_pro.tools.contracts import ToolFailure

MAX_DATABASE_BYTES = 64 * 1024**2
SNAPSHOT_TIMEOUT = 1.5
PENDING_BYTE = 0x40000000
LOCK_BYTES = 512


def _environment():
    return {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'}


def _close_lease(process):
    if process.stdin:
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # A reader that failed startup may already have closed stdin.
    try:
        process.wait(timeout=.5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()
    if process.stdout:
        process.stdout.close()


class DatabaseLease:
    """Keep the authorised inode alive without a DB descriptor in the host.

    A long-lived isolated holder prevents inode-number reuse after unlink. It
    performs no SQLite reads or locking and exits on parent EOF or explicit close.
    """
    def __init__(self, context):
        try:
            process = subprocess.Popen(
                [sys.executable, '-I', '-B', '-m', 'ai_agent_pro.tools.sqlite_snapshot', '--lease'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_environment())
        except OSError:
            raise ToolFailure('permission_denied') from None
        self._process = process
        self._cleanup = weakref.finalize(self, _close_lease, process)
        try:
            process.stdin.write(json.dumps(context).encode() + b'\n')
            process.stdin.flush()
            if not select.select([process.stdout], [], [], SNAPSHOT_TIMEOUT)[0]:
                raise ToolFailure('permission_denied')
            value = json.loads(process.stdout.readline(256))
            if value != context['database_identity'] or process.poll() is not None:
                raise ToolFailure('permission_denied')
            self.identity = value
        except Exception:
            self.close()
            raise ToolFailure('permission_denied') from None

    def alive(self):
        return self._process.poll() is None

    def close(self):
        self._cleanup()


def _primitives():
    if (not all(hasattr(os, name) for name in ('O_NOFOLLOW', 'O_DIRECTORY', 'O_NONBLOCK'))
            or os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd
            or os.stat not in os.supports_follow_symlinks):
        raise ToolFailure('permission_denied')


@contextmanager
def parent_directory(context):
    """Walk directories relative to a pinned root, atomically rejecting symlinks."""
    _primitives()
    root, database = Path(context['root']), Path(context['database'])
    descriptors = []
    try:
        parts = database.relative_to(root).parts
        if not parts or any(p in ('.', '..') for p in parts):
            raise ToolFailure('permission_denied')
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(fd)
        info = os.fstat(fd)
        if [info.st_dev, info.st_ino] != list(context['root_identity']):
            raise ToolFailure('permission_denied')
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        yield fd, parts[-1]
    except (OSError, ValueError, KeyError, TypeError):
        raise ToolFailure('permission_denied') from None
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def identity(info):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ToolFailure('permission_denied')
    return [info.st_dev, info.st_ino]


def pin_database(context):
    """Pin a host-authorised entry without opening a SQLite file in the host."""
    try:
        with parent_directory(context) as (directory, name):
            return identity(os.stat(name, dir_fd=directory, follow_symlinks=False))
    except ToolFailure:
        raise


def _sidecars_absent(directory, name):
    for suffix in ('-journal', '-wal', '-shm'):
        try:
            os.stat(name + suffix, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise ToolFailure('permission_denied')


def _copy(context):
    """Only call in the isolated reader: shared POSIX locks block SQLite writers."""
    import fcntl
    expected = context.get('database_identity')
    if type(expected) is not list or len(expected) != 2 or any(type(v) is not int for v in expected):
        raise ToolFailure('permission_denied')
    with parent_directory(context) as (directory, name):
        source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = os.fstat(source)
            if identity(before) != expected:
                raise ToolFailure('permission_denied')
            if not 512 <= before.st_size <= MAX_DATABASE_BYTES:
                raise ToolFailure('resource_limit')
            # Include RESERVED/PENDING and all SHARED lock bytes. A writer must
            # acquire an exclusive RESERVED lock before changing rollback DBs.
            fcntl.lockf(source, fcntl.LOCK_SH | fcntl.LOCK_NB, LOCK_BYTES, PENDING_BYTE, os.SEEK_SET)
            _sidecars_absent(directory, name)
            data = bytearray()
            while block := os.read(source, min(65536, before.st_size + 1 - len(data))):
                data.extend(block)
                if len(data) > before.st_size:
                    raise ToolFailure('permission_denied')
            if (len(data) != before.st_size or data[:16] != b'SQLite format 3\x00'
                    or data[18:20] != b'\x01\x01'):
                raise ToolFailure('permission_denied')
            after = os.fstat(source)
            if (identity(after) != expected or
                    (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ToolFailure('permission_denied')
            # Detect namespace replacement too; correctness does not depend on
            # this check: all bytes above came from the already-open pinned fd.
            if identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) != expected:
                raise ToolFailure('permission_denied')
            _sidecars_absent(directory, name)
            return data
        finally:
            os.close(source)


def snapshot(context):
    """Receive a bounded snapshot from a fixed, credential-free reader process."""
    if context.get('database_identity') is None:
        raise ToolFailure('permission_denied')
    request = {k: context[k] for k in ('root', 'root_identity', 'database', 'database_identity')}
    try:
        process = subprocess.run(
            [sys.executable, '-I', '-B', '-m', 'ai_agent_pro.tools.sqlite_snapshot'],
            input=json.dumps(request).encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=SNAPSHOT_TIMEOUT,
            env=_environment())
    except subprocess.TimeoutExpired:
        raise ToolFailure('timeout') from None
    except OSError:
        raise ToolFailure('permission_denied') from None
    if process.returncode != 0:
        raise ToolFailure('resource_limit' if process.returncode == 11 else 'permission_denied')
    if not 512 <= len(process.stdout) <= MAX_DATABASE_BYTES:
        raise ToolFailure('permission_denied')
    return process.stdout


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (192 * 1024**2, 192 * 1024**2))
        leasing = sys.argv[1:] == ['--lease']
        raw = sys.stdin.buffer.readline(16385) if leasing else sys.stdin.buffer.read(16385)
        if len(raw) > 16384:
            raise ToolFailure('permission_denied')
        context = json.loads(raw)
        if leasing:
            with parent_directory(context) as (directory, name):
                source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                try:
                    if identity(os.fstat(source)) != context['database_identity']:
                        raise ToolFailure('permission_denied')
                    sys.stdout.buffer.write(json.dumps(context['database_identity']).encode() + b'\n')
                    sys.stdout.buffer.flush()
                    sys.stdin.buffer.read()
                finally:
                    os.close(source)
            return
        data = _copy(context)
        sys.stdout.buffer.write(data)
    except ToolFailure as error:
        raise SystemExit(11 if error.code == 'resource_limit' else 10)
    except Exception:
        raise SystemExit(10)


if __name__ == '__main__':
    main()
