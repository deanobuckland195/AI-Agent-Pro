"""Fixed, descriptor-relative practice workspace. No model-chosen roots or handlers."""
from contextlib import contextmanager
import hashlib
import fcntl
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from ai_agent_pro.config import AppError
from ai_agent_pro.safety import redact
from ai_agent_pro.tools.builtins import obj, text

NAME = 'practice_replace__v1_0_0'
SCHEMA = dict(type='function', name=NAME, strict=True,
    description='Propose replacing an explicitly selected practice text file. Pauses for human diff approval; never writes immediately.',
    parameters=obj(path=text(80, 1), content=text(16000)))
MAX_BYTES = 16000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filename(value):
    if (type(value) is not str or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,70}\.(txt|md)', value)
            or any(x in value.lower() for x in ('secret', 'credential', 'password', 'token', 'api', 'private', 'id_rsa'))):
        raise AppError('Use a public practice .txt or .md basename; private files and directories are forbidden.')
    return value


def content(value):
    try:
        size = len(value.encode('utf-8')) if type(value) is str else MAX_BYTES + 1
    except UnicodeError:
        raise AppError('Practice text must be valid UTF-8.') from None
    if (type(value) is not str or size > MAX_BYTES
            or any(ord(c) < 32 and c not in '\n\r\t' for c in value) or redact(value) != value):
        raise AppError('Practice text must be bounded UTF-8 without binary data or recognizable credentials.')
    return value


class PracticeWorkspace:
    def __init__(self, project_root):
        root = Path(project_root)
        if root.is_symlink() or not root.is_absolute():
            raise AppError('Practice project root is not trusted.')
        self.project = root.resolve(strict=True)
        self.path = self.project / 'practice_workspace'
        self.project_identity = self.identity(self.project.stat())

    @staticmethod
    def identity(info):
        return [info.st_dev, info.st_ino]

    @contextmanager
    def directory(self, create=False):
        base = directory = None
        try:
            base = os.open(self.project, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if self.identity(os.fstat(base)) != self.project_identity:
                raise AppError('Practice project identity changed.')
            if create:
                try:
                    os.mkdir('practice_workspace', 0o700, dir_fd=base)
                except FileExistsError:
                    pass
            directory = os.open('practice_workspace', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base)
            yield directory
            if self.identity(os.stat('practice_workspace', dir_fd=base, follow_symlinks=False)) != self.identity(os.fstat(directory)):
                raise AppError('Practice workspace identity changed.')
        except (OSError, UnicodeError):
            raise AppError('Practice file is unavailable or unsafe. Use a regular file in the dedicated workspace.') from None
        finally:
            if directory is not None:
                os.close(directory)
            if base is not None:
                os.close(base)

    def _read(self, directory, name):
        fd = os.open(filename(name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
                raise AppError('Practice files must be small regular files without hard links.')
            raw = os.read(fd, MAX_BYTES + 1)
            value = content(raw.decode('utf-8'))
            return dict(before=value, before_hash=digest(raw), file_identity=self.identity(info),
                        workspace_identity=self.identity(os.fstat(directory)), project_identity=self.project_identity)
        finally:
            os.close(fd)

    def read(self, name):
        with self.directory() as directory:
            return self._read(directory, name)

    def create_example(self):
        # Explicit UI setup only. Never overwrite an existing example.
        with self.directory(create=True) as directory:
            fd = os.open('example.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                os.write(fd, b'This is a practice file.\n')
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(directory)

    def replace(self, proposal, check=lambda: None):
        check()
        name = filename(proposal['arguments']['path'])
        raw = content(proposal['arguments']['content']).encode('utf-8')
        temporary = '.approved-' + uuid4().hex
        with self.directory() as directory:
            try:
                fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise AppError('Practice workspace is busy publishing another approved action. No automatic retry was made.') from None
            current = self._read(directory, name)
            if any(current[k] != proposal[k] for k in current):
                raise AppError('Practice file or workspace changed after proposal; request a new approval.')
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(raw)
                    output.flush()
                    os.fsync(output.fileno())
                # Recheck after staging, before the atomic publication. Replacing
                # a directory entry never follows a raced symlink/hard link.
                if self._read(directory, name) != current:
                    raise AppError('Practice file changed before publication; no replacement was published.')
                if self.path.is_symlink() or self.identity(self.path.stat()) != current['workspace_identity']:
                    raise AppError('Practice workspace changed before publication.')
                check()
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass

    def files(self):
        try:
            with self.directory() as directory:
                names = os.listdir(directory)
            result = []
            for name in names:
                try:
                    self.read(name)
                    result.append(name)
                except AppError:
                    pass
            return sorted(result)[:32]
        except AppError:
            return []
