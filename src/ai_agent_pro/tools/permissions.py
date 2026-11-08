"""Host-owned grants. Never deserialize permissions or principals from model input."""
from dataclasses import dataclass
from pathlib import Path
import threading
from time import monotonic

from ai_agent_pro.tools.contracts import ToolFailure
from ai_agent_pro.tools.filesystem import PROJECT_ROOT, read_file, relative_path
from ai_agent_pro.tools.sqlite_snapshot import DatabaseLease, pin_database


@dataclass(frozen=True)
class Grant:
    agent_id: str
    revision: int
    files: tuple[str, ...]


class ToolPermissions:
    def __init__(self, project_root=PROJECT_ROOT, knowledge_database=None, *, bind_knowledge=True):
        # Binding a root/database is trusted application setup, not a tool argument.
        root = Path(project_root)
        if root.is_symlink():
            raise ValueError('Project root must not be a symlink')
        self.root = root.resolve(strict=True)
        info = self.root.stat()
        if not self.root.is_dir():
            raise ValueError('Project root must be a directory')
        self.identity = (info.st_dev, info.st_ino)
        database = Path(knowledge_database) if knowledge_database else self.root / 'data/agentpro.sqlite3'
        if not database.is_absolute():
            database = self.root / database
        # Other project databases cannot be attached to this authority.
        if not database.is_relative_to(self.root) or '..' in database.parts:
            raise ValueError('Knowledge database must be inside the approved project')
        self.database = database
        self.database_identity = None
        self._database_lease = None
        # A missing database permits non-knowledge tools only. A new authority
        # must be created once the DB exists; never silently trust a replacement.
        try:
            if not bind_knowledge:
                raise ToolFailure('permission_denied')
            self.database_identity = pin_database(dict(root=str(self.root), root_identity=list(self.identity),
                                                       database=str(self.database)))
            self._database_lease = DatabaseLease(dict(root=str(self.root), root_identity=list(self.identity),
                database=str(self.database), database_identity=self.database_identity))
        except ToolFailure:
            if bind_knowledge and (self.database.exists() or self.database.is_symlink()):
                raise ValueError('Knowledge database cannot be bound securely') from None
        self._lock = threading.RLock()
        self._grants, self._files, self._revisions = {}, {}, {}

    def close(self):
        if self._database_lease is not None:
            self._database_lease.close()

    def _agent(self, agent_id):
        if type(agent_id) is not str or not agent_id or len(agent_id) > 80:
            raise ToolFailure('permission_denied')

    def allow(self, agent_id, name, version='1.0.0'):
        self._agent(agent_id)
        if type(name) is not str or type(version) is not str:
            raise ValueError('A specific tool version is required')
        with self._lock:
            self._grants.setdefault(agent_id, set()).add((name, version))
            self._revisions[agent_id] = self._revisions.get(agent_id, 0) + 1

    def revoke(self, agent_id, name, version='1.0.0'):
        with self._lock:
            self._grants.setdefault(agent_id, set()).discard((name, version))
            self._revisions[agent_id] = self._revisions.get(agent_id, 0) + 1

    def approve_files(self, agent_id, files):
        self._agent(agent_id)
        if type(files) not in (list, tuple) or len(files) > 32:
            raise ValueError('Approve at most 32 explicit files')
        names = tuple(sorted({relative_path(p) for p in files}))
        for name in names:
            read_file(str(self.root), self.identity, name)
        with self._lock:
            self._files[agent_id] = names
            self._revisions[agent_id] = self._revisions.get(agent_id, 0) + 1

    def authorize(self, agent_id, definition):
        self._agent(agent_id)
        with self._lock:
            if (definition.name, definition.version) not in self._grants.get(agent_id, set()):
                raise ToolFailure('permission_denied')
            if definition.permission == 'knowledge.read' and (
                    self._database_lease is None or not self._database_lease.alive()):
                raise ToolFailure('permission_denied')
            return Grant(agent_id, self._revisions.get(agent_id, 0), self._files.get(agent_id, ()))

    def recheck(self, grant, definition):
        current = self.authorize(grant.agent_id, definition)
        if current != grant:
            raise ToolFailure('permission_denied')

    def context(self, grant):
        return dict(agent_id=grant.agent_id, root=str(self.root), root_identity=list(self.identity),
                    files=list(grant.files), database=str(self.database),
                    database_identity=list(self.database_identity) if self.database_identity is not None else None)


class ToolBudget:
    """Optional shared, host-owned budget across manual calls; no model loop."""
    def __init__(self, max_calls=10, seconds=30, result_bytes=131072):
        if type(max_calls) is not int or not 1 <= max_calls <= 100:
            raise ValueError('Invalid call budget')
        if type(seconds) not in (int, float) or not .05 <= seconds <= 60:
            raise ValueError('Invalid deadline')
        if type(result_bytes) is not int or not 1 <= result_bytes <= 1048576:
            raise ValueError('Invalid result budget')
        self._calls, self._remaining = max_calls, result_bytes
        self._deadline = monotonic() + seconds
        self._lock = threading.Lock()

    def reserve(self):
        with self._lock:
            remaining = self._deadline - monotonic()
            if self._calls <= 0 or remaining <= 0 or self._remaining <= 0:
                raise ToolFailure('budget_exhausted')
            self._calls -= 1
            return remaining, self._remaining

    def charge(self, size):
        with self._lock:
            if size > self._remaining or monotonic() >= self._deadline:
                raise ToolFailure('budget_exhausted')
            self._remaining -= size
