"""Descriptor-relative project reads; never follow symlinks or hard links."""
import os
import re
from pathlib import Path
import stat

from ai_agent_pro.safety import SECRET_PATTERNS
from ai_agent_pro.tools.contracts import ToolFailure

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_FILE_BYTES = 262144
MAX_TOTAL_BYTES = 1048576
SUFFIXES = {'.py', '.md', '.rst', '.txt', '.js', '.jsx', '.ts', '.tsx', '.css', '.html', '.sql'}
BLOCKED_DIRS = {'data', 'env', 'venv', 'node_modules', 'private', 'credentials', 'secrets'}


def relative_path(value):
    if type(value) is not str or not 1 <= len(value) <= 240 or value != value.strip():
        raise ToolFailure('permission_denied')
    if any(c in value for c in '\\:%*?[]') or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ToolFailure('permission_denied')
    parts = value.split('/')
    if any(not p or p in ('.', '..') or p.startswith('.') for p in parts):
        raise ToolFailure('permission_denied')
    lower = [p.casefold() for p in parts]
    if any(p in BLOCKED_DIRS for p in lower[:-1]):
        raise ToolFailure('permission_denied')
    if any(word in component for component in lower for word in
           ('secret', 'credential', 'password', 'api.key', 'api_key', 'token', 'id_rsa', 'id_ed25519')):
        raise ToolFailure('permission_denied')
    if Path(lower[-1]).stem in {'key', 'keys', 'authorization', 'authentication'}:
        raise ToolFailure('permission_denied')
    if Path(parts[-1]).suffix.casefold() not in SUFFIXES:
        raise ToolFailure('permission_denied')
    return '/'.join(parts)


def read_file(root, identity, value, maximum=MAX_FILE_BYTES):
    name = relative_path(value)
    descriptors = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(fd)
        info = os.fstat(fd)
        if [info.st_dev, info.st_ino] != list(identity):
            raise ToolFailure('permission_denied')
        parts = name.split('/')
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        descriptors.append(source)
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ToolFailure('permission_denied')
        if before.st_size > maximum:
            raise ToolFailure('resource_limit')
        chunks, size = [], 0
        while True:
            block = os.read(source, min(65536, maximum + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > maximum:
                raise ToolFailure('resource_limit')
        after = os.fstat(source)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ToolFailure('permission_denied')
        text = b''.join(chunks).decode('utf-8')
        if '\x00' in text:
            raise ToolFailure('permission_denied')
        # Keep source line numbers stable when a labeled secret spans lines.
        for pattern in SECRET_PATTERNS:
            text = pattern.sub(lambda match: '[REDACTED]' + ''.join(re.findall(
                r'\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]', match.group())), text)
        return text, size
    except (OSError, UnicodeError, ValueError):
        raise ToolFailure('permission_denied') from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def approved_read(context, name, maximum=MAX_FILE_BYTES):
    name = relative_path(name)
    if name not in context['files']:
        raise ToolFailure('permission_denied')
    return read_file(context['root'], context['root_identity'], name, maximum)


def read(arguments, context):
    text, _ = approved_read(context, arguments['path'])
    lines = text.splitlines()
    start = arguments['start_line'] - 1
    selected = lines[start:start + arguments['line_count']]
    return dict(path=arguments['path'], start_line=start + 1, lines=[s[:1000] for s in selected],
                truncated=start + len(selected) < len(lines) or any(len(s) > 1000 for s in selected))


def search(arguments, context):
    paths = arguments['paths'] or context['files']
    # Check the entire request before reading any files.
    if not set(paths) <= set(context['files']):
        raise ToolFailure('permission_denied')
    for path in paths:
        relative_path(path)
    query = arguments['query'].casefold()
    if not query.strip():
        raise ToolFailure('invalid_input')
    matches, total, count, truncated = [], 0, 0, False
    for path in sorted(set(paths)):
        text, size = approved_read(context, path, min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ToolFailure('resource_limit')
        count += 1
        for line, content in enumerate(text.splitlines(), 1):
            if query in content.casefold():
                if len(matches) == arguments['max_results']:
                    truncated = True
                    break
                matches.append(dict(path=path, line=line, text=content[:1000]))
                truncated |= len(content) > 1000
        if len(matches) == arguments['max_results']:
            truncated |= count < len(set(paths))
            break
    return dict(matches=matches, files_searched=count, truncated=bool(truncated))
