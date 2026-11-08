"""Test-process isolation, also loaded by Python-based browser servers.

Never put this directory on a production application's PYTHONPATH.
"""
import ipaddress
import os
from pathlib import Path
import sys


class ExternalNetworkBlocked(RuntimeError):
    pass


def loopback_host(host):
    if isinstance(host, bytes):
        host = host.decode('ascii', errors='replace')
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _network_event(event, arguments):
    if event in ('socket.connect', 'socket.sendto'):
        address = arguments[1]
        # Local Unix sockets are used by asyncio and browser automation.
        if isinstance(address, tuple) and not loopback_host(address[0]):
            raise ExternalNetworkBlocked('Automated tests permit loopback networking only.')
    elif event in ('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr'):
        host = arguments[0]
        if host is not None and not loopback_host(host):
            raise ExternalNetworkBlocked('Automated tests prohibit external DNS lookups.')


def isolate():
    # Delete by name without retrieving or retaining inherited secret values.
    for name in list(os.environ):
        upper = name.upper()
        if (upper.startswith(('OPENAI_', 'AZURE_', 'AWS_', 'ANTHROPIC_', 'GOOGLE_'))
                or any(word in upper for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))):
            del os.environ[name]
    os.environ['OPENAI_API_KEY'] = ''
    os.environ['STREAMLIT_SECRETS_FILES'] = str(Path(__file__).with_name('empty-secrets.toml'))
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['AI_AGENT_PRO_TEST_ISOLATION'] = '1'
    sys.addaudithook(_network_event)


if os.environ.get('AI_AGENT_PRO_TEST_ISOLATION') == '1':
    isolate()
