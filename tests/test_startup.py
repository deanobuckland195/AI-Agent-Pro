"""Real headless Streamlit process and HTTP health endpoint smoke test."""
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def test_streamlit_server_starts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, AI_AGENT_PRO_DB=str(tmp_path / 'startup.sqlite3'))
    with (tmp_path / 'streamlit.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.headless=true', f'--server.port={port}', '--server.address=127.0.0.1',
            '--server.fileWatcherType=none', '--browser.gatherUsageStats=false'],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                try:
                    with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
                        assert response.status == 200
                        assert response.read() == b'ok'
                        return
                except (URLError, TimeoutError):
                    time.sleep(.1)
            raise AssertionError('Streamlit did not become healthy within 20 seconds')
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
