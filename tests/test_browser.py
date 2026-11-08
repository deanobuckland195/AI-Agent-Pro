"""Opt-in isolated Chromium smoke tests. No credentials or paid requests.

Install .[browser], install Chromium, then set AI_AGENT_PRO_BROWSER_TESTS=1.
All mutations use a disposable test database, never the application database.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('AI_AGENT_PRO_BROWSER_TESTS') != '1',
                                reason='Opt-in browser tests: set AI_AGENT_PRO_BROWSER_TESTS=1')


@pytest.fixture
def browser_server(tmp_path, request):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    env = dict(os.environ, AI_AGENT_PRO_DB=str(tmp_path / 'browser.sqlite3'), OPENAI_API_KEY='', PYTHONDONTWRITEBYTECODE='1')
    script = 'app.py'
    if getattr(request, 'param', None) == 'streaming':
        script = 'tests/browser_stream_app.py'
        env['AI_AGENT_PRO_BROWSER_CONTROL'] = str(tmp_path)
    if getattr(request, 'param', None) == 'tools':
        script = 'tests/browser_tool_app.py'
        env['AI_AGENT_PRO_BROWSER_CONTROL'] = str(tmp_path)
    if getattr(request, 'param', None) == 'approvals':
        script = 'tests/browser_approval_app.py'
        env['AI_AGENT_PRO_BROWSER_CONTROL'] = str(tmp_path)
    if getattr(request, 'param', None) == 'workflows':
        script = 'tests/browser_workflow_app.py'
        env['AI_AGENT_PRO_BROWSER_CONTROL'] = str(tmp_path)
    with (tmp_path / 'browser-server.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', script,
            '--server.headless=true', f'--server.port={port}', '--server.address=127.0.0.1',
            '--server.fileWatcherType=none', '--browser.gatherUsageStats=false'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail('Isolated Streamlit test server exited before becoming healthy.')
                try:
                    with urlopen(url + '/_stcore/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(.1)
            else:
                pytest.fail('Isolated Streamlit test server did not become healthy.')
            yield url, tmp_path
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize('width', [390, 768, 1440])
def test_responsive_keyboard_and_actions(browser_server, width):
    from playwright.sync_api import sync_playwright, expect
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(root / 'data' / 'browser-cache'))
    url, artifact_dir = browser_server
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={'width': width, 'height': 1000})
        context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(url + '/') else route.abort())
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            page.goto(url)
            expect(page.get_by_role('heading', name='Overview', exact=True)).to_be_visible()
            expect(page.get_by_text('No recorded tasks in this scope.', exact=False)).to_be_visible()
            page.get_by_role('button', name='Create an agent', exact=True).click()
            expect(page.get_by_role('heading', name='Agents', exact=True)).to_be_visible()
            page.get_by_role('textbox', name='Agent name', exact=True).fill('Browser test assistant')
            # Submit using the keyboard, rather than a pointer-only interaction.
            page.get_by_role('button', name='Create agent', exact=True).focus()
            page.keyboard.press('Enter')
            expect(page.get_by_role('button', name='Open chat', exact=True)).to_be_visible()
            page.get_by_role('button', name='Open chat', exact=True).click()
            expect(page.get_by_role('heading', name='Chat', exact=True)).to_be_visible()
            expect(page.get_by_role('textbox', name='Message this agent')).to_be_disabled()
            page.get_by_role('button', name='View collections', exact=True).click()
            expect(page.get_by_role('heading', name='Knowledge base', exact=True)).to_be_visible()
            page.get_by_role('textbox', name='Collection name', exact=True).fill('Browser test documents')
            page.get_by_role('button', name='Create collection', exact=True).click()
            expect(page.get_by_role('heading', name='Add a document', exact=True)).to_be_visible()
            page.get_by_test_id('stFileUploader').locator('input[type=file]').set_input_files(
                {'name': 'notes.txt', 'mimeType': 'text/plain', 'buffer': b'Local browser test document.'})
            page.get_by_role('button', name='Upload document', exact=True).click()
            expect(page.get_by_role('button', name='Index document', exact=True)).to_be_disabled()
            expect(page.get_by_role('heading', name='notes.txt', exact=True)).to_be_visible()
            # No page-wide horizontal overflow; table scrolling is intentionally local.
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
            assert not page.get_by_test_id('stException').count()
            assert errors == []
            page.screenshot(path=str(artifact_dir / f'library-{width}.png'), full_page=True)
            # Record an actual offline engine execution in this disposable DB,
            # with usage explicitly unknown. Then verify chart rendering in Chromium.
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            from unittest.mock import AsyncMock
            from ai_agent_pro.database import Repository
            from ai_agent_pro.engine import AgentEngine
            from ai_agent_pro.provider import Completion
            repo = Repository(artifact_dir / 'browser.sqlite3')
            agent = repo.agents()[0]
            cid = repo.new_conversation(agent.id)
            provider = type('OfflineProvider', (), {'complete': AsyncMock(return_value=Completion('Browser test answer', None, None))})()
            # Playwright's synchronous API owns a loop in this thread.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: asyncio.run(AgentEngine(repo, provider).execute(agent.id, cid, 'Browser execution test'))).result()
            page.reload()
            expect(page.get_by_role('heading', name='Overview', exact=True)).to_be_visible()
            expect(page.get_by_test_id('stVegaLiteChart').first).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
            assert not page.get_by_test_id('stException').count()
            assert errors == []
            page.screenshot(path=str(artifact_dir / f'overview-{width}.png'), full_page=True)
        finally:
            context.close()
            browser.close()


@pytest.mark.parametrize('width', [390, 768, 1440])
def test_assistant_markdown_formatting_and_safety(browser_server, width):
    import asyncio
    from unittest.mock import AsyncMock
    from playwright.sync_api import sync_playwright, expect
    from ai_agent_pro.config import Agent
    from ai_agent_pro.database import Repository
    from ai_agent_pro.engine import AgentEngine
    from ai_agent_pro.provider import Completion
    from test_markdown import FORMATTED_REPLY, UNTRUSTED_REPLY

    url, artifact_dir = browser_server
    repo = Repository(artifact_dir / 'browser.sqlite3')
    agent = Agent.create('Markdown assistant', '', 'Be helpful')
    repo.save_agent(agent)
    cid = repo.new_conversation(agent.id)
    reply = FORMATTED_REPLY + '\n' + UNTRUSTED_REPLY
    provider = type('OfflineProvider', (), {'complete': AsyncMock(return_value=Completion(reply, 10, 5))})()
    asyncio.run(AgentEngine(repo, provider).execute(agent.id, cid, 'Show formatting'))

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(root / 'data' / 'browser-cache'))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={'width': width, 'height': 1000})
        external_requests = []

        def route_request(route):
            if route.request.url.startswith(url + '/'):
                route.continue_()
            else:
                external_requests.append(route.request.url)
                route.abort()

        context.route('**/*', route_request)
        page = context.new_page()
        try:
            page.goto(url)
            # Use visible shortcuts, including when the mobile sidebar is collapsed.
            page.get_by_role('button', name='Inspect task', exact=True).click()
            page.get_by_role('button', name='Open conversation', exact=True).click()
            message = page.get_by_test_id('stChatMessage').last
            expect(message.locator('h3')).to_have_text('Formatted answer')
            expect(message.locator('strong')).to_have_text('bold text')
            expect(message.locator('ul > li')).to_have_text(['First bullet', 'Second bullet'])
            expect(message.locator('ol > li')).to_have_text(['First step', 'Second step'])
            expect(message.locator('pre code')).to_contain_text('print("<example>")')
            expect(message).to_contain_text('<script>window.__modelExecuted = true</script>')
            # Native code-block controls contain trusted SVG icons.
            assert message.locator('script, iframe, img, object, embed').count() == 0
            assert message.locator('[onerror], [onload], [onclick]').count() == 0
            hrefs = message.locator('a').evaluate_all('(nodes) => nodes.map(node => node.getAttribute("href") || "")')
            assert all(not href.lower().startswith(('javascript:', 'data:', 'vbscript:')) for href in hrefs), hrefs
            assert page.evaluate('window.__modelExecuted === undefined')
            assert not any('untrusted.invalid' in request for request in external_requests)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
            assert not page.get_by_test_id('stException').count()
        finally:
            context.close()
            browser.close()


@pytest.mark.parametrize('browser_server', ['streaming'], indirect=True)
@pytest.mark.parametrize('outcome', ['complete', 'cancel', 'disconnect', 'timeout', 'output limit'])
def test_live_stream_stop_errors_and_code_clipboard(browser_server, outcome):
    from playwright.sync_api import sync_playwright, expect
    from ai_agent_pro.config import Agent
    from ai_agent_pro.database import Repository
    from browser_stream_app import CODE, SECOND_CODE

    url, artifacts = browser_server
    repo = Repository(artifacts / 'browser.sqlite3')
    agent = Agent.create('Live browser assistant', '', 'Helpful')
    repo.save_agent(agent)
    cid = repo.new_conversation(agent.id)
    # A real retained task supplies a visible navigation shortcut on mobile.
    rid = repo.begin_run(cid, agent, 'Earlier task')
    repo.finish_run(rid, 'completed', 0, 0, 0, 0, [], answer='Earlier answer')
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(root / 'data' / 'browser-cache'))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={'width': 390, 'height': 1000},
                                      permissions=['clipboard-read', 'clipboard-write'])
        context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(url + '/') else route.abort())
        page = context.new_page()
        page.add_init_script('''window.__previews = [];
            new MutationObserver(() => {
                const texts = [...document.querySelectorAll('[data-testid="stChatMessage"]')].map(n => n.textContent).join(' ');
                window.__previews.push(texts);
            }).observe(document, {childList: true, subtree: true, characterData: true});''')
        try:
            page.goto(url)
            page.get_by_role('button', name='Inspect task', exact=True).click()
            page.get_by_role('button', name='Open conversation', exact=True).click()
            chat = page.get_by_role('textbox', name='Message this agent')
            chat.fill('Please ' + outcome)
            chat.press('Enter')
            expect(page.get_by_text('First progressive text', exact=False)).to_be_visible()
            expect(chat).to_be_disabled()
            expect(page.get_by_role('button', name='Stop generation')).to_be_enabled()
            expect(page.get_by_text('Finished answer.', exact=True)).to_have_count(0)
            assert len(repo.runs(agent.id)) == 2
            assert all(m['content'] != 'Finished answer.' for m in repo.messages(cid))
            if outcome == 'cancel':
                page.get_by_role('button', name='Stop generation').click()
                expect(page.get_by_text('Cancelled — no partial answer was saved.', exact=True)).to_be_visible()
            else:
                (artifacts / 'release').touch()
                if outcome == 'complete':
                    expect(page.get_by_text('Completed — response saved.', exact=True)).to_be_visible()
                    block = page.get_by_test_id('stChatMessage').last.locator('pre code').first
                    expect(block).to_have_text(CODE.rstrip('\n'))
                    # Syntax highlighting adds token spans inside the code block.
                    assert block.locator('span').count() > 0
                    block.hover()  # Native toolbar is revealed on hover/focus.
                    copy = page.get_by_test_id('stChatMessage').last.get_by_role('button', name='Copy to clipboard').first
                    expect(copy).to_be_visible()
                    copy.focus()
                    page.keyboard.press('Enter')
                    page.wait_for_function('(code) => navigator.clipboard.readText().then(text => text === code)', arg=CODE)
                    assert page.evaluate('navigator.clipboard.readText()') == CODE
                    second = page.get_by_test_id('stChatMessage').last.locator('pre code').nth(1)
                    second.hover()
                    second_copy = page.get_by_test_id('stChatMessage').last.get_by_role('button', name='Copy to clipboard')
                    second_copy.last.click()
                    page.wait_for_function('(code) => navigator.clipboard.readText().then(text => text === code)', arg=SECOND_CODE)
                    assert page.evaluate('window.__codeRan === undefined')
                    page.reload()
                    page.get_by_role('button', name='Inspect task', exact=True).first.click()
                    page.get_by_role('button', name='Open conversation', exact=True).click()
                    expect(page.get_by_test_id('stChatMessage').last.locator('pre code').first).to_have_text(CODE)
                else:
                    expect(page.get_by_text('Failed — no partial answer was saved.', exact=True)).to_be_visible()
                    if outcome == 'output limit':
                        expect(page.get_by_text('Output limit reached:', exact=False)).to_be_visible()
                        expect(page.get_by_text('Open Agent settings → Output tokens per request.', exact=False)).to_be_visible()
            if outcome != 'complete':
                expect(page.get_by_text('First progressive text', exact=False)).to_have_count(0)
                assert [m['role'] for m in repo.messages(cid)].count('assistant') == 1
                assert repo.runs(agent.id)[0]['status'] == ('interrupted' if outcome == 'cancel' else 'failed')
            expect(page.get_by_role('textbox', name='Message this agent')).to_be_enabled()
            previews = page.evaluate('window.__previews')
            assert not any('sk-private' in text or 'privateabcdefgh' in text for text in previews)
            assert not page.get_by_test_id('stException').count()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        finally:
            context.close()
            browser.close()


@pytest.mark.parametrize('browser_server', ['tools'], indirect=True)
@pytest.mark.parametrize('outcome', ['sequential', 'cancel', 'malformed', 'partial', 'unapproved'])
def test_controlled_tool_activity_and_safe_failures(browser_server, outcome):
    from playwright.sync_api import sync_playwright, expect
    from ai_agent_pro.config import Agent
    from ai_agent_pro.database import Repository
    from ai_agent_pro.telemetry import events
    url, artifacts = browser_server
    repo = Repository(artifacts / 'browser.sqlite3')
    agent = Agent.create('Tool browser assistant', '', 'Helpful')
    repo.save_agent(agent)
    cid = repo.new_conversation(agent.id)
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(root / 'data' / 'browser-cache'))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={'width':1440, 'height':1000})
        page.route('**/*', lambda route: route.continue_() if route.request.url.startswith(url + '/') else route.abort())
        try:
            page.goto(url)
            page.get_by_test_id('stSidebar').get_by_text('Chat', exact=True).click()
            # Wait for the full Chat render. Interacting while the script is
            # still running can open an empty menu that the render then closes.
            expect(page.get_by_role('textbox', name='Message this agent')).to_be_enabled()
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state', 'notRunning')
            page.get_by_text('Tools for the next message', exact=True).click()
            chooser = page.get_by_role('combobox', name='Approve tools for this agent and execution')
            if outcome != 'unapproved':
                option = page.get_by_role('listbox').get_by_text('calculator v1.0.0', exact=True)
                # The expanding panel can interrupt the first dropdown open.
                # Reopen only the UI menu; never retry a submitted execution.
                for attempt in range(3):
                    chooser.fill('calculator')
                    chooser.press('ArrowDown')
                    try:
                        expect(option).to_be_visible(timeout=1500)
                        break
                    except AssertionError:
                        if attempt == 2:
                            raise
                        chooser.press('Escape')
                option.click()
                chooser.press('Escape')
                expect(page.get_by_test_id('stMultiSelect').get_by_text('calculator v1.0.0', exact=True)).to_be_visible()
            chat = page.get_by_role('textbox', name='Message this agent')
            chat.fill('Please calculate ' + outcome)
            chat.press('Enter')
            if outcome in ('sequential', 'cancel'):
                expect(page.get_by_text('Ready to finish.', exact=False)).to_be_visible()
                count = 2 if outcome == 'sequential' else 1
                expect(page.get_by_text('calculator v1.0.0 · completed', exact=True)).to_have_count(count)
                expect(page.get_by_test_id('stJson')).to_have_count(count)
                expect(chat).to_be_disabled()
                assert all(m['role'] != 'assistant' for m in repo.messages(cid))
                if outcome == 'cancel':
                    page.get_by_role('button', name='Stop generation').click()
                    expect(page.get_by_text('Cancelled — no partial answer was saved.', exact=True)).to_be_visible()
                else:
                    (artifacts / 'release').touch()
                    expect(page.get_by_text('Completed — response saved.', exact=True)).to_be_visible()
                    expect(page.get_by_test_id('stChatMessage').last.locator('strong')).to_have_text('4')
                    assert repo.runs(agent.id)[0]['requests'] == 3
                    page.get_by_test_id('stSidebar').get_by_text('Activity', exact=True).click()
                    expect(page.get_by_text('Recorded workflow', exact=True)).to_be_visible()
                    expect(page.get_by_text('✓ Tool execution 1 · completed', exact=True)).to_be_visible()
            else:
                expect(page.get_by_text('Failed — no partial answer was saved.', exact=True)).to_be_visible()
                assert all(m['role'] != 'assistant' for m in repo.messages(cid))
                assert not any(e['stage'] == 'tool' for e in events(repo, run_id=repo.runs(agent.id)[0]['id']))
            assert not page.get_by_test_id('stException').count()
            assert 'sk-private' not in page.locator('body').inner_text()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        finally:
            browser.close()
