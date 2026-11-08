"""Real Chromium approval controls with SDK streams and isolated SQLite/files."""
import os
from pathlib import Path
import pytest
from test_browser import browser_server, pytestmark


@pytest.mark.parametrize('browser_server', ['approvals'], indirect=True)
@pytest.mark.parametrize('decision', ['approve', 'deny', 'cancel', 'changed'])
def test_browser_exact_diff_decisions_survive_reload(browser_server, decision):
    from playwright.sync_api import sync_playwright, expect, TimeoutError as BrowserTimeout
    from ai_agent_pro.config import Agent
    from ai_agent_pro.database import Repository
    from ai_agent_pro.practice import PracticeWorkspace
    url, artifacts = browser_server
    practice = PracticeWorkspace(artifacts)
    practice.create_example()
    repo = Repository(artifacts/'browser.sqlite3')
    agent = Agent.create('Practice browser assistant', '', 'Propose changes only')
    repo.save_agent(agent)
    cid = repo.new_conversation(agent.id)
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(Path(__file__).resolve().parents[1]/'data/browser-cache'))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width':1440, 'height':1000})
        page.route('**/*', lambda r: r.continue_() if r.request.url.startswith(url+'/') else r.abort())
        try:
            page.goto(url)
            page.get_by_test_id('stSidebar').get_by_text('Chat',exact=True).click()
            chat = page.get_by_role('textbox', name='Message this agent')
            expect(chat).to_be_enabled()
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            page.get_by_text('Safe practice file changes',exact=True).click()
            page.get_by_text('Allow practice file change proposals',exact=True).click()
            expect(page.get_by_role('checkbox',name='Allow practice file change proposals')).to_be_checked()
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            chooser = page.get_by_role('combobox',name='Explicitly approved practice files')
            option = page.get_by_role('listbox').get_by_text('example.txt',exact=True)
            for attempt in range(3):
                chooser.fill('example')
                chooser.press('ArrowDown')
                try:
                    expect(option).to_be_visible(timeout=1500)
                    option.click(timeout=1500)
                    break
                except (AssertionError, BrowserTimeout):
                    if attempt == 2:
                        raise
                    chooser.press('Escape')
            chooser.press('Escape')
            expect(page.get_by_test_id('stMultiSelect').get_by_text('example.txt',exact=True)).to_be_visible()
            chat.fill('Propose a replacement for example.txt')
            chat.press('Enter')
            expect(page.get_by_role('button',name='Approve file change',exact=True)).to_be_visible()
            assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
            assert repo.runs(agent.id)[0]['requests'] == 1
            # New browser session state; pending action/preview must come from DB.
            page.reload()
            page.get_by_test_id('stSidebar').get_by_text('Chat',exact=True).click()
            expect(page.get_by_text('Exact file: '+str(practice.path/'example.txt'),exact=True)).to_be_visible()
            expect(page.get_by_text('Exact diff',exact=True)).to_be_visible()
            expect(page.get_by_role('button',name='Approve file change',exact=True)).to_be_visible()
            expect(chat).to_be_disabled()
            if decision == 'changed':
                (practice.path/'example.txt').write_text('User edit.\n')
            labels = {'approve':'Approve file change','deny':'Deny file change','cancel':'Cancel paused execution','changed':'Approve file change'}
            page.get_by_role('button',name=labels[decision],exact=True).click()
            if decision == 'approve':
                expect(page.get_by_text('Completed — response saved.',exact=True)).to_be_visible()
                assert (practice.path/'example.txt').read_text() == 'Approved browser text.\n'
                assert repo.runs(agent.id)[0]['requests'] == 2
            elif decision == 'changed':
                expect(page.get_by_text('File changed since proposal; request a new approval.',exact=False)).to_be_visible()
                assert (practice.path/'example.txt').read_text() == 'User edit.\n'
            else:
                expect(page.get_by_text('File change approval · '+('denied' if decision=='deny' else 'cancelled'),exact=True)).to_be_visible()
                assert (practice.path/'example.txt').read_text() == 'This is a practice file.\n'
            assert not page.get_by_test_id('stException').count()
        finally:
            browser.close()
