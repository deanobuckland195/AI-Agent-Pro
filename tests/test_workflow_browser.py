"""Chromium sequential workflow controls, progress, cancellation and approvals."""
import os
from pathlib import Path
import pytest
from test_browser import browser_server,pytestmark


@pytest.mark.parametrize('browser_server',['workflows'],indirect=True)
@pytest.mark.parametrize('outcome',['complete','invalid','cancel','approve','deny'])
def test_browser_workflow_progress_and_safe_decisions(browser_server,outcome):
    from playwright.sync_api import sync_playwright,expect,TimeoutError as BrowserTimeout
    from ai_agent_pro.database import Repository
    from ai_agent_pro.practice import PracticeWorkspace
    from ai_agent_pro.workflow_contracts import software_workflow
    from ai_agent_pro.workflow_store import WorkflowStore
    from ai_agent_pro.config import Agent
    url,artifacts=browser_server
    repo=Repository(artifacts/'browser.sqlite3')
    agents=[Agent.create('Specialist '+str(i),'','Helpful specialist') for i in range(5)]
    for a in agents:repo.save_agent(a)
    store=WorkflowStore(repo)
    store.save(software_workflow([a.id for a in agents]))
    practice=PracticeWorkspace(artifacts);practice.create_example()
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(Path(__file__).resolve().parents[1]/'data/browser-cache'))
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.route('**/*',lambda r:r.continue_() if r.request.url.startswith(url+'/') else r.abort())
        try:
            page.goto(url)
            page.get_by_test_id('stSidebar').get_by_text('Workflows',exact=True).click()
            goal=page.get_by_role('textbox',name='Workflow goal',exact=True)
            expect(goal).to_be_enabled()
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            if outcome in ('approve','deny'):
                page.get_by_text('Step 1: Requirements Analyst',exact=True).click()
                page.get_by_text('Allow practice proposals for step 1',exact=True).click()
                expect(page.get_by_role('checkbox',name='Allow practice proposals for step 1',exact=True)).to_be_checked()
                chooser=page.get_by_role('combobox',name='Practice files for step 1',exact=True)
                option=page.get_by_role('listbox').get_by_text('example.txt',exact=True)
                for attempt in range(3):
                    chooser.fill('example');chooser.press('ArrowDown')
                    try:
                        expect(option).to_be_visible(timeout=1500)
                        option.click(timeout=1500)
                        break
                    except (AssertionError,BrowserTimeout):
                        if attempt==2:raise
                        chooser.press('Escape')
                chooser.press('Escape')
            goal.fill('Please '+outcome+' a harmless software workflow')
            goal.press('Tab')
            page.get_by_role('button',name='Start workflow',exact=True).click()
            if outcome=='cancel':
                expect(page.get_by_text('Workflow status: running',exact=True)).to_be_visible()
                expect(page.locator('[class*="st-key-wf_node_0_status_running"]').get_by_text('▶ Running',exact=True)).to_be_visible()
                page.get_by_role('button',name='Cancel workflow',exact=True).click()
                expect(page.get_by_text('Workflow status: cancelled',exact=True)).to_be_visible()
                expect(page.locator('[class*="st-key-wf_node_0_status_cancelled"]').get_by_text('■ Cancelled',exact=True)).to_be_visible()
                assert store.runs()[0]['requests']==1
            elif outcome=='invalid':
                expect(page.get_by_text('Workflow status: failed',exact=True)).to_be_visible()
                expect(page.locator('[class*="st-key-wf_node_0_status_failed"]').get_by_text('! Failed',exact=True)).to_be_visible()
                expect(page.get_by_text('Workflow step output is not a valid versioned JSON handoff.',exact=False)).to_be_visible()
                assert store.runs()[0]['requests']==1
            elif outcome in ('approve','deny'):
                approve=page.get_by_role('button',name='Approve file change',exact=True)
                expect(approve).to_be_visible()
                expect(page.locator('[class*="st-key-wf_node_0_status_waiting_approval"]').get_by_text('? Awaiting approval',exact=True)).to_be_visible()
                assert store.runs()[0]['requests']==1
                assert (practice.path/'example.txt').read_text()=='This is a practice file.\n'
                expect(page.get_by_text('Exact diff',exact=True)).to_be_visible()
                if outcome=='deny':
                    page.get_by_role('button',name='Deny file change',exact=True).click()
                    expect(page.get_by_text('Workflow status: failed',exact=True)).to_be_visible()
                    assert (practice.path/'example.txt').read_text()=='This is a practice file.\n'
                    assert store.runs()[0]['requests']==1
                else:
                    approve.click()
                    expect(page.get_by_text('Workflow status: ready',exact=True)).to_be_visible()
                    assert (practice.path/'example.txt').read_text()=='Reviewed workflow practice text.\n'
                    row=store.runs()[0]
                    assert row['cursor']==1 and row['requests']==2
                    page.get_by_role('button',name='Continue workflow',exact=True).click()
                    expect(page.get_by_text('Workflow status: completed',exact=True)).to_be_visible()
                    assert store.runs()[0]['requests']==6
            else:
                expect(page.get_by_text('Workflow status: completed',exact=True)).to_be_visible()
                expect(page.locator('[class*="st-key-wf_node_4_status_completed"]').get_by_text('Reviewer',exact=True)).to_be_visible()
                assert store.runs()[0]['requests']==5
            assert not page.get_by_test_id('stException').count()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        finally:
            browser.close()
