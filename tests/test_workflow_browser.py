"""Chromium sequential workflow controls, progress, cancellation and approvals."""
import os
from pathlib import Path
import pytest
from test_browser import browser_server,pytestmark
from support.browser_sync import StreamlitSession, workflow_state


@pytest.mark.parametrize('browser_server',['workflows'],indirect=True)
@pytest.mark.parametrize('count,outcome,hold_reviewer',
    [(count,outcome,False) for count in (5,6) for outcome in ('complete','invalid','cancel','approve','deny')]
    + [(count,'complete',True) for count in (5,6)])
def test_browser_workflow_progress_and_safe_decisions(browser_server,outcome,count,hold_reviewer):
    from playwright.sync_api import sync_playwright,expect
    from ai_agent_pro.database import Repository
    from ai_agent_pro.practice import PracticeWorkspace
    from ai_agent_pro.workflow_contracts import software_workflow, Limits
    from ai_agent_pro.workflow_store import WorkflowStore
    from ai_agent_pro.config import Agent
    url,artifacts=browser_server
    repo=Repository(artifacts/'browser.sqlite3')
    agents=[Agent.create('Specialist '+str(i),'','Helpful specialist') for i in range(count)]
    for a in agents:repo.save_agent(a)
    store=WorkflowStore(repo)
    store.save(software_workflow([a.id for a in agents],
        Limits(max_requests=8,token_budget=30000,max_tool_calls=1) if count==6 and outcome in ('approve','deny') else None))
    practice=PracticeWorkspace(artifacts);practice.create_example()
    if hold_reviewer:
        (artifacts/'hold-reviewer').touch()
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(Path(__file__).resolve().parents[1]/'data/browser-cache'))
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1100})
        session=StreamlitSession(page)
        page.route('**/*',lambda r:r.continue_() if r.request.url.startswith(url+'/') else r.abort())
        try:
            page.goto(url)
            expect(page.get_by_test_id('stSidebar').get_by_text('Workflows',exact=True)).to_be_visible()
            session.wait_initial()
            session.change(lambda: page.get_by_test_id('stSidebar').get_by_text('Workflows',exact=True).click())
            goal=page.get_by_role('textbox',name='Workflow goal',exact=True)
            expect(goal).to_be_enabled()
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            if outcome in ('approve','deny'):
                page.get_by_text('Step 1: Requirements Analyst',exact=True).click()
                session.change(lambda: page.get_by_text('Allow practice proposals for step 1',exact=True).click())
                expect(page.get_by_role('checkbox',name='Allow practice proposals for step 1',exact=True)).to_be_checked()
                chooser=page.get_by_role('combobox',name='Practice files for step 1',exact=True)
                chooser.click()
                chooser.press_sequentially('example',delay=20)
                expect(page.get_by_role('listbox').get_by_text('example.txt',exact=True)).to_be_visible()
                def select_file():
                    page.get_by_role('listbox').get_by_text('example.txt',exact=True).click()
                    chooser.press('Escape')
                    chooser.press('Tab')
                session.change(select_file,expected_value=['example.txt'])
            def enter_goal():
                goal.fill('Please '+outcome+' a harmless software workflow')
                goal.press('Tab')
            session.change(enter_goal,expected_value='Please '+outcome+' a harmless software workflow')
            page.get_by_role('button',name='Start workflow',exact=True).click()
            if hold_reviewer:
                import time
                deadline=time.monotonic()+30
                while not (artifacts/'reviewer-waiting').exists() and time.monotonic()<deadline:
                    page.wait_for_timeout(50)
                assert (artifacts/'reviewer-waiting').exists(), 'Reviewer never reached the mocked completion gate.'
                expect(page.get_by_text('Workflow status: running',exact=True)).to_be_visible()
                row=store.runs()[0]
                steps=store.steps(row['id'])
                assert row['cursor']==count-1 and row['requests']==count
                assert steps[-1]['status']=='running' and steps[-1]['output'] is None
                assert all(s['status']=='completed' for s in steps[:-1])
                # Text deltas are not completion. Release the exact existing
                # response only after verifying no premature final checkpoint.
                (artifacts/'release-reviewer').touch()
            if outcome=='cancel':
                expect(page.get_by_text('Workflow status: running',exact=True)).to_be_visible()
                expect(page.locator('[class*="st-key-wf_node_0_status_running"]').get_by_text('▶ Running',exact=True)).to_be_visible()
                page.get_by_role('button',name='Cancel workflow',exact=True).click()
                workflow_state(page,'cancelled')
                expect(page.locator('[class*="st-key-wf_node_0_status_cancelled"]').get_by_text('■ Cancelled',exact=True)).to_be_visible()
                assert store.runs()[0]['requests']==1
            elif outcome=='invalid':
                workflow_state(page,'failed')
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
                    workflow_state(page,'failed')
                    assert (practice.path/'example.txt').read_text()=='This is a practice file.\n'
                    assert store.runs()[0]['requests']==1
                else:
                    approve.click()
                    workflow_state(page,'ready')
                    assert (practice.path/'example.txt').read_text()=='Reviewed workflow practice text.\n'
                    row=store.runs()[0]
                    assert row['cursor']==1 and row['requests']==2
                    page.get_by_role('button',name='Continue workflow',exact=True).click()
                    workflow_state(page,'completed')
                    assert store.runs()[0]['requests']==count+1
            else:
                workflow_state(page,'completed')
                expect(page.locator(f'[class*="st-key-wf_node_{count-1}_status_completed"]').get_by_text('Reviewer',exact=True)).to_be_visible()
                assert store.runs()[0]['requests']==count
            if outcome=='complete' or outcome=='approve':
                row=store.runs()[0]
                assert row['status']=='completed' and row['cursor']==count
                steps=store.steps(row['id'])
                assert len(steps)==count and all(s['status']=='completed' and s['output'] for s in steps)
                assert len({s['run_id'] for s in steps})==count
            assert not page.get_by_test_id('stException').count()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        finally:
            browser.close()
