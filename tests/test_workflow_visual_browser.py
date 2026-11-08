"""Visual layout, keyboard access and untrusted handoff rendering in Chromium."""
import json
import os
from pathlib import Path

import pytest
from test_browser import browser_server,pytestmark


@pytest.mark.parametrize('width',[390,768,1440])
def test_connected_nodes_keyboard_inspection_and_responsive_safety(browser_server,width):
    from playwright.sync_api import sync_playwright,expect,TimeoutError as BrowserTimeout
    from ai_agent_pro.database import Repository
    from test_workflows import prepare,execute,provider,result
    from dataclasses import replace

    url,artifacts=browser_server
    repo=Repository(artifacts/'browser.sqlite3')
    store,wid,agents,value=prepare(repo,5)
    hostile='''<script>window.__visualInjected=true</script>
<img src="https://untrusted.invalid/pixel" onerror="window.__visualInjected=true">
![Tracking](https://untrusted.invalid/markdown-image)
[Unsafe](javascript:window.__visualInjected=true)
'''
    outputs=[]
    for i in range(5):
        payload=json.loads(result(hostile if i==1 else 'Deliverable '+str(i)).text)
        payload['risks']=['Confirm the boundary cases.']
        outputs.append(replace(result(),text=json.dumps(payload)))
    execute(repo,provider(*outputs),wid)
    before=store.get(wid)
    checkpoints=store.steps(wid)
    root=Path(__file__).resolve().parents[1]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(root/'data/browser-cache'))
    with sync_playwright() as p:
        browser=p.chromium.launch()
        context=browser.new_context(viewport={'width':width,'height':1000},reduced_motion='reduce')
        external=[]
        def route(r):
            if r.request.url.startswith(url+'/'):r.continue_()
            else:
                external.append(r.request.url)
                r.abort()
        context.route('**/*',route)
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto(url)
            expect(page.get_by_role('heading',name='Overview',exact=True)).to_be_visible()
            # Native keyboard navigation works even with a collapsed sidebar.
            if not page.get_by_test_id('stSidebar').is_visible():
                page.get_by_role('button',name='Expand sidebar',exact=True).click()
            page.get_by_test_id('stSidebar').get_by_text('Workflows',exact=True).click()
            if width<=768:
                page.get_by_role('button',name='Collapse sidebar',exact=True).click()
            expect(page.get_by_role('textbox',name='Workflow goal',exact=True)).to_be_visible()
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            combo=page.get_by_role('combobox',name='Workflow execution',exact=True)
            label='Build a safe sample feature. · '+wid[:8]
            for attempt in range(3):
                combo.click()
                try:
                    page.get_by_role('listbox').get_by_text(label,exact=True).click(timeout=1500)
                    break
                except BrowserTimeout:
                    if attempt==2:raise
                    combo.press('Escape')
            canvas=page.locator('.st-key-workflow_canvas')
            expect(canvas.get_by_text('Workflow status: completed',exact=True)).to_be_visible()
            expect(canvas.get_by_role('button',name='Inspect step 1',exact=True)).to_be_enabled()
            assert canvas.get_by_role('button',name='Inspect step 1',exact=True).get_attribute('tabindex')!='-1'
            nodes=canvas.locator('[class*="st-key-wf_node_"]')
            expect(nodes).to_have_count(5)
            for i in range(5):
                expect(nodes.nth(i).get_by_text('✓ Completed',exact=True)).to_be_visible()
                expect(nodes.nth(i).get_by_text(agents[i].name,exact=True)).to_be_visible()
            assert nodes.nth(0).evaluate("el=>getComputedStyle(el,'::after').content")!='none'
            assert nodes.nth(0).evaluate("el=>getComputedStyle(el,'::before').content")!='none'
            assert nodes.nth(4).evaluate("el=>getComputedStyle(el,'::after').content")=='none'
            canvas.get_by_role('heading',name='Agent workflow',exact=True).scroll_into_view_if_needed()
            page.screenshot(path=str(artifacts/f'workflow-canvas-{width}.png'),full_page=True)
            page.locator('.st-key-workflow_nodes').scroll_into_view_if_needed()
            page.screenshot(path=str(artifacts/f'workflow-nodes-{width}.png'),full_page=True)
            node=canvas.get_by_role('button',name='Inspect step 2',exact=True)
            node.focus()
            assert node.evaluate("el=>el===document.activeElement")
            node.press('Enter')
            expect(canvas.get_by_role('heading',name='Step 2 inspection',exact=True)).to_be_visible()
            expect(canvas.get_by_text('1. Confirm the boundary cases.',exact=True)).to_be_visible()
            # Tab order stays inside ordinary browser controls, with no custom
            # canvas keyboard trap or pointer-only interaction requirement.
            third=canvas.get_by_role('button',name='Inspect step 3',exact=True)
            node=canvas.get_by_role('button',name='Inspect step 2',exact=True)
            node.focus();page.keyboard.press('Tab')
            assert third.evaluate("el=>el===document.activeElement")
            page.keyboard.press('Enter')
            expect(canvas.get_by_role('heading',name='Step 3 inspection',exact=True)).to_be_visible()
            canvas.get_by_role('button',name='Inspect step 2',exact=True).click()
            canvas.get_by_text('Deliverable and structured handoff',exact=True).click()
            inspector=page.locator('.st-key-workflow_inspector')
            expect(inspector).to_contain_text('Tracking')
            assert inspector.locator('script,img,iframe,object,embed').count()==0
            assert inspector.locator('[onclick],[onerror],[onload]').count()==0
            assert page.evaluate('window.__visualInjected===undefined')
            assert not any('untrusted.invalid' in u for u in external)
            canvas.get_by_text('Execution details',exact=True).click()
            expect(inspector.get_by_text('Recorded execution status: completed',exact=True)).to_be_visible()
            expect(canvas.get_by_text('Model requests',exact=True)).to_be_visible()
            expect(canvas.get_by_text('Charged / reserved tokens',exact=True)).to_be_visible()
            expect(canvas.get_by_text('Active time',exact=True)).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
            assert not page.get_by_test_id('stException').count()
            assert errors==[]
            assert store.get(wid)==before and store.steps(wid)==checkpoints
            page.screenshot(path=str(artifacts/f'workflow-visual-{width}.png'),full_page=True)
            # Verify browser zoom/text enlargement does not clip the canvas.
            page.evaluate("document.documentElement.style.fontSize='20px'")
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
        finally:
            context.close();browser.close()
