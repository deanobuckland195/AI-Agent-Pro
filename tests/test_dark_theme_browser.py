"""Native dark screens, readable forms and six-node layouts; no paid calls."""
from pathlib import Path
import os

import pytest
from test_browser import browser_server, pytestmark


@pytest.mark.parametrize('width',[390,768,1440])
def test_dark_theme_across_screens_and_six_stage_inspection(browser_server,width):
    from playwright.sync_api import sync_playwright,expect
    from ai_agent_pro.database import Repository
    from ai_agent_pro.rag.repository import KnowledgeRepository
    from ai_agent_pro.rag.storage import DocumentStorage
    from test_six_stage_workflows import prepare_six
    from test_analytics import record
    url,artifacts=browser_server
    repo=Repository(artifacts/'browser.sqlite3')
    store,wid,agents,_=prepare_six(repo)
    record(repo,agents[0],elapsed=1.2,steps=[{'step':1,'stage':'answer','input_tokens':20,'output_tokens':10}])
    knowledge=KnowledgeRepository(repo)
    collection=knowledge.create_collection('Theme preview documents')
    DocumentStorage(knowledge).upload(collection,'notes.txt',b'Synthetic UI preview document.')
    before=store.get(wid)
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(Path(__file__).resolve().parents[1]/'data/browser-cache'))
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':width,'height':1100},reduced_motion='reduce')
        page.route('**/*',lambda r:r.continue_() if r.request.url.startswith(url+'/') else r.abort())
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            page.goto(url)
            expect(page.get_by_role('heading',name='Overview',exact=True)).to_be_visible()
            for route in ['Overview','Agents','Chat','Knowledge base','Activity','Workflows']:
                sidebar=page.get_by_test_id('stSidebar')
                box=sidebar.bounding_box()
                if not box or box['x']<0:
                    page.get_by_role('button',name='Expand sidebar',exact=True).click()
                expect(sidebar).to_be_in_viewport()
                sidebar.get_by_text(route,exact=True).click()
                expect(page.get_by_role('heading',name=route,exact=True)).to_be_visible()
                if width<=768:
                    page.get_by_role('button',name='Collapse sidebar',exact=True).click()
                app=page.get_by_test_id('stApp')
                assert app.evaluate('el=>getComputedStyle(el).backgroundColor')=='rgb(9, 15, 32)'
                assert 'radial-gradient' in app.evaluate('el=>getComputedStyle(el).backgroundImage')
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
                assert not page.get_by_test_id('stException').count()
                if route=='Chat':
                    expect(page.get_by_role('textbox',name='Message this agent')).to_be_disabled()
                if route=='Overview':
                    primary=page.get_by_role('button',name='Create an agent',exact=True)
                    assert primary.evaluate('el=>getComputedStyle(el).color')=='rgb(255, 255, 255)'
                    assert 'linear-gradient' in primary.evaluate('el=>getComputedStyle(el).backgroundImage')
                if route=='Activity':
                    grid=page.locator('[data-testid="stDataFrame"]:visible').first
                    expect(grid.locator('canvas').first).to_be_visible()
                    assert grid.evaluate('el=>getComputedStyle(el).overflow')!='hidden'
                    grid.hover()
                    # Native table actions must remain reachable, not clipped
                    # by decorative rounding. Trial clicks do not export data.
                    grid.get_by_role('button',name='Fullscreen',exact=True).click(trial=True)
                if route=='Workflows':
                    canvas=page.locator('.st-key-workflow_canvas')
                    nodes=canvas.locator('[class*="st-key-wf_node_"]')
                    expect(nodes).to_have_count(6)
                    for node in nodes.all():
                        expect(node.get_by_text('○ Pending',exact=True)).to_be_visible()
                    page.get_by_role('button',name='Inspect step 5',exact=True).click()
                    inspector=page.locator('.st-key-workflow_inspector')
                    expect(inspector).to_contain_text('QA Testing Engineer')
                    expect(inspector).to_contain_text('Test execution: NOT RUN')
                    selected=canvas.locator('[class*="st-key-wf_node_4_"]')
                    assert selected.evaluate('el=>getComputedStyle(el).outlineColor')=='rgb(182, 162, 255)'
                    assert nodes.nth(0).evaluate("el=>getComputedStyle(el,'::before').content")!='none'
                    assert nodes.nth(5).evaluate("el=>getComputedStyle(el,'::after').content")=='none'
                    canvas.screenshot(path=str(artifacts/f'dark-workflow-{width}.png'),animations='disabled')
                page.screenshot(path=str(artifacts/f'dark-{route.replace(" ","-").lower()}-{width}.png'),full_page=True)
            assert errors==[]
            assert store.get(wid)==before
        finally:
            browser.close()


@pytest.mark.parametrize('width',[390,768,1440])
def test_form_fields_visible_at_rest_focus_and_disabled(browser_server,width):
    from playwright.sync_api import sync_playwright,expect
    from ai_agent_pro.database import Repository
    from test_six_stage_workflows import prepare_six
    from support.browser_sync import StreamlitSession
    url,artifacts=browser_server
    repo=Repository(artifacts/'browser.sqlite3')
    store,wid,_,_=prepare_six(repo)
    agents_before=repo.agents()
    run_before=store.get(wid)
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(Path(__file__).resolve().parents[1]/'data/browser-cache'))

    def normal(field,root):
        expect(field).not_to_be_focused()
        expect(root).to_have_css('background-color','rgb(24, 42, 67)')
        expect(root).to_have_css('border-top-color','rgb(99, 127, 159)')
        expect(root).to_have_css('border-top-width','1px')
        expect(root).to_have_css('border-top-style','solid')
        expect(field).to_have_css('color','rgb(237, 243, 255)')

    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':width,'height':1100},reduced_motion='reduce')
        session=StreamlitSession(page)
        page.route('**/*',lambda r:r.continue_() if r.request.url.startswith(url+'/') else r.abort())
        try:
            page.goto(url)
            expect(page.get_by_role('heading',name='Overview',exact=True)).to_be_visible()
            session.wait_initial()
            for route in ['Agents','Overview','Knowledge base','Activity','Workflows','Chat']:
                sidebar=page.get_by_test_id('stSidebar')
                box=sidebar.bounding_box()
                if not box or box['x']<0:
                    page.get_by_role('button',name='Expand sidebar',exact=True).click()
                session.change(lambda: sidebar.get_by_text(route,exact=True).click())
                if width<=768:
                    page.get_by_role('button',name='Collapse sidebar',exact=True).click()
                if route=='Agents':
                    for label,root_id in [('Agent name','stTextInputRootElement'),
                                          ('Description','stTextAreaRootElement'),
                                          ('Agent instructions','stTextAreaRootElement')]:
                        field=page.get_by_role('textbox',name=label,exact=True)
                        normal(field,field.locator('..'))
                        assert field.locator('..').get_attribute('data-testid')==root_id
                    name=page.get_by_role('textbox',name='Agent name',exact=True)
                    description=page.get_by_role('textbox',name='Description',exact=True)
                    name.focus()
                    name.press('Tab')
                    expect(description).to_be_focused()
                    expect(description.locator('..')).to_have_css('border-top-color','rgb(138, 180, 255)')
                    assert description.evaluate('el=>getComputedStyle(el).outlineStyle')=='solid'
                    description.press('Shift+Tab')
                    expect(name).to_be_focused()
                    expect(name.locator('..')).to_have_css('border-top-color','rgb(138, 180, 255)')
                    name.fill('Unsaved visual preview')
                    expect(name).to_have_css('color','rgb(237, 243, 255)')
                    # Blur without submitting the form, then verify the outline
                    # stays visible after focus leaves the edited field.
                    name.press('Tab')
                    normal(name,name.locator('..'))
                    model=page.get_by_role('combobox',name='Model',exact=True)
                    normal(model,model.locator('..'))
                    number=page.get_by_label('Maximum requests / stages',exact=True)
                    normal(number,number.locator('..'))
                    search=page.get_by_role('textbox',name='Search agents',exact=True)
                    assert search.evaluate("el=>getComputedStyle(el,'::placeholder').color")=='rgb(184, 201, 226)'
                    page.get_by_test_id('stForm').screenshot(path=str(artifacts/f'visible-agent-form-{width}.png'))
                elif route=='Overview':
                    field=page.get_by_role('combobox',name='Activity scope',exact=True)
                    normal(field,field.locator('..'))
                elif route=='Knowledge base':
                    field=page.get_by_role('textbox',name='Collection name',exact=True)
                    normal(field,field.locator('..'))
                elif route=='Activity':
                    field=page.get_by_role('textbox',name='Search task titles',exact=True)
                    normal(field,field.locator('..'))
                elif route=='Workflows':
                    field=page.get_by_role('textbox',name='Workflow goal',exact=True)
                    normal(field,field.locator('..'))
                else:
                    field=page.get_by_role('textbox',name='Message this agent')
                    expect(field).to_be_disabled()
                    root=page.get_by_test_id('stChatInput')
                    expect(root).to_have_css('background-color','rgb(16, 29, 49)')
                    expect(root).to_have_css('border-top-color','rgb(96, 116, 143)')
                    expect(field).to_have_css('color','rgb(180, 196, 220)')
                    expect(field).to_have_css('-webkit-text-fill-color','rgb(180, 196, 220)')
                    assert not field.evaluate('el=>{el.focus(); return document.activeElement===el}')
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
                assert not page.get_by_test_id('stException').count()
            assert repo.agents()==agents_before
            assert store.get(wid)==run_before
        finally:
            browser.close()
