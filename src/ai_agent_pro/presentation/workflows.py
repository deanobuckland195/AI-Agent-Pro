"""Visual workflow studio over the existing sequential coordinator."""
import json
import streamlit as st

from ai_agent_pro.config import AppError
from ai_agent_pro.execution import ChatJob
from ai_agent_pro.provider import OpenAIProvider
from ai_agent_pro.rag.embeddings import OpenAIEmbeddings
from ai_agent_pro.rag.service import KnowledgeBase
from ai_agent_pro.presentation.components import page_header
from ai_agent_pro.presentation.workflow_visual import visual_workflow
from ai_agent_pro.presentation.workspace import practice_workspace
from ai_agent_pro.presentation.approvals import approval_panel
from ai_agent_pro.tools.builtins import registry
from ai_agent_pro.tools.contracts import ToolFailure
from ai_agent_pro.tools.executor import ToolExecutor
from ai_agent_pro.tools.permissions import ToolPermissions
from ai_agent_pro.tools.filesystem import PROJECT_ROOT
from ai_agent_pro.tool_calling import ToolAccess
from ai_agent_pro.workflow_contracts import ROLES, Limits, software_workflow
from ai_agent_pro.workflow_engine import WorkflowEngine, StepAccess, BudgetedEmbeddings
from ai_agent_pro.workflow_store import WorkflowStore


def authority(repo, knowledge=False):
    path = repo.path.absolute()
    inside = path.is_relative_to(PROJECT_ROOT)
    if knowledge and not inside:
        raise AppError('Workflow knowledge tools require the bound project database.')
    return ToolExecutor(ToolPermissions(PROJECT_ROOT,path if inside else None,bind_knowledge=knowledge))


def build_access(repo,specs,choices,start=0):
    result = {}
    try:
        for i,spec in enumerate(specs):
            if i < start:
                continue
            tools,files,practice_files,enabled = choices[i]
            if enabled and not practice_files:
                raise AppError('Select existing practice files separately for each proposing step.')
            executor = authority(repo,'knowledge_search' in tools) if tools or files else None
            try:
                if executor:
                    executor.permissions.approve_files(spec['agent_id'],files)
                    for name in tools:
                        executor.registry.get(name,'1.0.0')
                        executor.permissions.allow(spec['agent_id'],name,'1.0.0')
                access = ToolAccess(executor,tuple((n,'1.0.0') for n in tools),practice_workspace() if enabled else None,
                                    tuple(practice_files) if enabled else ()) if tools or enabled else None
                if enabled and executor is None:
                    executor = authority(repo)
                    access = ToolAccess(executor,(),practice_workspace(),tuple(practice_files))
                for name in access.practice_files if access else ():
                    access.practice.read(name)
                result[i] = StepAccess(spec['agent_id'],access,executor,tuple(files),True).validate(spec['agent_id'])
            except BaseException:
                if executor:
                    executor.permissions.close()
                raise
        return result
    except (AppError,ToolFailure,ValueError,OSError):
        for p in result.values():
            if p.context_executor:
                p.context_executor.permissions.close()
        raise AppError('Workflow step permissions or files could not be approved. Select only eligible resources for each step.') from None


def workflow_job(repo,key,wid,accesses=None,approval_id=None):
    store = WorkflowStore(repo)
    if approval_id:
        step = store.steps(wid)[store.get(wid)['cursor']]
        files = tuple(json.loads(step['provenance']).get('files',{}))
        # Only this same child's already-authorised context is rechecked. No
        # read-tool grants are reconstructed and no later agent is started.
        executor = authority(repo) if files else None
        accesses = {step['ordinal']:StepAccess(step['agent_id'],None,executor,files,True)}
    async def operation(engine):
        def retrieval(agent_id,coordinator):
            return KnowledgeBase(repo,BudgetedEmbeddings(coordinator,OpenAIEmbeddings(key))).retrieval
        coordinator = WorkflowEngine(repo,engine.provider,retrieval_factory=retrieval,workspace=practice_workspace(),
            on_text=engine.on_text,on_tool=engine.on_tool,observer=engine.observer,cancel_requested=engine.cancel_requested)
        return await coordinator.run(wid,accesses,approval_id=approval_id)
    return ChatJob(repo,OpenAIProvider(key),'workflow',wid,'',operation=operation).start()


def workflows_view(repo,key):
    page_header('Workflows','Visual agent workflows with explicit permissions, validated handoffs and durable checkpoints.')
    store = WorkflowStore(repo)
    agents = repo.agents()
    if not agents:
        st.info('Create the specialised agents in Agents first. Then assign one to each workflow role.')
        return
    definitions = store.definitions()
    job = st.session_state.get('_workflow_job')
    active = bool(job and job.snapshot().active)
    with st.expander('Configure software-engineering workflow',expanded=not definitions):
        ids = [a.id for a in agents]
        labels = {a.id:a.name for a in agents}
        selected = [st.selectbox(role+' agent',ids,index=None,key='wf_agent_'+str(i),format_func=labels.get,disabled=active) for i,role in enumerate(ROLES)]
        name = st.text_input('Workflow name',value='Software engineering',max_chars=80,disabled=active)
        existing = list(dict.fromkeys(d['id'] for d in definitions))
        target = st.selectbox('Definition to version',['new']+existing,format_func=lambda x:'New workflow' if x=='new' else next(d['name'] for d in definitions if d['id']==x),disabled=active)
        version = st.number_input('Workflow version',1,1000,1,disabled=active)
        requests = st.number_input('Workflow model-request limit',1,40,20,disabled=active)
        tokens = st.number_input('Workflow token budget',4096,200000,100000,disabled=active)
        seconds = st.number_input('Workflow active-time limit (seconds)',5,900,300,disabled=active)
        tools = st.number_input('Workflow tool-call limit',0,20,10,disabled=active)
        if st.button('Save workflow version',disabled=active or not all(selected)):
            value = software_workflow(selected,Limits(requests,tokens,seconds,tools))
            value['name'] = name
            identifier,v = store.save(value,None if target=='new' else target,version)
            st.session_state['wf_definition'] = (identifier,v)
            st.rerun()
    definitions = store.definitions()
    if not definitions:
        return
    options = [(d['id'],d['version']) for d in definitions]
    chosen = st.selectbox('Saved workflow',options,key='wf_definition',disabled=active,
        format_func=lambda x:next(f"{d['name']} · v{d['version']} · engine 1.0.0" for d in definitions if (d['id'],d['version'])==x))
    value = json.loads(next(d['definition'] for d in definitions if (d['id'],d['version'])==chosen))
    runs = store.runs()
    requested = st.session_state.pop('_wf_execution_request',None)
    if requested:
        st.session_state['wf_execution'] = requested
    selected_run = st.selectbox('Workflow execution',['new']+[r['id'] for r in runs],key='wf_execution',disabled=active,
        format_func=lambda x:'New execution' if x=='new' else next(r['goal'][:60]+' · '+r['id'][:8] for r in runs if r['id']==x))
    current = store.sync(selected_run) if selected_run != 'new' else None
    if current:
        value = current['data']['definition']
    canvas = st.container()
    if not current:
        with canvas:
            visual_workflow(repo,value,agents,scope=chosen[0]+'_'+str(chosen[1]))
    choices = []
    st.subheader('Step permissions')
    st.caption('Each step starts without tools. Select permissions separately; agent chat approvals are not copied. '
               'The workflow goal is shared with all mapped steps. Derived document/file context must also be authorised for its recipient.')
    for i,spec in enumerate(value['steps']):
        suffix = (current['id'] if current else chosen[0]+'_'+str(chosen[1]))+'_'+str(i)
        with st.expander(f"Step {i+1}: {spec['role']}"):
            st.text('Agent: '+next((a.name for a in agents if a.id==spec['agent_id']),'Missing agent')+' · '+spec['agent_id'])
            st.text(spec['task'])
            tool_names = [d['name'] for d in registry().catalog()]
            tools = st.multiselect('Tools for step '+str(i+1),tool_names,key='wf_tools_'+suffix,disabled=active)
            files = st.text_area('Authorised project context files for step '+str(i+1),key='wf_files_'+suffix,max_chars=8000,disabled=active)
            enabled = st.checkbox('Allow practice proposals for step '+str(i+1),key='wf_practice_'+suffix,disabled=active)
            practice_files = st.multiselect('Practice files for step '+str(i+1),practice_workspace().files(),key='wf_practice_files_'+suffix,disabled=active)
            choices.append((tuple(tools),tuple(f for f in files.splitlines() if f),tuple(practice_files),enabled))
    st.caption('Limits: '+json.dumps(value['limits'])+'; at most 8 sequential steps. Connections show the saved order; parallel execution is unavailable.')
    if current:
        st.text('Goal: '+current['goal'])
    goal = st.text_area('Workflow goal',max_chars=2000,disabled=active or current is not None)
    if not key:
        st.info('Configure the existing OpenAI API credential before starting a workflow.')
    if current is None and st.button('Start workflow',disabled=active or not key,type='primary'):
        if not goal.strip():
            raise AppError('Enter a workflow goal before starting.')
        accesses = build_access(repo,value['steps'],choices)
        wid = store.start(*chosen,goal)
        st.session_state['_workflow_job'] = workflow_job(repo,key,wid,accesses)
        st.session_state['_wf_execution_request'] = wid
        st.rerun()
    if not current:
        return
    owned_job = job if job and job.conversation_id == current['id'] else None
    polling = bool((owned_job and owned_job.snapshot().active) or current['status']=='running')
    @st.fragment(run_every=.2 if polling else None)
    def progress():
        row = store.sync(current['id'])
        if polling and row['status'] != 'running' and (not owned_job or not owned_job.snapshot().active):
            st.rerun(scope='app')
        visual_workflow(repo,value,agents,scope=row['id'],row=row,steps=store.steps(row['id']))
        st.subheader('Workflow controls')
        if row['error']:
            st.error(row['error'])
        if job and job.conversation_id == row['id']:
            snapshot = job.snapshot()
            if snapshot.active and snapshot.text:
                st.caption('Provisional current-step output; not yet a validated handoff.')
                st.code(snapshot.text,language='json')
        if row['status'] == 'ready' and st.button('Continue workflow',disabled=active or not key):
            accesses = build_access(repo,value['steps'],choices,row['cursor'])
            st.session_state['_workflow_job'] = workflow_job(repo,key,row['id'],accesses)
            st.rerun(scope='app')
        if row['status'] in ('ready','running','waiting_approval'):
            def cancel_workflow():
                try:
                    store.cancel(row['id'])
                except AppError as exc:
                    st.session_state['_wf_action_error'] = str(exc)
                if job and job.conversation_id == row['id']:
                    job.cancel()
            st.button('Cancel workflow',key='wf_cancel_'+row['id'],on_click=cancel_workflow)
        action_error = st.session_state.pop('_wf_action_error',None)
        if action_error:
            st.error(action_error)
        if row['status'] == 'running':
            st.caption('Recovery never repeats an uncertain API request or a completed action.')
            if st.button('Recover interrupted workflow'):
                store.recover(row['id'])
                st.rerun(scope='app')
        if row['status'] == 'waiting_approval':
            step = store.steps(row['id'])[row['cursor']]
            def approved(action):
                st.session_state['_workflow_job'] = workflow_job(repo,key,row['id'],approval_id=action['id'])
            approval_panel(repo,repo.agent(step['agent_id']),step['conversation_id'],key,practice_workspace(),OpenAIProvider,
                           busy=active,resume_callback=approved)
        st.button('Refresh workflow progress')
    with canvas:
        progress()
