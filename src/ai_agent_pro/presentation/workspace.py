"""Agent chat and source-aware execution workspace."""
from pathlib import Path
import streamlit as st
from ai_agent_pro.config import AppError
from ai_agent_pro.execution import ChatJob
from ai_agent_pro.provider import OpenAIProvider
from ai_agent_pro.rag.embeddings import OpenAIEmbeddings
from ai_agent_pro.rag.service import KnowledgeBase
from ai_agent_pro.rag.ui import source_panels
from ai_agent_pro.presentation.components import navigate, page_header, feedback
from ai_agent_pro.presentation.markdown import safe_markdown


def assistant_response(content):
    """Render untrusted Markdown through Streamlit's safe native renderer.

    Never enable HTML or convert model output to an HTML component. Keep the
    stored message unchanged so history and subsequent model context are intact.
    """
    st.markdown(safe_markdown(content), unsafe_allow_html=False)


@st.cache_resource
def practice_workspace():
    from ai_agent_pro.practice import PracticeWorkspace
    from ai_agent_pro.tools.filesystem import PROJECT_ROOT
    return PracticeWorkspace(PROJECT_ROOT)


@st.cache_resource
def tool_executor(database):
    from ai_agent_pro.tools.executor import ToolExecutor
    from ai_agent_pro.tools.permissions import ToolPermissions
    from ai_agent_pro.tools.filesystem import PROJECT_ROOT
    path = Path(database).absolute()
    # Test/custom databases outside the project never grant knowledge access.
    inside = path.is_relative_to(PROJECT_ROOT)
    return ToolExecutor(ToolPermissions(PROJECT_ROOT, path if inside else None, bind_knowledge=inside))


def execution_tools(repo, agent):
    from ai_agent_pro.tool_calling import ToolAccess
    from ai_agent_pro.tools.contracts import ToolFailure
    selected = st.session_state.get(f'tools_{agent.id}', [])
    practice = None
    practice_files = ()
    if st.session_state.get(f'practice_enabled_{agent.id}', False):
        practice = practice_workspace()
        practice_files = tuple(st.session_state.get(f'practice_files_{agent.id}', []))
        if not practice_files:
            raise AppError('Explicitly select at least one existing practice file before proposing changes.')
        for name in practice_files:
            practice.read(name)
    if not selected and not practice:
        return None
    try:
        executor = tool_executor(str(repo.path))
        files = st.session_state.get(f'tool_files_{agent.id}', '').splitlines()
        if any(name in selected for name in ('project_read', 'project_search')):
            executor.permissions.approve_files(agent.id, files)
        for name in selected:
            definition = executor.registry.get(name, '1.0.0')
            if definition.permission == 'knowledge.read' and executor.permissions.database_identity is None:
                raise ToolFailure('permission_denied')
            executor.permissions.allow(agent.id, name, '1.0.0')
        return ToolAccess(executor, tuple((name, '1.0.0') for name in selected), practice, practice_files)
    except (ToolFailure, ValueError, OSError):
        raise AppError('Selected tools or files cannot be approved. Use eligible project files and the local knowledge database.') from None


def tool_activity(snapshot):
    if not snapshot.tools:
        return
    with st.expander('Actual tool activity', expanded=True):
        for item in snapshot.tools:
            st.text(f"{item['tool']} v{item['version']} · {item['state']}")
            if item['data'] is not None:
                st.json(item['data'])
        st.caption('Sanitized results are session-only. Knowledge results show source counts; excerpts are not cached here. '
                   'Activity retains tool names and statuses only.')


def workspace(repo, agent, key):
    page_header('Chat', 'Persistent conversations with bounded execution and traceable document sources.')
    st.subheader(agent.name)
    st.text(agent.description)
    st.caption(f'{agent.settings.model} · up to {agent.settings.max_steps} answer/review stages · '
               f'{agent.settings.max_output_tokens:,} output tokens/request · '
               f'{agent.settings.timeout_seconds}s deadline · {agent.settings.token_budget:,} token budget')
    if agent.settings.rag_enabled:
        st.info('Document-grounded mode: each question uses fresh sources from assigned collections. '
                'Prior conversation context is omitted to avoid reusing deleted or revoked sources.')
    left, right = st.columns(2)
    left.button('Agent settings', on_click=navigate, args=('Agents', agent.id, 'Edit selected agent'), icon=':material/settings:')
    right.button('View collections', on_click=navigate, args=('Knowledge base', agent.id), icon=':material/folder_open:')
    job = st.session_state.get('_chat_job')
    busy = bool(job and job.snapshot().active)
    if st.button('New conversation', icon=':material/add_comment:', disabled=busy):
        cid = repo.new_conversation(agent.id)
        st.session_state[f'conversation_{agent.id}'] = cid
        st.rerun()
    conversations = repo.conversations(agent.id)
    if not conversations:
        repo.new_conversation(agent.id)
        st.rerun()
    ids = [c['id'] for c in conversations]
    labels = {c['id']: f"{c['title']} · {c['created_at'][:16]}" for c in conversations}
    widget_key = f'conversation_{agent.id}'
    if st.session_state.get(widget_key) not in ids:
        st.session_state[widget_key] = ids[0]
    cid = st.selectbox('Conversation', ids, format_func=labels.get, key=widget_key, disabled=busy)
    if job and busy and (job.agent_id != agent.id or job.conversation_id != cid):
        job.cancel()
    with st.expander('Clear this conversation'):
        confirm = st.checkbox('Delete messages and task records permanently', key=f'clear_{cid}')
        if st.button('Clear conversation', disabled=not confirm or busy):
            repo.clear_conversation(cid)
            if job and job.conversation_id == cid:
                st.session_state.pop('_chat_job', None)
            st.rerun()
    with st.expander('Tools for the next message'):
        from ai_agent_pro.tools.builtins import registry
        options = [t['name'] for t in registry().catalog()]
        st.multiselect('Approve tools for this agent and execution', options, key=f'tools_{agent.id}',
                       format_func=lambda name: name + ' v1.0.0', disabled=busy)
        st.text_area('Approved project files (one relative filename per line)', key=f'tool_files_{agent.id}',
                     max_chars=8000, disabled=busy)
        st.caption('None approved by default. Selected tool results are sent to OpenAI. '
                   'Selecting tools requires at least one actual tool call before an answer can be saved. '
                   'Up to 8 model requests and 5 sequential tool calls per task; existing time and token limits still apply. '
                   'Files must be explicitly listed and pass the private-file restrictions. Grants are not saved across restarts.')
    with st.expander('Safe practice file changes'):
        practice = practice_workspace()
        st.caption('Only existing public .txt and .md files directly in ' + str(practice.path) +
                   ' may be selected. Every proposed replacement requires a separate exact-diff decision. No tools or files are selected by default.')
        st.caption('Example setup creates only ' + str(practice.path / 'example.txt') +
                   ' with the text “This is a practice file.” and a final newline. It never overwrites a file.')
        if st.button('Create example practice file', disabled=busy):
            try:
                practice.create_example()
            except AppError as exc:
                feedback(exc)
            else:
                st.rerun()
        st.checkbox('Allow practice file change proposals', key=f'practice_enabled_{agent.id}', disabled=busy)
        st.multiselect('Explicitly approved practice files', practice.files(), key=f'practice_files_{agent.id}', disabled=busy)
    polling = bool(job and job.snapshot().active)

    @st.fragment(run_every=.2 if polling else None)
    def panel():
        chat_panel(repo, agent, cid, key, polling)

    panel()


def chat_panel(repo, agent, cid, key, polling=False):
    job = st.session_state.get('_chat_job')
    snapshot = job.snapshot() if job else None
    if polling and snapshot and not snapshot.active:
        # Rebuild once from persisted messages and stop the periodic refresh.
        st.rerun(scope='app')
    messages = repo.messages(cid)
    if not messages:
        st.info('Start a conversation. Your message and recent context will be sent to OpenAI.')
    for message in messages:
        with st.chat_message(message['role']):
            if message['role'] == 'assistant':
                assistant_response(message['content'])
                source_panels(repo, message['run_id'], agent.id)
            else:
                st.text(message['content'])
    runs = [r for r in repo.runs(agent.id) if r['conversation_id'] == cid]
    running = any(r['status'] == 'running' for r in runs)
    owns_job = bool(job and job.agent_id == agent.id and job.conversation_id == cid)
    if snapshot and owns_job:
        tool_activity(snapshot)
        if snapshot.active:
            st.info(snapshot.state + '…')
            if st.button('Stop generation', disabled=snapshot.state in ('Saving', 'Stopping'),
                         icon=':material/stop_circle:'):
                job.cancel()
                st.rerun(scope='app')
            st.caption('Stopping closes the active connection. Already incurred usage may still be billed.')
            if snapshot.text:
                with st.chat_message('assistant'):
                    st.caption('Provisional response — not yet validated or saved.')
                    assistant_response(snapshot.text)
        elif snapshot.state == 'Awaiting approval':
            if any(r['status'] == 'waiting_approval' for r in runs):
                st.info('Execution paused — review the exact file change below.')
            elif runs and runs[0]['error']:
                st.info(runs[0]['error'])
        elif snapshot.state == 'Completed':
            st.success('Completed — response saved.')
            if runs:
                st.caption(f"Recorded execution: {runs[0]['elapsed']:.2f}s · {runs[0]['requests']} generation request(s)")
        elif snapshot.state == 'Cancelled':
            st.info('Cancelled — no partial answer was saved.')
        else:
            st.error('Failed — no partial answer was saved.')
            feedback(snapshot.error)
    elif running:
        st.warning('A task is running in another session. See Activity for recovery after an interrupted session.')
    elif runs and runs[0]['status'] in ('failed', 'interrupted'):
        if runs[0]['error'] and runs[0]['error'].startswith('Cancelled'):
            st.info('Cancelled — no partial answer was saved.')
        else:
            st.error(runs[0]['error'])
    from ai_agent_pro.presentation.approvals import approval_panel
    approval_panel(repo, agent, cid, key, practice_workspace(), OpenAIProvider,
                   busy=bool(snapshot and snapshot.active))
    waiting = any(r['status'] == 'waiting_approval' for r in repo.runs(agent.id) if r['conversation_id'] == cid)
    if not key:
        st.info('Set OPENAI_API_KEY in your environment or private .streamlit/secrets.toml to enable chat.')
    busy = bool(snapshot and snapshot.active)
    with repo.connect() as db:
        managed = bool(db.execute('SELECT 1 FROM workflow_steps WHERE conversation_id=?',(cid,)).fetchone())
    if managed:
        st.caption('This is a workflow step conversation. Start or resume work from Workflows.')
    prompt = st.chat_input('Message this agent', max_chars=12000, disabled=not key or running or busy or waiting or managed)
    if prompt and not running and not busy and not waiting and not managed:
        retrieval = KnowledgeBase(repo, OpenAIEmbeddings(key)).retrieval if agent.settings.rag_enabled else None
        try:
            access = execution_tools(repo, agent)
        except AppError as exc:
            feedback(exc)
            return
        job = ChatJob(repo, OpenAIProvider(key), agent.id, cid, prompt, retrieval, tool_access=access)
        st.session_state['_chat_job'] = job.start()
        st.rerun(scope='app')
