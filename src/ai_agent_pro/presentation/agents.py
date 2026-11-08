"""Agent directory and configuration forms."""
from dataclasses import replace
import streamlit as st
from ai_agent_pro.config import Agent, Settings, MODELS, MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS
from ai_agent_pro.presentation.components import navigate, page_header
from ai_agent_pro.analytics import Analytics, Filters
from ai_agent_pro.rag.repository import KnowledgeRepository

def agent_form(repo, agent=None):
    settings = agent.settings if agent else Settings()
    with st.form(f"agent_form_{agent.id if agent else 'new'}" ):
        st.subheader('Identity')
        name = st.text_input('Agent name', value=agent.name if agent else '', max_chars=80)
        description = st.text_area('Description', value=agent.description if agent else '', max_chars=500)
        st.subheader('Behaviour')
        instructions = st.text_area('Agent instructions', value=agent.instructions if agent else
                                    'Help the user solve problems clearly and accurately. Explain uncertainty.',
                                    max_chars=8000, height=160)
        st.subheader('Execution limits')
        left, right = st.columns(2)
        with left:
            model = st.selectbox('Model', MODELS, index=MODELS.index(settings.model))
            steps = st.number_input('Maximum requests / stages', 1, 3, settings.max_steps)
            output = st.number_input('Output tokens per request', MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS,
                                     settings.max_output_tokens,
                                     help='A per-request output cap, separate from the total task budget. '
                                          'For detailed multi-step answers with code, try 4,096–8,192. '
                                          'Higher caps can increase latency and cost; the task budget and deadline still apply.')
        with right:
            budget = st.number_input('Total task token budget', 1024, 100000, settings.token_budget)
            timeout = st.number_input('Task deadline (seconds)', 5, 180, settings.timeout_seconds)
            history = st.number_input('Recent messages sent as context', 0, 30, settings.history_messages)
        st.caption('One stage answers directly; two stages draft then review; three add another review. '
                   'Each stage is a billed API request. The budget includes input and output tokens; '
                   'a conservative byte-based reservation can reject a request before it is sent. '
                   'Document mode can add a separately tracked query embedding request.')
        st.caption('Output limit: 64–16,384 tokens per request; existing agents keep their saved cap. '
                   'The 1,200-token default is economical but may truncate detailed tutorials. '
                   'Try 4,096–8,192 for longer answers, or split the request into smaller parts.')
        st.subheader('Documents')
        rag_enabled = st.toggle('Document-grounded answers', value=settings.rag_enabled,
                                help='Uses only assigned ready documents. Each question retrieves fresh sources; prior chat context is omitted. Assign collections in Knowledge base.')
        saved = st.form_submit_button('Save agent' if agent else 'Create agent', type='primary')
    if saved:
        config = Settings(model, int(steps), int(output), int(budget), int(timeout), int(history), rag_enabled)
        updated = replace(agent, name=name, description=description, instructions=instructions,
                          settings=config) if agent else Agent.create(name, description, instructions, config)
        repo.save_agent(updated)
        st.session_state['notice'] = 'Agent saved.'
        st.session_state['_agent_request'] = updated.id
        st.rerun()



def agents_view(repo, agent):
    page_header('Agents', 'Specialist assistants with explicit settings, document scope and recorded execution.')
    agents = repo.agents()
    search = st.text_input('Search agents', placeholder='Search names or descriptions', key='agent_search')
    matches = [a for a in agents if search.casefold() in f'{a.name} {a.description}'.casefold()]
    if not agents:
        st.info('Create your first agent below. You can add documents later.')
    elif not matches:
        st.info('No agents match your search.')
    else:
        from ai_agent_pro.presentation.components import paginate
        columns = st.columns(2)
        for index, item in enumerate(paginate(matches, 'agents_page', 6)):
            with columns[index % 2], st.container(border=True):
                st.subheader(item.name)
                st.text(item.description or 'No description')
                st.caption(f'{item.settings.model} · {item.settings.max_steps} answer stages · '
                           f'{"Document mode" if item.settings.rag_enabled else "Ordinary chat"}')
                st.caption(f'{len(KnowledgeRepository(repo).assigned(item.id))} assigned collections')
                st.caption(f'{Analytics(repo).count(Filters(item.id))} retained task records')
                st.button('Open chat', key=f'chat_{item.id}', icon=':material/chat:',
                          on_click=navigate, args=('Chat', item.id), type='primary')
                st.button('Edit agent', key=f'edit_{item.id}', icon=':material/edit:',
                          on_click=navigate, args=('Agents', item.id, 'Edit selected agent'))
                st.button('Manage collections', key=f'collections_{item.id}', icon=':material/folder_open:',
                          on_click=navigate, args=('Knowledge base', item.id))
    st.divider()
    mode = st.radio('Agent management', ['Create agent', 'Edit selected agent'], key='agents_mode', horizontal=True)
    if mode == 'Create agent':
        agent_form(repo)
    elif agent:
        st.subheader(f'Configure {agent.name}')
        if agent.settings.rag_enabled and not KnowledgeRepository(repo).assigned(agent.id):
            st.warning('Document mode is enabled, but no collections are assigned. Use Manage collections above.')
        elif agent.settings.rag_enabled and not Analytics(repo).knowledge(agent.id)['ready']:
            st.warning('No compatible indexed documents are assigned to this agent. Upload or re-index documents in Knowledge base.')
        if agent.settings.token_budget <= agent.settings.max_output_tokens:
            st.warning('The task budget cannot cover the output cap plus instructions. Increase the budget or reduce the output cap.')
        agent_form(repo, agent)
        with st.expander('Delete selected agent'):
            confirmed = st.checkbox('Permanently delete this agent, conversations and task history')
            st.caption('Shared document collections are retained.')
            if st.button('Delete agent', disabled=not confirmed, icon=':material/delete:'):
                repo.delete_agent(agent.id)
                st.session_state['notice'] = 'Agent deleted. Shared document collections were retained.'
                st.rerun()
    else:
        st.info('Create your first agent to edit its settings.')
