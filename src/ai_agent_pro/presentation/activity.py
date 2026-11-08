"""Filterable, paginated retained activity and evidence-backed task details."""
import json
import math
import streamlit as st
from ai_agent_pro.analytics import Analytics
from ai_agent_pro.rag.repository import KnowledgeRepository
from ai_agent_pro.rag.ui import source_panels
from ai_agent_pro.presentation.components import page_header, navigate, recovery_wait, notice
from ai_agent_pro.presentation.dashboard import filters_view, summary_view, charts_view
from ai_agent_pro.telemetry import events
from ai_agent_pro.presentation.progress import workflow_view


def open_conversation(agent_id, conversation_id):
    st.session_state[f'conversation_{agent_id}'] = conversation_id
    navigate('Chat', agent_id)


def history_view(repo, agent=None):
    page_header('Activity', 'Inspect recorded executions, their source evidence and measured usage. Dates are UTC.')
    st.button('Refresh records', icon=':material/refresh:')
    filters = filters_view(repo, 'activity', include_status=True)
    analytics = Analytics(repo)
    summary_view(analytics.summary(filters))
    with st.expander('Activity charts'):
        charts_view(analytics.series(filters))
    count = analytics.count(filters)
    if not count:
        st.info('No recorded tasks match these filters.')
        return
    requested = st.session_state.pop('_task_request', None)
    if requested:
        position = analytics.position(requested, filters)
        if position is not None:
            st.session_state['activity_page'] = position // 10 + 1
            st.session_state['activity_task'] = requested
    pages = max(1, math.ceil(count / 10))
    if st.session_state.get('activity_page', 1) > pages:
        st.session_state['activity_page'] = 1
    page = st.selectbox('Task page', list(range(1, pages + 1)), key='activity_page',
                        format_func=lambda p: f'{p} of {pages}')
    runs = analytics.runs(filters, offset=(page - 1) * 10)
    st.caption(f'{count} matching tasks · up to 10 per page')
    st.dataframe([{'Task ID': r['id'], 'Agent': r['agent_name'], 'Title': r['title'],
                   'Status': r['status'], 'Started (UTC)': r['started_at'],
                   'Generation requests': r['requests']} for r in runs], hide_index=True, width='stretch')
    ids = [r['id'] for r in runs]
    if st.session_state.get('activity_task') not in ids:
        st.session_state['activity_task'] = ids[0]
    selected = st.selectbox('Inspect task', ids, key='activity_task',
                            format_func=lambda rid: next(f"{r['title']} · {r['status']}" for r in runs if r['id'] == rid))
    task_details(repo, next(r for r in runs if r['id'] == selected))


def task_details(repo, run):
    knowledge = KnowledgeRepository(repo)
    with st.container(border=True):
        st.subheader('Task details')
        st.text(run['title'])
        st.text(f"{run['status'].upper()} · {run['agent_name']} · {run['model']}")
        st.caption(f"Task ID: {run['id']} · Started (UTC): {run['started_at']}")
        a, b, c = st.columns(3)
        a.metric('Generation requests', run['requests'])
        b.metric('Recorded execution time', f"{run['elapsed']:.2f}s" if run['status'] in ('completed','failed') else 'Unavailable')
        c.metric('Reported run tokens', f"{run['input_tokens'] + run['output_tokens']:,}" if run['input_tokens'] is not None and run['output_tokens'] is not None else 'Unavailable')
        st.button('Open conversation', icon=':material/chat:', on_click=open_conversation,
                  args=(run['agent_id'], run['conversation_id']))
        if run['error']:
            from ai_agent_pro.presentation.components import feedback
            feedback(run['error'])
        if run['status'] == 'running':
            wait = recovery_wait(run['active_since'] or run['started_at'], json.loads(run['settings'])['timeout_seconds'])
            st.caption(f'Recovery available in {wait}s. Recovery marks an expired record interrupted; it does not cancel a live API request.')
            if st.button('Recover interrupted task', key=run['id'], disabled=wait > 0):
                repo.recover_run(run['id'])
                notice('Interrupted task recovered. No request was retried.')
                st.rerun()
        st.subheader('Recorded workflow')
        recorded = events(repo, run_id=run['id'])
        workflow_view(recorded)
        stages = json.loads(run['steps'])
        if stages and not recorded:
            for stage in stages:
                st.text(f"✓ Stage {stage['step']}: {stage['stage']} completed")
                st.caption(f"Input tokens: {stage['input_tokens'] if stage['input_tokens'] is not None else 'Unavailable'} · "
                           f"Output tokens: {stage['output_tokens'] if stage['output_tokens'] is not None else 'Unavailable'}")
        elif not stages:
            st.caption('No completed generation stages were recorded. This may include a no-evidence response or a failure before generation.')
        usage = knowledge.usage(run['id'])
        if usage:
            st.write('Query embedding usage (separate from generation totals)')
            st.text(f"{usage['requests']} request(s) · Reported tokens: {usage['input_tokens'] if usage['input_tokens'] is not None else 'Unavailable'}")
            st.caption(f"Reservation: {usage['reserved_tokens']} — not measured usage")
        source_panels(repo, run['id'], run['agent_id'])
        with st.expander('Advanced configuration snapshot'):
            st.json(json.loads(run['settings']))

