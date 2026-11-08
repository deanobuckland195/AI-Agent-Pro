"""Recorded-data overview and charts with accessible data-table alternatives."""
import altair as alt
import streamlit as st
from ai_agent_pro.analytics import Analytics, Filters, PERIODS
from ai_agent_pro.presentation.components import page_header, navigate, format_bytes

STATUS_COLORS = ['#2457D6', '#087F5B', '#B42318', '#9A6700', '#7551A8']


def filters_view(repo, prefix, include_status=False):
    agents = repo.agents()
    ids, labels = ['all'] + [a.id for a in agents], {a.id: a.name for a in agents}
    key = f'{prefix}_scope'
    if st.session_state.get(key, 'all') not in ids:
        st.session_state[key] = 'all'
    left, right = st.columns(2)
    scope = left.selectbox('Activity scope', ids, key=key,
                           format_func=lambda aid: 'All agents' if aid == 'all' else labels[aid])
    period = right.selectbox('Time range (UTC)', list(PERIODS), key=f'{prefix}_period')
    status, search = 'All', ''
    if include_status:
        left, right = st.columns(2)
        status = left.selectbox('Task status', ['All', 'completed', 'failed', 'running', 'interrupted', 'waiting_approval'], key=f'{prefix}_status')
        search = right.text_input('Search task titles', key=f'{prefix}_search', max_chars=100)
    return Filters(None if scope == 'all' else scope, period, status, search)


def chart_spec(rows, kind):
    """Pure chart builder: data is supplied by repository aggregates, never seeded."""
    if not rows:
        return None
    base = alt.Chart(alt.Data(values=rows))
    if kind == 'activity':
        return base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('day:O', title='Day (UTC)'), y=alt.Y('tasks:Q', title='Recorded tasks'),
            color=alt.Color('status:N', scale=alt.Scale(domain=['running', 'completed', 'failed', 'interrupted', 'waiting_approval'], range=STATUS_COLORS)),
            tooltip=['day:O', 'status:N', 'tasks:Q'])
    if kind == 'tokens':
        return base.transform_fold(['input_tokens', 'output_tokens'], as_=['kind', 'tokens']).mark_bar().encode(
            x=alt.X('day:O', title='Day (UTC)'), y=alt.Y('tokens:Q', title='Known generation tokens'),
            color=alt.Color('kind:N', scale=alt.Scale(range=['#2457D6', '#087F5B'])), tooltip=['day:O', 'kind:N', 'tokens:Q'])
    if kind in ('durations', 'index_durations'):
        return base.mark_line(point=True, color='#2457D6').encode(
            x=alt.X('day:O', title='Day (UTC)'), y=alt.Y('seconds:Q', title='Mean recorded indexing seconds' if kind == 'index_durations' else 'Mean recorded task seconds'),
            tooltip=['day:O', alt.Tooltip('seconds:Q', format='.2f'), 'samples:Q'])
    if kind == 'agents':
        return base.mark_bar(color='#2457D6', cornerRadiusEnd=3).encode(
            x=alt.X('tasks:Q', title='Recorded tasks'), y=alt.Y('agent_id:N', title='Agent ID'), tooltip=['agent:N', 'agent_id:N', 'tasks:Q'])
    if kind == 'errors':
        return base.mark_bar(color='#B42318', cornerRadiusEnd=3).encode(
            x=alt.X('tasks:Q', title='Recorded errors'), y=alt.Y('category:N', title=None), tooltip=['category:N', 'tasks:Q'])
    raise ValueError('Unsupported chart type')


def charts_view(series):
    activity_tab, usage_tab, duration_tab, errors_tab = st.tabs(['Activity', 'Generation usage', 'Execution time', 'Errors'])
    for container, kind, title in [(activity_tab, 'activity', 'Task activity'), (usage_tab, 'tokens', 'Known generation usage'),
                                   (duration_tab, 'durations', 'Recorded execution time'), (errors_tab, 'errors', 'Safe error categories')]:
        with container:
            st.subheader(title)
            chart = chart_spec(series[kind], kind)
            if chart is None:
                st.info('No recorded data for this chart in the selected scope.')
            else:
                st.altair_chart(chart, width='stretch')
                with st.expander(f'{title}: data table'):
                    st.dataframe(series[kind], hide_index=True, width='stretch')
    with activity_tab:
        chart = chart_spec(series['agents'], 'agents')
        if chart is not None:
            st.caption('Recorded tasks by agent. IDs distinguish agents with identical names.')
            st.altair_chart(chart, width='stretch')
            with st.expander('Agent activity: data table'):
                st.dataframe(series['agents'], hide_index=True, width='stretch')


def summary_view(summary):
    with st.container(key='metrics'):
        a, b, c, d = st.columns(4)
        a.metric('Recorded tasks', summary['tasks'])
        rate = summary['completion_rate']
        b.metric('Completion rate', f'{rate:.0%}' if rate is not None else 'Unavailable')
        seconds = summary['median_seconds']
        c.metric('Median task time', f'{seconds:.2f}s' if seconds is not None else 'Unavailable')
        tokens = summary['generation_tokens']
        d.metric('Known generation tokens', f'{tokens:,}' if tokens is not None else 'Unavailable')
    st.caption(f"{summary.get('waiting_approval', 0)} awaiting approval · {summary['running']} running · {summary['failed']} failed · {summary['interrupted']} interrupted · "
               f"{summary['duration_samples']} measured durations · {summary['known_stages']}/{summary['stages']} recorded stages with known usage")
    st.caption('Completion is an execution status, not answer accuracy. Durations cover completed/failed tasks; '
               'interrupted records are excluded. Unknown or failed-call usage is not zero. Reservations are not usage.')


def overview(repo):
    page_header('Overview', 'A clear view of your retained agent activity, document readiness and usage.')
    with st.container(key='intro'):
        st.subheader('Build with context. Keep the evidence.')
        st.write('Configure specialist assistants, connect documents and inspect each recorded execution.')
        a, b = st.columns(2)
        a.button('Create an agent', icon=':material/add:', on_click=navigate,
                 args=('Agents', None, 'Create agent'), type='primary')
        b.button('Open document library', icon=':material/folder_open:', on_click=navigate, args=('Knowledge base',))
    filters = filters_view(repo, 'overview')
    analytics = Analytics(repo)
    summary = analytics.summary(filters)
    summary_view(summary)
    if not summary['tasks']:
        st.info('No recorded tasks in this scope. Start a conversation to create real activity.')
    charts_view(analytics.series(filters))
    knowledge = analytics.knowledge(filters.agent_id, filters.period)
    st.subheader('Knowledge & embedding usage')
    a, b, c = st.columns(3)
    a.metric('Current documents', knowledge['documents'])
    b.metric('Compatible ready indexes', knowledge['ready'])
    c.metric('Original upload size', format_bytes(knowledge['bytes']))
    st.caption('Document inventory is current and scoped by agent; the task date filter does not turn it into a historical snapshot. '
               'Original upload size is not the SQLite file size.')
    st.write(f"Query embedding tokens: {summary['query_tokens'] if summary['query_tokens'] is not None else 'Unavailable'} "
             f"({summary['known_queries']}/{summary['query_records']} known records)")
    st.write(f"Index embedding tokens: {knowledge['index_tokens'] if knowledge['index_tokens'] is not None else 'Unavailable'} "
             f"({knowledge['known_jobs']}/{knowledge['index_jobs']} known jobs)")
    st.caption('Embedding totals are separate from generation totals and follow the selected time range. Failed calls may still be billed. '
               'Deleting documents or clearing conversations removes associated records; these are retained-record totals, not account billing.')
    durations = analytics.index_durations(filters.agent_id, filters.period)
    with st.expander('Recorded indexing performance'):
        chart = chart_spec(durations, 'index_durations')
        if chart is None:
            st.info('No measured indexing durations in this scope. Legacy and interrupted jobs are not assigned a duration.')
        else:
            st.altair_chart(chart, width='stretch')
            st.dataframe(durations, hide_index=True, width='stretch')
    st.subheader('Recent tasks')
    for run in analytics.runs(filters, limit=5):
        with st.container(border=True):
            st.text(f"{run['agent_name']} · {run['status'].upper()}")
            st.text(run['title'])
            st.button('Inspect task', key=f"inspect_{run['id']}", on_click=inspect_task, args=(run['id'],))


def inspect_task(run_id):
    st.session_state['_task_request'] = run_id
    # A shortcut should not inherit filters that hide the selected record.
    st.session_state['activity_scope'] = 'all'
    st.session_state['activity_period'] = 'All time'
    st.session_state['activity_status'] = 'All'
    st.session_state['activity_search'] = ''
    st.session_state['activity_page'] = 1
    navigate('Activity')
