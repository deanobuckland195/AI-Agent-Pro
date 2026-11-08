"""Read-only workflow projection and native, accessible node inspection.

No coordinator, grant creation or action decision is reachable from node
selection. Dynamic text uses native text/Markdown rendering, never HTML/CSS.
"""
from time import time

import streamlit as st

from ai_agent_pro.presentation.markdown import safe_markdown
from ai_agent_pro.workflow_contracts import artifact
from ai_agent_pro.config import AppError

STATUSES = {
    'pending': ('○', 'Pending'), 'running': ('▶', 'Running'),
    'completed': ('✓', 'Completed'), 'failed': ('!', 'Failed'),
    'cancelled': ('■', 'Cancelled'), 'waiting_approval': ('?', 'Awaiting approval'),
    'interrupted': ('!', 'Interrupted'), 'ready': ('○', 'Ready to continue'),
}

# Static selectors and palette only. No model, file, agent or database text is
# interpolated into HTML, CSS or selectors. Native widgets retain their roles.
STYLE = '''<style>
.st-key-workflow_canvas {padding:1.1rem; border:1px solid var(--secondary-background-color,#D5DCE8);
 border-radius:16px; background:var(--background-color); margin:.5rem 0 1rem}
.st-key-workflow_nodes [data-testid="stHorizontalBlock"] {gap:1.15rem; align-items:stretch}
.st-key-workflow_nodes [data-testid="stColumn"] {min-width:0}
[class*="st-key-wf_node_"] {position:relative; height:100%; border-radius:12px;
 border-top:4px solid #64748B!important; padding:.85rem!important; overflow-wrap:anywhere; overflow:visible!important}
[class*="st-key-wf_node_"] p {overflow-wrap:anywhere}
[class*="st-key-wf_node_"] button {min-height:44px}
[class*="st-key-wf_node_"]:not([class*="_last"])::before {content:""; position:absolute;
 right:-.75rem; top:50%; width:.75rem; border-top:2px solid #64748B}
[class*="st-key-wf_node_"]:not([class*="_last"])::after {content:""; position:absolute;
 right:-.88rem; top:50%; width:.48rem; height:.48rem; border-top:2px solid #64748B;
 border-right:2px solid #64748B; transform:rotate(45deg)}
[class*="st-key-wf_node_"][class*="_status_running"] {border-top-color:#2563EB!important}
[class*="st-key-wf_node_"][class*="_status_completed"] {border-top-color:#15803D!important}
[class*="st-key-wf_node_"][class*="_status_failed"],
[class*="st-key-wf_node_"][class*="_status_interrupted"] {border-top-color:#B91C1C!important}
[class*="st-key-wf_node_"][class*="_status_waiting_approval"] {border-top-color:#B45309!important}
[class*="st-key-wf_node_"][class*="_chosen"] {outline:2px solid var(--primary-color,#2563EB); outline-offset:2px}
.st-key-workflow_inspector {margin-top:1rem; padding:1rem; border-radius:12px}
.st-key-workflow_metrics [data-testid="stMetricLabel"] p {white-space:normal!important; overflow:visible!important; text-overflow:clip!important}
.st-key-workflow_canvas button:focus-visible {outline:3px solid var(--primary-color,#2563EB)!important;
 outline-offset:3px!important}
@media (max-width:1100px) {
 .st-key-workflow_nodes [data-testid="stHorizontalBlock"] {flex-direction:column!important}
 .st-key-workflow_nodes [data-testid="stColumn"] {width:100%!important; flex:1 1 100%!important}
 [class*="st-key-wf_node_"]:not([class*="_last"])::before {right:auto; left:50%; top:auto;
 bottom:-.75rem; width:0; height:.75rem; border-top:0; border-left:2px solid #64748B}
 [class*="st-key-wf_node_"]:not([class*="_last"])::after {right:auto; left:50%; top:auto;
 bottom:-.85rem; transform:rotate(135deg)}
}
@media (max-width:640px) {
 .st-key-workflow_canvas {padding:.75rem}
 .st-key-workflow_metrics [data-testid="stHorizontalBlock"] {flex-wrap:wrap}
 .st-key-workflow_metrics [data-testid="stColumn"] {min-width:40%!important; flex:1 1 40%!important}
}
@media (prefers-reduced-motion:reduce) {
 .st-key-workflow_canvas * {animation:none!important; transition:none!important; scroll-behavior:auto!important}
}
</style>'''


def status_label(status):
    icon, label = STATUSES.get(status, ('?', 'Unknown'))
    return icon + ' ' + label


def nodes_for(value, agents, row=None, steps=()):
    """Project stored states without upgrading a pending node to success."""
    names = {a.id: a.name for a in agents}
    if row:
        names.update({a['id']:a['name'] for a in row['data']['agents']})
    by_ordinal = {s['ordinal']:s for s in steps}
    result = []
    for ordinal,spec in enumerate(value['steps']):
        step = by_ordinal.get(ordinal,{})
        status = step.get('status','pending')
        if row and ordinal == row['cursor'] and row['status'] == 'interrupted' and status != 'completed':
            status = 'interrupted'
        if status not in STATUSES:
            status = 'unknown'
        result.append(dict(ordinal=ordinal, role=spec['role'], agent_id=spec['agent_id'],
            name=names.get(spec['agent_id'],'Unavailable agent'), status=status, step=step, task=spec['task']))
    return result


def elapsed_display(row, limits, current=None):
    recorded = max(0,row.get('active_seconds',0))
    if row['status'] != 'running' or row.get('lease_until') is None:
        return recorded
    # Display-only estimate from the existing coordinator lease, never a budget
    # decision. Authoritative enforcement remains monotonic in WorkflowEngine.
    estimated = limits['timeout_seconds'] + 10 - (row['lease_until'] - (time() if current is None else current))
    return min(limits['timeout_seconds'],max(recorded,estimated,0))


def metrics_view(row, limits, count):
    with st.container(key='workflow_metrics'):
        columns = st.columns(4)
        columns[0].metric('Model requests',row['requests'],help='Generation and embedding requests; failed requests may still be billable.')
        columns[0].caption(f"Limit {limits['max_requests']}")
        columns[1].metric('Charged / reserved tokens',f"{row['tokens']:,}",help='Known usage replaces reservations. This is not an invoice or a promise that another request will fit.')
        columns[1].caption(f"Budget {limits['token_budget']:,}")
        columns[2].metric('Active time',f"{elapsed_display(row,limits):.2f}s",help='Live estimate while running; measured checkpoint time otherwise. Human approval waits are excluded.')
        columns[2].caption(f"Limit {limits['timeout_seconds']}s")
        columns[3].metric('Tool calls',row['tool_calls'])
        columns[3].caption(f"Limit {limits['max_tool_calls']}")
    st.progress(min(1.0,max(0.0,row['cursor']/count)),text=f"{row['cursor']}/{count} validated steps")
    st.caption('Token and request counters are durable. Active time is estimated between measured checkpoints; all backend limits remain enforced.')


def choose_node(key, ordinal):
    st.session_state[key] = ordinal


def inspector(repo, node, row=None):
    with st.container(border=True,key='workflow_inspector'):
        st.subheader(f"Step {node['ordinal']+1} inspection")
        st.text(node['role'])
        st.text(node['name'])
        st.text(status_label(node['status']))
        st.caption('Agent ID: '+node['agent_id'])
        st.write('Step responsibility')
        st.text(node['task'])
        step = node['step']
        if step.get('output'):
            try:
                output = artifact(step['output'])
            except AppError:
                st.error('The saved handoff is invalid. Inspection does not repair or resume execution.')
            else:
                st.write('Handoff summary')
                st.text(output['summary'])
                st.write('Risks and open questions')
                if output['risks']:
                    for i,risk in enumerate(output['risks'],1):
                        st.text(f'{i}. {risk}')
                else:
                    st.caption('No risks listed by this agent. This is not independent verification.')
                with st.expander('Deliverable and structured handoff'):
                    st.markdown(safe_markdown(output['deliverable']),unsafe_allow_html=False)
                    st.json(output)
        elif node['status'] == 'waiting_approval':
            st.info('Awaiting an exact-action human decision. Review the file diff in Workflow controls below; inspecting this node grants nothing.')
        elif node['status'] == 'running':
            st.info('This step is running. Provisional output is not a validated handoff.')
        else:
            st.caption('No validated handoff is saved for this step.')
        with st.expander('Execution details'):
            if not step.get('run_id'):
                st.caption('No model execution has been recorded for this node.')
            else:
                with repo.connect() as db:
                    child = db.execute('SELECT status,requests,input_tokens,output_tokens,elapsed,error FROM runs WHERE id=?',(step['run_id'],)).fetchone()
                st.text('Execution ID: '+step['run_id'])
                st.text('Conversation ID: '+step['conversation_id'])
                if child:
                    st.text('Recorded execution status: '+child['status'])
                    st.text(f"{child['requests']} agent-stage attempt(s) · {child['elapsed']:.2f}s recorded time")
                    st.text('Input tokens: '+str(child['input_tokens'])+' · Output tokens: '+str(child['output_tokens']))
                    if child['error']:
                        st.error(safe_markdown(child['error']))
            if row:
                st.caption(f"Definition version {row['version']} · engine {row['data']['definition']['engine_version']}")
            st.caption('Inspection is read-only. Tools and action approvals are never granted by selecting a node.')


def visual_workflow(repo, value, agents, *, scope, row=None, steps=()):
    st.markdown(STYLE,unsafe_allow_html=True)
    nodes = nodes_for(value,agents,row,steps)
    selection = 'wf_selected_node_'+scope
    if selection not in st.session_state or st.session_state[selection] not in range(len(nodes)):
        st.session_state[selection] = min(row['cursor'],len(nodes)-1) if row else 0
    selected = st.session_state[selection]
    with st.container(key='workflow_canvas'):
        st.subheader('Agent workflow')
        st.caption('Sequential handoffs · numbered nodes show execution order. Inspect a step to review its handoff and recorded details.')
        if row:
            st.text('Workflow status: '+row['status'])
            st.text(status_label(row['status']))
            metrics_view(row,value['limits'],len(nodes))
        else:
            st.caption('Definition preview · no execution started and no permissions granted.')
        with st.container(key='workflow_nodes'):
            columns = st.columns(len(nodes))
            for node,column in zip(nodes,columns):
                i=node['ordinal']
                key=f"wf_node_{i}_status_{node['status']}"+('_chosen' if i==selected else '')+('_last' if i==len(nodes)-1 else '')
                with column, st.container(border=True,key=key):
                    st.caption('AGENT '+str(i+1))
                    st.text(node['role'])
                    st.text(node['name'])
                    st.text(status_label(node['status']))
                    st.button('Inspect step '+str(i+1),key='wf_inspect_'+scope+'_'+str(i),
                        type='primary' if i==selected else 'secondary',width='stretch',
                        help='View the handoff and execution details. This does not run the agent.',
                        on_click=choose_node,args=(selection,i))
                    if i==selected:
                        st.caption('Selected step')
        st.caption('Status legend: ○ Pending · ▶ Running · ✓ Completed · ! Failed · ■ Cancelled · ? Awaiting approval')
        inspector(repo,nodes[selected],row)
