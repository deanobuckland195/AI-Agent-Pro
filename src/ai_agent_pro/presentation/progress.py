"""Live stage feedback and historical workflows from recorded events only."""
import streamlit as st
from ai_agent_pro.telemetry import STAGES


def workflow_rows(records):
    """No inferred stages, timings or success states for historical records."""
    rows = {}
    for event in records:
        key = (event['stage'], event['ordinal'])
        row = rows.setdefault(key, {'Stage': STAGES[event['stage']], 'Sequence': event['ordinal'],
                                    'State': 'Unknown', 'Started (UTC)': None, 'Duration (s)': None,
                                    'Completed units': None, 'Total units': None, 'Outcome': None})
        if event['state'] == 'started':
            row['Started (UTC)'] = event['recorded_at']
        row['State'] = event['state']
        row['Last event (UTC)'] = event['recorded_at']
        row['Duration (s)'] = event['elapsed_seconds']
        row['Completed units'], row['Total units'] = event['completed_units'], event['total_units']
        row['Outcome'] = event['outcome'] or event['error_category']
    return list(rows.values())


def workflow_view(records):
    rows = workflow_rows(records)
    if not rows:
        st.caption('Detailed stage telemetry is unavailable for this record. No timing is inferred.')
        return
    # A native step list is responsive and screen-reader friendly; table exposes timings.
    for row in rows:
        icon = {'completed': '✓', 'failed': '!', 'started': '…'}[row['State']]
        st.text(f"{icon} {row['Stage']}{' ' + str(row['Sequence']) if row['Sequence'] else ''} · {row['State']}")
    st.dataframe(rows, hide_index=True, width='stretch')
    st.caption('Only recorded events are shown. A started event without a terminal event has an unknown outcome. '
               'Completed units refer to that stage, not the overall task.')


def live_progress(status):
    label, bar = st.empty(), st.empty()

    def observe(event):
        stage = STAGES[event['stage']]
        state = event['state']
        status.update(label=f'{stage} · {state}', state='running')
        label.text(f"{stage}{' ' + str(event['ordinal']) if event['ordinal'] else ''}: {state}")
        if state == 'started':
            bar.empty()
        if event['outcome'] == 'no_evidence':
            label.text('No supporting evidence found. Saving the insufficient-evidence response.')
        elif event['outcome'] == 'cached':
            label.text('Unchanged chunks reused: no new embedding requests were needed.')
        elif event['total_units'] and event['completed_units'] is not None:
            completed, total = event['completed_units'], event['total_units']
            if 0 <= completed <= total:
                noun = 'Embedding batches' if event['stage'] == 'embeddings' else 'Generation stages'
                bar.progress(completed / total, text=f'{noun}: {completed}/{total}')
    return observe
