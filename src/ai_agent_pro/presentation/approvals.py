"""Human-only action decisions. Never interpret model text as approval."""
from datetime import datetime, timezone
import streamlit as st
from ai_agent_pro.approvals import Approvals
from ai_agent_pro.config import AppError
from ai_agent_pro.execution import ChatJob
from ai_agent_pro.presentation.components import feedback


def approval_panel(repo, agent, cid, key, workspace, provider_factory, busy=False, resume_callback=None):
    if resume_callback is None:
        with repo.connect() as db:
            managed = db.execute('SELECT 1 FROM workflow_steps WHERE conversation_id=?',(cid,)).fetchone()
        if managed:
            st.info('This approval belongs to a workflow. Review and resume it from Workflows so workflow limits and step identity remain enforced.')
            return
    service = Approvals(repo, workspace)
    for row in service.list(agent.id, cid):
        p = row['payload']
        with st.expander('File change approval · ' + row['status'], expanded=row['status'] in ('pending', 'approved')):
            st.text('Exact file: ' + p['path'])
            st.caption(f"Action: {row['id']} · Execution: {row['run_id']} · practice_replace v1.0.0")
            st.caption('Expires (UTC): ' + datetime.fromtimestamp(p['expires_at'], timezone.utc).isoformat())
            st.text('Before SHA-256: ' + p['before_hash'])
            st.text('After SHA-256: ' + p['after_hash'])
            st.write('Before')
            st.code(p['before'], language='text')
            st.write('After')
            st.code(p['arguments']['content'], language='text')
            st.write('Exact diff')
            st.code(service.diff(row), language='diff')
            if row['error']:
                st.warning(row['error'])
            if row['status'] == 'pending':
                st.caption('Approval authorizes only this exact change. Denial or cancellation never writes a file.')
                approve = st.button('Approve file change', key='approve_' + row['id'], disabled=busy or not key)
                deny = st.button('Deny file change', key='deny_' + row['id'], disabled=busy)
                if approve or deny:
                    try:
                        service.decide(row['id'], agent.id, row['run_id'], row['binding'], approve)
                    except AppError as exc:
                        feedback(exc)
                        return
                    if approve and resume_callback:
                        resume_callback(row)
                    elif approve:
                        st.session_state['_chat_job'] = ChatJob(repo, provider_factory(key), agent.id, cid, '',
                            approval_id=row['id'], practice=workspace).start()
                    st.rerun(scope='app')
            elif row['status'] in ('approved', 'applying', 'applied'):
                with repo.connect() as db:
                    run = db.execute('SELECT status FROM runs WHERE id=?', (row['run_id'],)).fetchone()
                if run and run['status'] == 'waiting_approval':
                    if st.button('Resume approved execution', key='resume_' + row['id'], disabled=busy or not key):
                        if resume_callback:
                            resume_callback(row)
                        else:
                            st.session_state['_chat_job'] = ChatJob(repo, provider_factory(key), agent.id, cid, '',
                                approval_id=row['id'], practice=workspace).start()
                        st.rerun(scope='app')
            if row['status'] in ('pending', 'approved') and st.button('Cancel paused execution', key='cancel_' + row['id'], disabled=busy):
                service.cancel(row['id'], agent.id, row['run_id'])
                st.rerun(scope='app')
