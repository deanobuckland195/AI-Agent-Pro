"""Accessible native controls and a small, static visual system."""
from datetime import datetime, timezone
import math
import streamlit as st
from ai_agent_pro.presentation.theme import DARK_STYLE

ROUTES = ('Overview', 'Agents', 'Chat', 'Knowledge base', 'Activity', 'Workflows', 'Help')


def navigate(route, agent_id=None, mode=None):
    """Button callback, applied before widgets are constructed on the next rerun."""
    st.session_state['route'] = route
    if agent_id:
        st.session_state['active_agent'] = agent_id
    if mode:
        st.session_state['agents_mode'] = mode


def page_header(title, description):
    st.header(title)
    st.caption(description)


def format_bytes(size):
    if size < 1024:
        return f'{size:,} B'
    if size < 1024**2:
        return f'{size / 1024:.1f} KiB'
    return f'{size / 1024**2:.1f} MiB'


def apply_style():
    # Static, application-owned CSS only. No user/model content is interpolated.
    st.markdown(DARK_STYLE, unsafe_allow_html=True)


def notice(message):
    st.session_state['notice'] = message


def view_state(restore=False):
    """Preserve view filters when Streamlit cleans up widgets on another screen."""
    keys = ('agent_search', 'agents_page', 'agents_mode', 'library_collection', 'document_search',
            'document_kind', 'document_state', 'documents_page', 'activity_scope', 'activity_status',
            'activity_period', 'activity_page', 'activity_search', 'activity_task', 'overview_scope', 'overview_period')
    memory = st.session_state.setdefault('_view_state', {})
    if restore:
        for key, value in memory.items():
            if key not in st.session_state:
                st.session_state[key] = value
    else:
        memory.update({k: st.session_state[k] for k in st.session_state if k in keys or k.startswith('conversation_')})


def recovery_wait(started_at, deadline_seconds, current=None):
    current = current or datetime.now(timezone.utc)
    age = (current - datetime.fromisoformat(started_at)).total_seconds()
    return max(0, math.ceil(deadline_seconds + 10 - age))


def paginate(rows, key, size=10):
    pages = max(1, math.ceil(len(rows) / size))
    if st.session_state.get(key, 1) > pages:
        st.session_state[key] = 1
    page = st.selectbox('Page', list(range(1, pages + 1)), key=key,
                        format_func=lambda value: f'{value} of {pages}')
    start = (page - 1) * size
    st.caption(f'{len(rows)} matching records · showing {start + 1 if rows else 0}–{min(start + size, len(rows))}')
    return rows[start:start + size]


def feedback(error):
    st.error(str(error))
    message = str(error).lower()
    if 'credential' in message or 'api_key' in message:
        st.caption('Configure OPENAI_API_KEY privately, then submit the action again. No automatic retry is made.')
    elif 'quota' in message or 'rate' in message:
        st.caption('Check your OpenAI quota or wait before trying again. Failed requests may still be billed.')
    elif 'termination reason: max_output_tokens' in message:
        st.caption('Open Agent settings → Output tokens per request. For a long tutorial, try 4,096–8,192 '
                   '(up to 16,384), keeping enough total task budget for input, output and any review stages. '
                   'If already at the maximum, split the question into smaller parts. Retry manually when ready; '
                   'the previous incomplete answer will not be reused.')
    elif 'termination reason: content_filter' in message:
        st.caption('Review or rephrase the request. Increasing the output cap does not resolve content filtering.')
    elif 'budget' in message or 'reservation' in message:
        st.caption('Shorten the question or context, or adjust the agent budget in Agents.')
    elif 'pdf' in message:
        st.caption('Use a valid, unencrypted PDF with selectable text, or upload UTF-8 text. OCR is not available.')
    else:
        st.caption('Review the action and try again when ready. Existing conversations and ready indexes are retained.')
