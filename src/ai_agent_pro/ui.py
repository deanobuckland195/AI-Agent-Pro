"""Application shell: navigation, dependencies and safe top-level errors."""
import os
from pathlib import Path
import sqlite3
import streamlit as st
from ai_agent_pro.config import AppError
from ai_agent_pro.database import Repository
from ai_agent_pro.safety import configure_logging
from ai_agent_pro.presentation.components import ROUTES, apply_style, navigate, page_header, feedback, view_state
from ai_agent_pro.presentation.agents import agents_view
from ai_agent_pro.presentation.workspace import workspace
from ai_agent_pro.presentation.activity import history_view
from ai_agent_pro.presentation.workflows import workflows_view
from ai_agent_pro.presentation.dashboard import overview
from ai_agent_pro.rag.ui import knowledge_view

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_repository():
    return Repository(os.environ.get('AI_AGENT_PRO_DB', str(PROJECT_ROOT / 'data' / 'agentpro.sqlite3')))


def api_key():
    value = os.environ.get('OPENAI_API_KEY', '').strip()
    if value:
        return value
    try:
        return str(st.secrets.get('OPENAI_API_KEY', '')).strip()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ''


def help_view(key):
    page_header('Help & setup', 'A local, evidence-aware workspace. Clear limits and private configuration.')
    st.subheader('Get started')
    st.write('1. Create an agent in Agents.\n2. Open Chat for an ordinary conversation.\n'
             '3. For document answers, upload and index files in Knowledge base, assign collections, '
             'then enable Document-grounded answers in the agent settings.')
    st.info('Credential present — connection unverified' if key else
            'API credentials missing. Set OPENAI_API_KEY in your environment or private .streamlit/secrets.toml.')
    st.subheader('Data and privacy')
    st.write('Original uploads, extracted text and conversations are stored locally in SQLite. '
             'Indexing sends extracted text to OpenAI; chat sends your question and applicable context. '
             'Original uploads retain original bytes. Do not upload secrets.')
    st.warning('Trusted local use only: no authentication or encryption at rest. Prompt-injection '
               'boundaries and redaction are best-effort. Review model answers before use.')
    st.subheader('Supported capabilities')
    st.write('Text chat, bounded draft/review stages, PDF/text retrieval and source citations. '
             'Explicitly approved read-only tools, bounded arithmetic and Python syntax inspection. '
             'Practice text replacements pause for an exact-diff human decision before writing. '
             'No shell execution, browsing, OCR, background queue or automatic paid retries. '
             'Document mode uses fresh sources and omits earlier conversation context.')
    st.caption('Recorded durations describe application execution, not independent verification or model confidence. '
               'Usage can be unavailable, and failed requests can still be billed. Recovery does not cancel a live request.')


def main():
    st.set_page_config(page_title='AI Agent Pro', page_icon=':material/smart_toy:', layout='wide')
    configure_logging()
    apply_style()
    view_state(restore=True)
    try:
        repo, key = get_repository(), api_key()
        agents = repo.agents()
        ids = [a.id for a in agents]
        requested = st.session_state.pop('_agent_request', None)
        if requested in ids:
            st.session_state['active_agent'] = requested
        if st.session_state.get('active_agent') not in ids:
            st.session_state['active_agent'] = ids[0] if ids else None
        with st.sidebar:
            with st.container(key='brand'):
                st.title('AI Agent Pro')
                st.caption('SOFTWARE ENGINEERING WORKSPACE')
            page = st.radio('Navigate', ROUTES, key='route')
            st.divider()
            selected = st.selectbox('Active agent', ids, key='active_agent',
                                    format_func=lambda aid: next(a.name for a in agents if a.id == aid),
                                    placeholder='Create your first agent')
            st.caption('Credential present — connection unverified' if key else 'API credentials missing')
            st.caption('Local SQLite · PDF & text knowledge')
        job = st.session_state.get('_chat_job')
        if job and (page != 'Chat' or selected != job.agent_id):
            job.cancel()
        workflow_job = st.session_state.get('_workflow_job')
        if workflow_job and page != 'Workflows':
            workflow_job.cancel()
        if 'notice' in st.session_state:
            st.success(st.session_state.pop('notice'))
        agent = repo.agent(selected) if selected else None
        if page == 'Overview':
            overview(repo)
        elif page == 'Agents':
            agents_view(repo, agent)
        elif page == 'Knowledge base':
            knowledge_view(repo, agent, key)
        elif page == 'Help':
            help_view(key)
        elif page == 'Workflows':
            workflows_view(repo,key)
        elif page == 'Activity':
            history_view(repo, agent)
        elif agent:
            workspace(repo, agent, key)
        else:
            page_header('Chat', 'Choose an assistant to start a persistent conversation.')
            st.info('Create an agent before starting a conversation.')
            st.button('Create an agent', on_click=navigate, args=('Agents', None, 'Create agent'), type='primary')
    except AppError as exc:
        feedback(exc)
    except sqlite3.Error:
        st.error('Local database operation failed. Check directory permissions and available disk space.')
    except OSError:
        st.error('Local storage operation failed. Check directory permissions and available disk space.')
    finally:
        view_state()
