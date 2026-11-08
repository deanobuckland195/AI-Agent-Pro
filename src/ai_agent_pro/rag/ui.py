"""Document library, collection assignments and source-version-aware viewer."""
import asyncio
import streamlit as st
from ai_agent_pro.rag.embeddings import OpenAIEmbeddings
from ai_agent_pro.rag.repository import KnowledgeRepository
from ai_agent_pro.rag.service import KnowledgeBase
from ai_agent_pro.rag.models import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, PROCESSING_VERSION
from ai_agent_pro.presentation.components import page_header, paginate, recovery_wait, notice, format_bytes
from ai_agent_pro.presentation.progress import live_progress, workflow_view
from ai_agent_pro.telemetry import events


def index_ready(document, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS):
    return bool(document['active_generation'] and document['model'] == model and
                document['dimensions'] == dimensions and document['version'] == PROCESSING_VERSION)


def source_panels(repo, run_id, agent_id):
    sources = KnowledgeRepository(repo).sources(run_id, agent_id)
    if not sources:
        return
    with st.expander('Retrieved sources (citations identify excerpts, not verified facts)'):
        ids = [s['label'] for s in sources]
        selected = st.radio('Source excerpt', ids, horizontal=True, key=f'source_{run_id}')
        source = next(s for s in sources if s['label'] == selected)
        st.text(f"[{source['label']}] {source['filename']} · {source['location']}")
        if source['excerpt'] is None:
            st.caption('This document was deleted or is no longer assigned to this agent.')
        else:
            st.text(source['excerpt'])
            st.caption(f"Saved index generation: {source['generation_id']}. This is the excerpt used for this answer, even after re-indexing.")


def knowledge_view(repo, agent, key):
    kb = KnowledgeBase(repo, OpenAIEmbeddings(key) if key else None)
    knowledge = kb.repository
    page_header('Knowledge base', 'A searchable document library with explicit agent access and traceable sources.')
    st.button('Refresh library', icon=':material/refresh:')
    with st.expander('Upload limits & data privacy'):
        st.write('PDF and UTF-8 text only, up to 10 MiB. Scanned PDFs require OCR and are unsupported. '
                 'Original files retain original bytes in local SQLite; do not upload secrets. '
                 'Uploading alone makes no API request. Indexing sends extracted text to OpenAI for embeddings; '
                 'document chat sends the question and selected excerpts.')
    with st.expander('Create a collection', expanded=not knowledge.collections()):
        with st.form('create_collection'):
            name = st.text_input('Collection name', max_chars=80)
            if st.form_submit_button('Create collection', icon=':material/create_new_folder:'):
                cid = knowledge.create_collection(name)
                st.session_state['_collection_request'] = cid
                notice('Collection created. Upload a document to get started.')
                st.rerun()
    collections = knowledge.collections()
    if not collections:
        st.info('Create a collection, then upload PDF or UTF-8 text documents.')
        return
    ids = [c['id'] for c in collections]
    labels = {c['id']: c['name'] for c in collections}
    requested = st.session_state.pop('_collection_request', None)
    if requested in ids:
        st.session_state['library_collection'] = requested
    if st.session_state.get('library_collection') not in ids:
        st.session_state['library_collection'] = ids[0]
    cid = st.selectbox('Manage collection', ids, format_func=labels.get, key='library_collection')
    if agent:
        with st.expander('Agent access', expanded=agent.settings.rag_enabled and not knowledge.assigned(agent.id)):
            st.text(f'Active agent: {agent.name}')
            with st.form(f'assign_collections_{agent.id}'):
                selected = st.multiselect('Collections available to the active agent', ids,
                                          default=knowledge.assigned(agent.id), format_func=labels.get)
                if st.form_submit_button('Save collection assignments', icon=':material/save:'):
                    knowledge.assign(agent.id, selected)
                    notice('Assignments saved. Enable Document-grounded answers in Agents if needed.')
                    st.rerun()
    with st.container(border=True):
        st.subheader('Add a document')
        upload = st.file_uploader('PDF or UTF-8 text document (maximum 10 MiB)', type=['pdf', 'txt'])
        if st.button('Upload document', disabled=upload is None, icon=':material/upload_file:', type='primary'):
            previous = {d['id'] for d in knowledge.documents(cid)}
            did = kb.storage.upload(cid, upload.name, upload.getvalue())
            notice('Already in this collection. The existing document was retained.' if did in previous else
                   'Document uploaded. Select Index document to create embeddings.')
            st.rerun()
    documents = knowledge.library_documents(cid)
    model = kb.indexer.embeddings.model if kb.indexer else EMBEDDING_MODEL
    dimensions = kb.indexer.embeddings.dimensions if kb.indexer else EMBEDDING_DIMENSIONS
    a, b, c = st.columns(3)
    a.metric('Documents', len(documents))
    b.metric('Compatible indexes', sum(index_ready(d, model, dimensions) for d in documents))
    c.metric('Original uploads', format_bytes(sum(d['size'] for d in documents)))
    search = st.text_input('Search documents', key='document_search', placeholder='Filter by filename')
    left, right = st.columns(2)
    kind = left.selectbox('File type', ['All', 'pdf', 'txt'], key='document_kind')
    state = right.selectbox('Index readiness', ['All', 'Ready', 'Needs indexing'], key='document_state')
    filtered = [d for d in documents if search.casefold() in d['filename'].casefold()
                and (kind == 'All' or d['kind'] == kind)
                and (state == 'All' or index_ready(d, model, dimensions) == (state == 'Ready'))]
    st.subheader('Document library')
    if not filtered:
        st.info('No documents match these filters.' if documents else 'No documents yet. Upload a document above.')
    else:
        page = paginate(filtered, 'documents_page')
        st.dataframe([{'Filename': d['filename'], 'Type': d['kind'].upper(), 'Bytes': d['size'],
                       'Active index': 'Ready' if index_ready(d, model, dimensions) else 'Needs indexing',
                       'Latest attempt': d['last_status'] or 'Not started', 'Chunks': d['chunks']}
                      for d in page], hide_index=True, width='stretch')
        selected = st.selectbox('Document details', [d['id'] for d in page],
                                format_func=lambda did: next(d['filename'] for d in page if d['id'] == did))
        document = next(d for d in page if d['id'] == selected)
        document_details(repo, kb, document, key, index_ready(document, model, dimensions))
    with st.expander('Delete collection'):
        st.caption('Removes documents, embeddings, assignments and retained citation excerpts. Historical answers may '
                   'still quote deleted material; clear those conversations separately. SQLite deletion is not secure erasure.')
        confirmed = st.checkbox('Delete this collection and all its documents', key=f'delete_collection_confirm_{cid}')
        if st.button('Delete collection', disabled=not confirmed, icon=':material/delete:'):
            knowledge.delete_collection(cid)
            notice('Collection and its documents deleted.')
            st.rerun()
    if not key:
        st.caption('Configure OPENAI_API_KEY privately to enable indexing and document chat.')


def document_details(repo, kb, document, key, ready):
    knowledge, did = kb.repository, document['id']
    with st.container(border=True):
        st.subheader(document['filename'])
        st.text(f"Active index: {'Ready for retrieval' if ready else 'Needs indexing'} · Latest attempt: {document['last_status'] or 'Not started'}")
        if document['active_generation']:
            st.caption(f"{document['chunks']} stored chunks · {document['model']} · {document['dimensions']} dimensions")
            st.caption(f"Generation: {document['active_generation']}")
        if document['error']:
            st.error(document['error'])
            if ready:
                st.caption('The previous ready index remains available.')
        if document['warning']:
            st.warning(document['warning'])
        running = document['last_status'] == 'running'
        action = 'Re-index document' if document['active_generation'] else 'Index document'
        if st.button(action, key=f'index_{did}', disabled=not key or running, icon=':material/sync:', type='primary'):
            with st.status('Extracting and indexing document…', expanded=True) as status:
                kb.indexer.observer = live_progress(status)
                asyncio.run(kb.indexer.index(did))
                status.update(label='Index published', state='complete')
            notice('Document indexed and ready for retrieval.')
            st.rerun()
        if running:
            wait = recovery_wait(document['job_started_at'], document['deadline_seconds'])
            st.caption(f'Recovery available in {wait}s. Recovery does not cancel a live request.')
            if st.button('Recover interrupted indexing', key=f'recover_index_{did}', disabled=wait > 0):
                knowledge.recover_index(did)
                notice('Interrupted indexing recovered. Index again when ready.')
                st.rerun()
        jobs = knowledge.jobs(did)
        with st.expander('Indexing history & usage'):
            if not jobs:
                st.caption('No indexing attempts yet.')
            for job in jobs:
                st.text(f"{job['status'].upper()} · {job['started_at']} · {job['requests']} embedding requests")
                st.caption(f"Reported embedding tokens: {job['input_tokens'] if job['input_tokens'] is not None else 'Unavailable'} · Reservation: {job['reserved_tokens']}")
                st.caption(f"Recorded duration: {str(round(job['elapsed'], 2)) + 's' if job['elapsed'] is not None else 'Unavailable'}")
                workflow_view(events(repo, job_id=job['id']))
            st.caption('Known usage is separate from generation totals. Failed calls may still be billed.')
        with st.expander('Delete document'):
            confirmed = st.checkbox('Delete original, chunks, vectors and retained source excerpts', key=f'delete_doc_confirm_{did}')
            if st.button('Delete document', key=f'delete_doc_{did}', disabled=not confirmed, icon=':material/delete:'):
                knowledge.delete_document(did)
                notice('Document and its retained source excerpts deleted.')
                st.rerun()
