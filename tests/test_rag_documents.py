"""Document validation, extraction, migrations and durable storage."""
from datetime import datetime, timedelta, timezone
import io
import sqlite3
import subprocess
from unittest.mock import patch
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from ai_agent_pro.config import AppError
from ai_agent_pro.database import Repository
from ai_agent_pro.migrations import BASE_SCHEMA
from ai_agent_pro.rag.chunking import chunk_sections
from ai_agent_pro.rag.extraction import extract
from ai_agent_pro.rag.models import MAX_FILE_BYTES, Section
from ai_agent_pro.rag.repository import KnowledgeRepository
from ai_agent_pro.rag.storage import DocumentStorage
from ai_agent_pro.rag.validation import validate_upload


def pdf_bytes(text='The deployment region is London.', encrypted=False, blank=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if not blank:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('private')
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def knowledge(repo):
    return KnowledgeRepository(repo)


@pytest.mark.parametrize('name,data', [('../escape.txt', b'hello'), ('C:\\escape.txt', b'hello'),
    ('file.exe', b'hello'), ('empty.txt', b''), ('bad.txt', b'\xff'), ('binary.txt', b'\x00hello'),
    ('fake.pdf', b'not a PDF'), ('fake.txt', b'%PDF-1.7'), ('dir/name.txt', b'hello'),
    ('bad\u202ename.txt', b'hello'), ('big.txt', b'x' * (MAX_FILE_BYTES + 1))])
def test_invalid_uploads(name, data):
    with pytest.raises(AppError):
        validate_upload(name, data)


def test_text_extraction_redaction_and_chunk_provenance():
    sections, warning = extract('txt', b'First line\npassword=private\nLondon deployment')
    assert warning is None
    assert 'private' not in sections[0].text
    chunks = chunk_sections(sections, size=32, overlap=4)
    assert chunks == chunk_sections(sections, size=32, overlap=4)
    assert all(len(c.text.encode()) <= 32 for c in chunks)
    assert chunks[0].location.startswith('lines 1')
    assert any('London' in c.text for c in chunks)


def test_unicode_chunks_preserve_text():
    text = '東京🌍 café ' * 100
    chunks = chunk_sections([Section(text, page=3)], size=64, overlap=0)
    assert ''.join(c.text for c in chunks) == text
    assert all(c.location == 'page 3' and len(c.text.encode()) <= 64 for c in chunks)


@pytest.mark.parametrize('size,overlap', [(0, 0), (100, 60), (100, -1), (True, 0)])
def test_invalid_chunk_configuration(size, overlap):
    with pytest.raises(AppError):
        chunk_sections([Section('hello')], size, overlap)


def test_real_pdf_worker_extracts_text():
    sections, warning = extract('pdf', pdf_bytes())
    assert 'London' in sections[0].text
    assert sections[0].page == 1
    assert warning is None


def test_pdf_page_limit_and_partial_ocr_warning():
    writer = PdfWriter()
    for _ in range(201):
        writer.add_blank_page(width=612, height=792)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(AppError, match='limits'):
        extract('pdf', output.getvalue())
    writer = PdfWriter()
    writer.append(io.BytesIO(pdf_bytes()))
    writer.add_blank_page(width=612, height=792)
    output = io.BytesIO()
    writer.write(output)
    sections, warning = extract('pdf', output.getvalue())
    assert len(sections) == 1 and sections[0].page == 1
    assert '1 pages' in warning


@pytest.mark.parametrize('data', [b'%PDF-broken', pdf_bytes(encrypted=True), pdf_bytes(blank=True)])
def test_reject_bad_encrypted_and_scanned_pdf(data):
    with pytest.raises(AppError):
        extract('pdf', data)


def test_extraction_timeout_and_child_has_no_key(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'private-value')
    def timeout(*args, **kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env']
        assert kwargs['timeout'] == 20
        raise subprocess.TimeoutExpired('parser', 20)
    with patch('ai_agent_pro.rag.extraction.subprocess.run', side_effect=timeout):
        with pytest.raises(AppError, match='timed out'):
            extract('pdf', pdf_bytes())


def test_upload_persists_and_deduplicates(repo, knowledge, tmp_path):
    cid = knowledge.create_collection('Engineering')
    storage = DocumentStorage(knowledge)
    did = storage.upload(cid, 'notes.txt', b'London deployment')
    assert storage.upload(cid, 'other.txt', b'London deployment') == did
    reopened = KnowledgeRepository(Repository(repo.path))
    assert reopened.document(did)['original'] == b'London deployment'
    assert len(reopened.documents(cid)) == 1
    reopened.delete_document(did)
    assert reopened.documents(cid) == []
    with pytest.raises(AppError):
        reopened.document(did)


def test_assignments_and_collection_delete(knowledge, agent):
    cid = knowledge.create_collection('Engineering')
    knowledge.assign(agent.id, [cid])
    assert knowledge.assigned(agent.id) == [cid]
    with pytest.raises(AppError):
        knowledge.assign(agent.id, ['missing'])
    assert knowledge.assigned(agent.id) == [cid]
    knowledge.delete_collection(cid)
    assert knowledge.assigned(agent.id) == []


def test_migrate_v1_preserves_existing_data(tmp_path):
    path = tmp_path / 'old.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(BASE_SCHEMA)
        db.execute('PRAGMA user_version=1')
        db.execute('INSERT INTO agents VALUES (?,?,?,?,?)', ('old', 'Legacy', '', 'Helpful', '{}'))
    repo = Repository(path)
    assert repo.agent('old').name == 'Legacy'
    Repository(path)
    with repo.connect() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 5


def test_future_schema_is_not_downgraded(tmp_path):
    path = tmp_path / 'future.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=99')
    with pytest.raises(AppError, match='newer'):
        Repository(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 99


def test_index_lock_recovery_and_active_assignment_guard(repo, knowledge, agent, conversation):
    cid = knowledge.create_collection('Engineering')
    did = DocumentStorage(knowledge).upload(cid, 'notes.txt', b'London deployment')
    jid = knowledge.begin_index(did)
    with pytest.raises(AppError, match='already'):
        knowledge.begin_index(did)
    with pytest.raises(AppError, match='Wait'):
        knowledge.recover_index(did)
    with repo.connect() as db:
        db.execute('UPDATE index_jobs SET started_at=? WHERE id=?', ((datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(), jid))
    knowledge.recover_index(did)
    knowledge.begin_index(did)
    repo.begin_run(conversation, agent, 'Question')
    with pytest.raises(AppError, match='running task'):
        knowledge.assign(agent.id, [cid])
