"""Offline lexical search over assigned, current indexed text. No embeddings/API calls."""
from contextlib import contextmanager
import re
import sqlite3

from ai_agent_pro.rag.models import PROCESSING_VERSION
from ai_agent_pro.safety import redact
from ai_agent_pro.tools.contracts import ToolFailure
from ai_agent_pro.tools.sqlite_snapshot import snapshot


@contextmanager
def connection(context):
    try:
        data = snapshot(context)
        # SQLite only sees an in-memory image, never the live pathname, fd alias
        # or untrusted sidecar names. There is intentionally no disk fallback.
        db = sqlite3.connect(':memory:')
        try:
            db.deserialize(data)
            del data
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA trusted_schema=OFF')
            db.execute('PRAGMA temp_store=MEMORY')
            db.execute('PRAGMA query_only=ON')
            tables = {'agents', 'documents', 'index_generations', 'document_chunks', 'agent_collections'}
            def authorize(action, table, column, database, trigger):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and database == 'main' and table in tables:
                    return sqlite3.SQLITE_OK
                # No ATTACH, writes, PRAGMA changes, extensions or SQL functions.
                return sqlite3.SQLITE_DENY
            db.set_authorizer(authorize)
            yield db
        finally:
            db.close()
    except (OSError, ValueError, sqlite3.Error, AttributeError):
        raise ToolFailure('permission_denied') from None


def run(arguments, context):
    terms = sorted(set(re.findall(r'[^\W_]+', arguments['query'].casefold())))
    if not terms or len(terms) > 20:
        raise ToolFailure('invalid_input')
    results, hashes = [], set()
    with connection(context) as db:
        if not db.execute('SELECT 1 FROM agents WHERE id=?', (context['agent_id'],)).fetchone():
            raise ToolFailure('permission_denied')
        rows = db.execute('''SELECT c.text,c.location,c.content_hash,d.id AS document_id,
            d.filename,g.id AS generation_id FROM document_chunks c
            JOIN index_generations g ON g.id=c.generation_id
            JOIN documents d ON d.active_generation=g.id
            JOIN agent_collections a ON a.collection_id=d.collection_id
            WHERE a.agent_id=? AND g.version=? ORDER BY d.id,c.ordinal LIMIT 20001''',
            (context['agent_id'], PROCESSING_VERSION))
        count = 0
        for row in rows:
            count += 1
            if count > 20000 or len(row['text'].encode('utf-8')) > 4000:
                raise ToolFailure('resource_limit')
            text = redact(row['text'])
            if all(term in text.casefold() for term in terms) and row['content_hash'] not in hashes:
                hashes.add(row['content_hash'])
                if len(results) < arguments['max_results']:
                    results.append(dict(label=f'S{len(results) + 1}', document_id=row['document_id'],
                        generation_id=row['generation_id'], filename=redact(row['filename']),
                        location=redact(row['location']), excerpt=text))
    return dict(method='local_lexical', sources=results)


def revalidate(result, context):
    """Recheck current assignments/generations after worker completion, before release."""
    with connection(context) as db:
        for source in result['sources']:
            if not db.execute('''SELECT 1 FROM documents d
                JOIN agent_collections a ON a.collection_id=d.collection_id
                JOIN index_generations g ON g.id=d.active_generation
                WHERE a.agent_id=? AND d.id=? AND d.active_generation=? AND g.version=?''',
                (context['agent_id'], source['document_id'], source['generation_id'], PROCESSING_VERSION)).fetchone():
                raise ToolFailure('permission_denied')
