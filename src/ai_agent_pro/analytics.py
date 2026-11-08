"""Read-only, parameterized aggregates over retained application records.

Generation usage comes from completed stages exactly once, never summed again
with run totals. Null usage remains unknown; reservations are not usage.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ai_agent_pro.rag.models import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, PROCESSING_VERSION

PERIODS = {'All time': None, 'Last 7 days': 7, 'Last 30 days': 30, 'Last 90 days': 90}


def _period_start(period, current=None):
    """Return the first calendar day's midnight, or None for all time."""
    days = PERIODS[period]
    if not days:
        return None
    current = current or datetime.now(timezone.utc)
    return current.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)


@dataclass(frozen=True)
class Filters:
    agent_id: str | None = None
    period: str = 'All time'
    status: str = 'All'
    search: str = ''

    def where(self, current=None):
        clauses, params = ['1=1'], []
        if self.agent_id:
            clauses.append('c.agent_id=?')
            params.append(self.agent_id)
        start = _period_start(self.period, current)
        if start is not None:
            clauses.append('r.started_at>=?')
            params.append(start.isoformat())
        if self.status != 'All':
            clauses.append('r.status=?')
            params.append(self.status)
        if self.search:
            clauses.append("c.title LIKE ? ESCAPE '\\'")
            literal = self.search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            params.append(f'%{literal}%')
        return ' AND '.join(clauses), params


JOIN = 'FROM runs r JOIN conversations c ON c.id=r.conversation_id JOIN agents a ON a.id=c.agent_id'


class Analytics:
    def __init__(self, repository):
        self.repo = repository

    def count(self, filters):
        where, params = filters.where()
        with self.repo.connect() as db:
            return db.execute(f'SELECT COUNT(*) {JOIN} WHERE {where}', params).fetchone()[0]

    def runs(self, filters, limit=10, offset=0):
        where, params = filters.where()
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute(f'''SELECT r.*,c.title,c.agent_id,a.name AS agent_name
                {JOIN} WHERE {where} ORDER BY r.started_at DESC,r.id DESC LIMIT ? OFFSET ?''',
                [*params, min(max(limit, 1), 100), max(offset, 0)])]

    def position(self, run_id, filters):
        where, params = filters.where()
        with self.repo.connect() as db:
            row = db.execute(f'''WITH ranked AS (SELECT r.id,
                ROW_NUMBER() OVER (ORDER BY r.started_at DESC,r.id DESC)-1 AS position
                {JOIN} WHERE {where}) SELECT position FROM ranked WHERE id=?''', [*params, run_id]).fetchone()
        return row[0] if row else None

    def summary(self, filters):
        where, params = filters.where()
        with self.repo.connect() as db:
            counts = dict(db.execute(f'SELECT r.status,COUNT(*) {JOIN} WHERE {where} GROUP BY r.status', params).fetchall())
            # SQL median avoids loading every run into the presentation process.
            duration = db.execute(f'''WITH values_in_order AS (
                SELECT r.elapsed AS value {JOIN} WHERE {where} AND r.status IN ('completed','failed')
                ORDER BY r.elapsed), n AS (SELECT COUNT(*) AS count FROM values_in_order)
                SELECT AVG(value), (SELECT count FROM n) FROM
                (SELECT value FROM values_in_order LIMIT 2-(SELECT count FROM n)%2
                 OFFSET ((SELECT count FROM n)-1)/2)''', params).fetchone()
            usage = db.execute(f'''SELECT SUM(json_extract(s.value,'$.input_tokens')+json_extract(s.value,'$.output_tokens')) AS tokens,
                COUNT(*) AS stages,
                SUM(CASE WHEN json_extract(s.value,'$.input_tokens') IS NOT NULL
                          AND json_extract(s.value,'$.output_tokens') IS NOT NULL THEN 1 ELSE 0 END) AS known
                {JOIN}, json_each(r.steps) s WHERE {where}''', params).fetchone()
            query = db.execute(f'''SELECT SUM(u.input_tokens),COUNT(*),COUNT(u.input_tokens)
                {JOIN} JOIN retrieval_usage u ON u.run_id=r.id WHERE {where}''', params).fetchone()
        terminal = sum(counts.get(s, 0) for s in ('completed', 'failed', 'interrupted'))
        return {'tasks': sum(counts.values()), 'completed': counts.get('completed', 0),
                'waiting_approval': counts.get('waiting_approval', 0), 'running': counts.get('running', 0), 'failed': counts.get('failed', 0),
                'interrupted': counts.get('interrupted', 0),
                'completion_rate': counts.get('completed', 0) / terminal if terminal else None,
                'median_seconds': duration[0], 'duration_samples': duration[1] or 0,
                'generation_tokens': usage['tokens'], 'known_stages': usage['known'] or 0, 'stages': usage['stages'],
                'query_tokens': query[0], 'query_records': query[1], 'known_queries': query[2]}

    def series(self, filters):
        where, params = filters.where()
        with self.repo.connect() as db:
            activity = [dict(r) for r in db.execute(f'''SELECT substr(r.started_at,1,10) AS day,r.status,COUNT(*) AS tasks
                {JOIN} WHERE {where} GROUP BY day,r.status ORDER BY day,r.status''', params)]
            agents = [dict(r) for r in db.execute(f'''SELECT a.name AS agent,a.id AS agent_id,COUNT(*) AS tasks
                {JOIN} WHERE {where} GROUP BY a.id ORDER BY tasks DESC,a.name''', params)]
            tokens = [dict(r) for r in db.execute(f'''SELECT substr(r.started_at,1,10) AS day,
                SUM(json_extract(s.value,'$.input_tokens')) AS input_tokens,
                SUM(json_extract(s.value,'$.output_tokens')) AS output_tokens
                {JOIN}, json_each(r.steps) s WHERE {where}
                AND json_extract(s.value,'$.input_tokens') IS NOT NULL
                AND json_extract(s.value,'$.output_tokens') IS NOT NULL GROUP BY day ORDER BY day''', params)]
            errors = [dict(r) for r in db.execute(f'''SELECT r.error,COUNT(*) AS tasks {JOIN}
                WHERE {where} AND r.status IN ('failed','interrupted') AND r.error IS NOT NULL
                GROUP BY r.error ORDER BY tasks DESC''', params)]
            durations = [dict(r) for r in db.execute(f'''SELECT substr(r.started_at,1,10) AS day,
                AVG(r.elapsed) AS seconds,COUNT(*) AS samples {JOIN}
                WHERE {where} AND r.status IN ('completed','failed') GROUP BY day ORDER BY day''', params)]
        # Categories are fixed application text, never arbitrary model/document content.
        from ai_agent_pro.telemetry import error_category
        categories = {}
        for row in errors:
            category = error_category(row['error'])
            categories[category] = categories.get(category, 0) + row['tasks']
        return {'activity': activity, 'agents': agents, 'tokens': tokens, 'durations': durations,
                'errors': [{'category': key, 'tasks': value} for key, value in sorted(categories.items())]}

    def knowledge(self, agent_id=None, period='All time'):
        # Document counts are current inventory, not filtered historical task counts.
        scope = 'WHERE EXISTS(SELECT 1 FROM agent_collections ac WHERE ac.collection_id=d.collection_id AND ac.agent_id=?)' if agent_id else ''
        params = [agent_id] if agent_id else []
        with self.repo.connect() as db:
            row = db.execute(f'''SELECT COUNT(*) AS documents,COALESCE(SUM(d.size),0) AS bytes,
                SUM(CASE WHEN g.model=? AND g.dimensions=? AND g.version=? THEN 1 ELSE 0 END) AS ready
                FROM documents d LEFT JOIN index_generations g ON g.id=d.active_generation {scope}''',
                [EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, PROCESSING_VERSION, *params]).fetchone()
            job_conditions = ["1=1"]
            job_params = list(params)
            if agent_id:
                job_conditions.append('EXISTS(SELECT 1 FROM agent_collections ac WHERE ac.collection_id=d.collection_id AND ac.agent_id=?)')
            start = _period_start(period)
            if start is not None:
                job_conditions.append('j.started_at>=?')
                job_params.append(start.isoformat())
            usage = db.execute(f'''SELECT SUM(j.input_tokens),COUNT(*),COUNT(j.input_tokens)
                FROM index_jobs j JOIN documents d ON d.id=j.document_id WHERE {' AND '.join(job_conditions)}''', job_params).fetchone()
        return {**dict(row), 'ready': row['ready'] or 0, 'index_tokens': usage[0], 'index_jobs': usage[1], 'known_jobs': usage[2]}

    def index_durations(self, agent_id=None, period='All time'):
        conditions, params = ["j.status IN ('completed','failed')", 'j.elapsed IS NOT NULL'], []
        if agent_id:
            conditions.append('EXISTS(SELECT 1 FROM agent_collections ac WHERE ac.collection_id=d.collection_id AND ac.agent_id=?)')
            params.append(agent_id)
        start = _period_start(period)
        if start is not None:
            conditions.append('j.started_at>=?')
            params.append(start.isoformat())
        with self.repo.connect() as db:
            return [dict(r) for r in db.execute(f'''SELECT substr(j.started_at,1,10) AS day,
                AVG(j.elapsed) AS seconds,COUNT(*) AS samples FROM index_jobs j JOIN documents d ON d.id=j.document_id
                WHERE {' AND '.join(conditions)} GROUP BY day ORDER BY day''', params)]
