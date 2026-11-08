"""Real-record aggregation contracts and charts without fabricated values."""
from datetime import datetime, timezone
import pytest
from ai_agent_pro.analytics import Analytics, Filters
from ai_agent_pro.config import Agent
from ai_agent_pro.presentation.dashboard import chart_spec
from ai_agent_pro.rag.repository import KnowledgeRepository


def record(repo, agent, status='completed', elapsed=1, steps=None, error=None, title='Question'):
    cid = repo.new_conversation(agent.id)
    rid = repo.begin_run(cid, agent, title)
    if status != 'running':
        repo.finish_run(rid, status, 1, 9999 if status == 'completed' else None,
                        9999 if status == 'completed' else None, elapsed,
                        steps or [], error=error)
    return rid


def test_empty_analytics_has_no_invented_metrics(repo):
    analytics = Analytics(repo)
    value = analytics.summary(Filters())
    assert value['tasks'] == 0
    assert value['completion_rate'] is None
    assert value['generation_tokens'] is None
    assert value['median_seconds'] is None
    assert all(not rows for rows in analytics.series(Filters()).values())
    assert analytics.knowledge()['documents'] == 0
    assert chart_spec([], 'activity') is None


def test_known_usage_counted_once_and_failed_stages_retained(repo, agent):
    record(repo, agent, elapsed=2, steps=[{'input_tokens': 10, 'output_tokens': 5}])
    record(repo, agent, 'failed', elapsed=8, steps=[{'input_tokens': 7, 'output_tokens': 3}], error='Quota limit reached')
    record(repo, agent, 'completed', elapsed=4, steps=[{'input_tokens': None, 'output_tokens': None}])
    record(repo, agent, 'running')
    record(repo, agent, 'interrupted', elapsed=0, error='Interrupted session')
    summary = Analytics(repo).summary(Filters())
    assert summary['tasks'] == 5
    assert summary['generation_tokens'] == 25  # not 39,996 from overlapping run totals
    assert summary['known_stages'] == 2 and summary['stages'] == 3
    assert summary['median_seconds'] == 4 and summary['duration_samples'] == 3
    assert summary['completion_rate'] == 0.5
    series = Analytics(repo).series(Filters())
    assert sum(row['input_tokens'] + row['output_tokens'] for row in series['tokens']) == 25
    assert sum(row['tasks'] for row in series['errors']) == 2
    assert series['durations'][0]['samples'] == 3


def test_filters_pagination_and_literal_search(repo, agent):
    other = Agent.create('Other', '', 'Helpful')
    repo.save_agent(other)
    first = record(repo, agent, title='100%_ready')
    record(repo, agent, title='ordinary')
    record(repo, other, 'failed')
    analytics = Analytics(repo)
    assert analytics.count(Filters(agent.id)) == 2
    assert analytics.count(Filters(status='failed')) == 1
    assert [r['id'] for r in analytics.runs(Filters(search='%_'))] == [first]
    assert analytics.count(Filters(search="'; DROP TABLE agents; --")) == 0
    assert len(analytics.runs(Filters(), limit=1, offset=1)) == 1
    assert analytics.runs(Filters(), offset=50) == []
    assert len(repo.agents()) == 2


def test_utc_date_boundary():
    where, params = Filters(period='Last 7 days').where(datetime(2026, 11, 4, 23, 59, tzinfo=timezone.utc))
    assert params == ['2026-10-29T00:00:00+00:00']
    assert 'started_at>=?' in where


def test_query_embedding_usage_does_not_become_generation_usage(repo, agent):
    rid = record(repo, agent)
    knowledge = KnowledgeRepository(repo)
    knowledge.charge_query(rid, 500)
    knowledge.query_usage(rid, 12)
    unknown = record(repo, agent)
    knowledge.charge_query(unknown, 300)
    summary = Analytics(repo).summary(Filters())
    assert summary['query_tokens'] == 12
    assert summary['query_records'] == 2 and summary['known_queries'] == 1
    assert summary['generation_tokens'] is None


def test_clear_removes_retained_activity(repo, agent):
    rid = record(repo, agent)
    cid = repo.runs(agent.id)[0]['conversation_id']
    repo.clear_conversation(cid)
    assert Analytics(repo).count(Filters()) == 0


@pytest.mark.parametrize('kind,rows', [
    ('activity', [{'day': '2026-11-04', 'status': 'completed', 'tasks': 3}]),
    ('tokens', [{'day': '2026-11-04', 'input_tokens': 10, 'output_tokens': 2}]),
    ('durations', [{'day': '2026-11-04', 'seconds': 1.2, 'samples': 4}]),
    ('errors', [{'category': 'Timeout', 'tasks': 2}]),
    ('agents', [{'agent': 'Assistant', 'agent_id': 'id', 'tasks': 4}]),
])
def test_chart_spec_contains_only_supplied_measurements(kind, rows):
    spec = chart_spec(rows, kind).to_dict()
    datasets = list(spec.get('datasets', {}).values())
    assert rows in datasets or spec['data'].get('values') == rows
    assert 'tooltip' in spec['encoding']


def test_inventory_scoped_before_counts_and_not_multiplied(repo, agent):
    from ai_agent_pro.rag.storage import DocumentStorage
    knowledge = KnowledgeRepository(repo)
    cid = knowledge.create_collection('Shared')
    knowledge.assign(agent.id, [cid])
    other = Agent.create('Other', '', 'Helpful')
    repo.save_agent(other)
    knowledge.assign(other.id, [cid])
    DocumentStorage(knowledge).upload(cid, 'notes.txt', b'London')
    assert Analytics(repo).knowledge()['documents'] == 1
    assert Analytics(repo).knowledge(agent.id)['documents'] == 1
    assert Analytics(repo).knowledge('unassigned')['documents'] == 0


def test_legacy_index_durations_remain_unavailable(repo):
    from ai_agent_pro.rag.storage import DocumentStorage
    knowledge = KnowledgeRepository(repo)
    cid = knowledge.create_collection('Documents')
    did = DocumentStorage(knowledge).upload(cid, 'notes.txt', b'hello')
    legacy = knowledge.begin_index(did)
    knowledge.fail_index(legacy, 'Failure')
    with repo.connect() as db:
        db.execute('UPDATE index_jobs SET elapsed=NULL,finished_at=NULL WHERE id=?', (legacy,))
    assert Analytics(repo).index_durations() == []
    current = knowledge.begin_index(did)
    knowledge.fail_index(current, 'Failure')
    rows = Analytics(repo).index_durations()
    assert rows[0]['samples'] == 1 and rows[0]['seconds'] >= 0
