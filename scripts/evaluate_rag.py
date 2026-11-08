"""Explicitly opt-in paid semantic retrieval evaluation in a disposable local DB.

Run: python scripts/evaluate_rag.py --live
Never runs as part of pytest. Does not read or modify the application's database.
"""
import argparse
import asyncio
import os
from pathlib import Path
import tempfile
from ai_agent_pro.config import Agent, AppError
from ai_agent_pro.database import Repository
from ai_agent_pro.rag.embeddings import OpenAIEmbeddings
from ai_agent_pro.rag.service import KnowledgeBase
from ai_agent_pro.safety import configure_logging

CASES = [
    ('deployment.txt', 'Production releases run in London. The staging environment runs in Dublin.',
     ['Where is the production service hosted?', 'Which city hosts staging?']),
    ('database.txt', 'The primary relational database is PostgreSQL 16. Backups are taken every six hours.',
     ['Which SQL database do we use?', 'How often are database backups created?']),
    ('security.txt', 'Service credentials must be rotated every thirty days. Never log access tokens.',
     ['When should service credentials be replaced?', 'May access tokens appear in logs?']),
    ('testing.txt', 'Run pytest before merging changes. A pull request requires two reviewer approvals.',
     ['What test runner is required before merge?', 'How many reviewers must approve a change?']),
    ('recovery.txt', 'The recovery time objective is forty minutes. Restoration exercises take place quarterly.',
     ['What is the target recovery duration?', 'How frequently do we practice restoring backups?']),
]


async def evaluate(repo, embeddings):
    kb = KnowledgeBase(repo, embeddings)
    agent = Agent.create('Retrieval evaluation', '', 'Use supplied evidence.')
    repo.save_agent(agent)
    cid = kb.repository.create_collection('Evaluation')
    kb.repository.assign(agent.id, [cid])
    expected = []
    for filename, text, questions in CASES:
        did = kb.storage.upload(cid, filename, text.encode())
        await kb.indexer.index(did)
        expected.extend((question, did) for question in questions)
    private = kb.repository.create_collection('Unassigned canary')
    secret_id = kb.storage.upload(private, 'canary.txt', b'Production releases run in London. Private canary.')
    await kb.indexer.index(secret_id)
    # Exercise persisted vectors and assignments after reopening the database.
    kb = KnowledgeBase(Repository(repo.path), embeddings)
    hits = 0
    for question, expected_id in expected:
        conversation = repo.new_conversation(agent.id)
        rid = repo.begin_run(conversation, agent, question)
        sources, _ = await kb.retrieval.retrieve(agent.id, question, rid, timeout=60, budget=40000)
        if any(s.document_id == secret_id for s in sources):
            raise AppError('Evaluation failed: cross-collection retrieval leakage.')
        hits += expected_id in [s.document_id for s in sources]
        repo.finish_run(rid, 'completed', 0, 0, 0, 0, [])
    recall = hits / len(expected)
    print(f'Recall@5: {hits}/{len(expected)} ({recall:.0%}); cross-collection leakage: 0')
    return recall >= 0.90


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Authorize paid OpenAI embedding requests for synthetic documents and questions.')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; this evaluation makes paid API requests.')
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        parser.error('Configure OPENAI_API_KEY privately in the environment.')
    root = Path(__file__).resolve().parents[1] / 'data' / 'evaluations'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with tempfile.TemporaryDirectory(dir=root) as directory:
            ok = asyncio.run(evaluate(Repository(Path(directory) / 'evaluation.sqlite3'), OpenAIEmbeddings(key)))
        return 0 if ok else 1
    except AppError as exc:
        print(str(exc))
        return 1
    except Exception:
        print('Evaluation failed; private error details were withheld.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
