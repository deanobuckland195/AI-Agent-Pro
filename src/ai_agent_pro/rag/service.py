"""Small composition root shared by UI, engine and tests."""
from ai_agent_pro.rag.repository import KnowledgeRepository
from ai_agent_pro.rag.storage import DocumentStorage
from ai_agent_pro.rag.indexing import DocumentIndexer
from ai_agent_pro.rag.retrieval import RetrievalService


class KnowledgeBase:
    def __init__(self, repository, embeddings=None):
        self.repository = KnowledgeRepository(repository)
        self.storage = DocumentStorage(self.repository)
        self.indexer = DocumentIndexer(self.repository, embeddings) if embeddings else None
        self.retrieval = RetrievalService(self.repository, embeddings) if embeddings else None
