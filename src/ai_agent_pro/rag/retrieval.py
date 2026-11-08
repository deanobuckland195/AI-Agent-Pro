"""Agent-scoped retrieval with bounded query embeddings and evidence."""
import asyncio
from ai_agent_pro.config import AppError
from ai_agent_pro.rag.models import Source, PROCESSING_VERSION
from ai_agent_pro.rag.vector_store import SQLiteVectorStore, encode_vectors


class RetrievalService:
    def __init__(self, knowledge_repository, embeddings, vector_store=None):
        self.knowledge = knowledge_repository
        self.embeddings = embeddings
        self.vectors = vector_store or SQLiteVectorStore(knowledge_repository)

    async def retrieve(self, agent_id, question, run_id, *, timeout, budget):
        if not self.knowledge.assigned(agent_id):
            raise AppError('Assign a document collection to this agent before using document mode.')
        reserve = len(question.encode())
        if reserve > 6000 or reserve + 1024 > budget:
            raise AppError('Question exceeds the retrieval input or task budget. Shorten it or raise the budget.')
        # No API request is needed when there are no compatible ready documents.
        candidates = self.knowledge.candidates(agent_id, self.embeddings.model, self.embeddings.dimensions, PROCESSING_VERSION)
        if not candidates:
            return [], 0
        self.knowledge.charge_query(run_id, reserve)
        result = await asyncio.wait_for(self.embeddings.embed([question], timeout=timeout), timeout)
        encode_vectors(result.vectors, 1, self.embeddings.dimensions)
        tokens = result.input_tokens
        if tokens is not None and (type(tokens) is not int or not 0 <= tokens <= reserve):
            raise AppError('Reported embedding usage exceeded the local reservation.')
        self.knowledge.query_usage(run_id, tokens)
        ranked = self.vectors.search(agent_id, result.vectors[0], self.embeddings.model,
                                    self.embeddings.dimensions, PROCESSING_VERSION)
        sources = [Source(f'S{i}', row['document_id'], row['generation_id'], row['filename'],
                          row['location'], row['text'], score) for i, (score, row) in enumerate(ranked, 1)]
        return sources, tokens if tokens is not None else reserve
