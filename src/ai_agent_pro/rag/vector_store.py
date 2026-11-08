"""Exact cosine search over scoped SQLite candidates, replaceable independently."""
import math
import numpy as np
from ai_agent_pro.config import AppError


def normalized(vector, dimensions):
    if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
        raise AppError('Embedding provider returned invalid dimensions.')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vector):
        raise AppError('Embedding provider returned invalid values.')
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if not np.isfinite(array).all() or not math.isfinite(float(norm)) or norm <= 0:
        raise AppError('Embedding provider returned an invalid vector.')
    return (array / norm).astype('<f4')


def encode_vectors(vectors, count, dimensions):
    if len(vectors) != count:
        raise AppError('Embedding provider returned the wrong vector count.')
    return [normalized(v, dimensions).tobytes() for v in vectors]


class SQLiteVectorStore:
    def __init__(self, knowledge_repository):
        self.knowledge = knowledge_repository

    def search(self, agent_id, query, model, dimensions, version, limit=5, threshold=0.25):
        query = normalized(query, dimensions)
        candidates = self.knowledge.candidates(agent_id, model, dimensions, version)
        if len(candidates) > 20000:
            raise AppError('Knowledge base exceeds retrieval limits.')
        ranked = []
        for row in candidates:
            if len(row['embedding']) != dimensions * 4:
                raise AppError('Stored embeddings are invalid. Re-index the affected documents.')
            vector = np.frombuffer(row['embedding'], dtype='<f4')
            if vector.size != dimensions or not np.isfinite(vector).all():
                raise AppError('Stored embeddings are invalid. Re-index the affected documents.')
            norm = np.linalg.norm(vector)
            if not 0.99 <= norm <= 1.01:
                raise AppError('Stored embeddings are invalid. Re-index the affected documents.')
            score = float(np.dot(vector, query))
            if score >= threshold:
                ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], item[1]['document_id'], item[1]['ordinal']))
        selected, hashes = [], set()
        for score, row in ranked:
            if row['content_hash'] not in hashes:
                selected.append((score, row))
                hashes.add(row['content_hash'])
            if len(selected) >= limit:
                break
        return selected
