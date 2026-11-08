"""Bounded indexing with atomic generations and separately recorded API usage."""
import asyncio
from time import monotonic
import numpy as np
from ai_agent_pro.config import AppError
from ai_agent_pro.rag.chunking import chunk_sections
from ai_agent_pro.rag.extraction import extract
from ai_agent_pro.rag.models import INDEX_TIMEOUT, INDEX_TOKEN_BUDGET, PROCESSING_VERSION
from ai_agent_pro.rag.vector_store import encode_vectors
from ai_agent_pro.telemetry import Recorder


class DocumentIndexer:
    def __init__(self, knowledge_repository, embeddings, observer=None):
        self.knowledge = knowledge_repository
        self.embeddings = embeddings
        self.observer = observer

    async def index(self, document_id):
        jid = self.knowledge.begin_index(document_id)
        started, reserved, usage, known = monotonic(), 0, 0, True
        recorder = Recorder(self.knowledge.repo, job_id=jid, observer=self.observer)
        try:
            document = self.knowledge.document(document_id)
            recorder.start('extraction')
            sections, warning = await asyncio.to_thread(extract, document['kind'], document['original'])
            recorder.complete('extraction')
            recorder.start('chunking')
            chunks = chunk_sections(sections)
            recorder.complete('chunking', completed=len(chunks), outcome='chunks_created')
            # Cache is restricted to this document and exact processing/model version.
            with self.knowledge.repo.connect() as db:
                cache = {r['content_hash']: r['embedding'] for r in db.execute('''SELECT c.content_hash,c.embedding
                    FROM document_chunks c JOIN index_generations g ON g.id=c.generation_id
                    WHERE g.document_id=? AND g.model=? AND g.dimensions=? AND g.version=?''',
                    (document_id, self.embeddings.model, self.embeddings.dimensions, PROCESSING_VERSION))}
            # Corrupt cached vectors must be regenerated, not carried into a fresh index.
            for digest, blob in list(cache.items()):
                if len(blob) != self.embeddings.dimensions * 4:
                    del cache[digest]
                    continue
                vector = np.frombuffer(blob, dtype='<f4')
                if not np.isfinite(vector).all() or not 0.99 <= np.linalg.norm(vector) <= 1.01:
                    del cache[digest]
            missing = list(dict.fromkeys(c.content_hash for c in chunks if c.content_hash not in cache))
            texts = {c.content_hash: c.text for c in chunks}
            batches = (len(missing) + 31) // 32
            recorder.start('embeddings', total=batches)
            for offset in range(0, len(missing), 32):
                hashes = missing[offset:offset + 32]
                batch = [texts[h] for h in hashes]
                reserve = sum(len(t.encode()) for t in batch)
                if reserved + reserve > INDEX_TOKEN_BUDGET:
                    raise AppError('Indexing token budget exceeded.')
                remaining = INDEX_TIMEOUT - (monotonic() - started)
                if remaining <= 0:
                    raise AppError('Indexing deadline reached.')
                self.knowledge.charge_index(jid, reserve)
                result = await asyncio.wait_for(self.embeddings.embed(batch, timeout=remaining), remaining)
                encoded = encode_vectors(result.vectors, len(batch), self.embeddings.dimensions)
                cache.update(zip(hashes, encoded, strict=True))
                reserved += reserve
                if result.input_tokens is None:
                    known = False
                elif type(result.input_tokens) is not int or result.input_tokens < 0 or result.input_tokens > reserve:
                    raise AppError('Reported embedding usage exceeded the local reservation.')
                else:
                    usage += result.input_tokens
                recorder.emit('embeddings', 'completed', ordinal=offset // 32 + 1,
                              completed=offset // 32 + 1, total=batches, outcome='batch_completed')
            if monotonic() - started > INDEX_TIMEOUT:
                raise AppError('Indexing deadline reached.')
            recorder.complete('embeddings', completed=batches, total=batches, outcome='cached' if not missing else 'embedded')
            recorder.start('publish')
            generation = self.knowledge.finish_index(jid, chunks, [cache[c.content_hash] for c in chunks],
                    self.embeddings.model, self.embeddings.dimensions, PROCESSING_VERSION, usage if known else None, warning)
            recorder.complete('publish')
            return generation
        except BaseException as exc:
            error = str(exc) if isinstance(exc, AppError) else (
                'Indexing deadline reached.' if isinstance(exc, asyncio.TimeoutError) else
                'Indexing failed or was cancelled. No automatic retry was made.')
            recorder.fail(error)
            self.knowledge.fail_index(jid, error)
            if isinstance(exc, asyncio.CancelledError) or not isinstance(exc, Exception):
                raise
            raise AppError(error) from None
