"""Official SDK embeddings adapter; explicit endpoint, timeout and no retries."""
import asyncio
from typing import Protocol
import openai
from ai_agent_pro.config import AppError
from ai_agent_pro.rag.models import EmbeddingBatch, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from ai_agent_pro.rag.vector_store import encode_vectors


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    async def embed(self, texts: list[str], *, timeout: float) -> EmbeddingBatch: ...


class OpenAIEmbeddings:
    model = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, api_key, client_factory=openai.AsyncOpenAI):
        if not isinstance(api_key, str) or not api_key.strip():
            raise AppError('Configure OPENAI_API_KEY before indexing or querying documents.')
        self._key, self._factory = api_key.strip(), client_factory

    async def embed(self, texts, *, timeout):
        if not texts or len(texts) > 32 or any(not isinstance(t, str) or not t.strip() or len(t.encode()) > 6000 for t in texts):
            raise AppError('Embedding input is empty or exceeds request limits.')
        try:
            async with self._factory(api_key=self._key, base_url='https://api.openai.com/v1',
                                    max_retries=0, timeout=timeout) as client:
                response = await asyncio.wait_for(client.embeddings.create(
                    input=texts, model=self.model, dimensions=self.dimensions,
                    encoding_format='float'), timeout=timeout)
            rows = sorted(response.data, key=lambda item: item.index)
            if [r.index for r in rows] != list(range(len(texts))):
                raise AppError('Embedding provider returned invalid batch indexes.')
            vectors = [r.embedding for r in rows]
            encode_vectors(vectors, len(texts), self.dimensions)
            tokens = response.usage.prompt_tokens if response.usage else None
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise AppError('Embedding provider returned invalid usage.')
            return EmbeddingBatch(vectors, tokens)
        except (openai.APITimeoutError, asyncio.TimeoutError):
            raise AppError('Embedding request timed out. No automatic retry was made.') from None
        except openai.AuthenticationError:
            raise AppError('OpenAI rejected the embedding credentials. Check private configuration.') from None
        except openai.RateLimitError:
            raise AppError('OpenAI embedding rate or quota limit reached. Try later.') from None
        except openai.APIConnectionError:
            raise AppError('Could not connect to OpenAI for embeddings.') from None
        except openai.APIStatusError:
            raise AppError('OpenAI rejected the embedding request or is unavailable.') from None
        except AppError:
            raise
        except Exception:
            raise AppError('Embedding response could not be validated.') from None
