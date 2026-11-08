"""Bounded RAG data contracts and versioned processing configuration."""
from dataclasses import dataclass

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_PAGES = 200
MAX_TEXT_BYTES = 1_000_000
MAX_CHUNKS = 2500
MAX_TOTAL_CHUNKS = 20000
INDEX_TIMEOUT = 180
INDEX_TOKEN_BUDGET = 1_100_000
PROCESSING_VERSION = 'pdf-text-v1/utf8-600-overlap-80-v1'
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSIONS = 1536


@dataclass(frozen=True)
class Section:
    text: str
    page: int | None = None
    first_line: int = 1


@dataclass(frozen=True)
class Chunk:
    text: str
    location: str
    content_hash: str


@dataclass(frozen=True)
class Source:
    label: str
    document_id: str
    generation_id: str
    filename: str
    location: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    input_tokens: int | None
