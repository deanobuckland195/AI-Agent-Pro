"""Deterministic UTF-8 bounded chunks with source provenance and overlap.

UTF-8 bytes conservatively upper-bound text tokens, as in the existing engine.
No remote tokenizer files or implicit downloads are needed.
"""
from hashlib import sha256
from ai_agent_pro.config import AppError
from ai_agent_pro.rag.models import Chunk, MAX_CHUNKS


def chunk_sections(sections, size=600, overlap=80):
    if type(size) is not int or type(overlap) is not int or not 16 <= size <= 4000 or not 0 <= overlap < size // 2:
        raise AppError('Invalid chunk size or overlap.')
    chunks = []
    for section in sections:
        text, start = section.text, 0
        while start < len(text):
            end, length = start, 0
            while end < len(text) and length + len(text[end].encode('utf-8')) <= size:
                length += len(text[end].encode('utf-8'))
                end += 1
            if end < len(text):
                boundary = max(text.rfind('\n', start, end), text.rfind(' ', start, end))
                if boundary > start + (end - start) // 2:
                    end = boundary + 1
            content = text[start:end]
            first = section.first_line + text.count('\n', 0, start)
            last = first + content.rstrip('\n').count('\n')
            location = f'page {section.page}' if section.page is not None else f'lines {first}–{last}'
            if content.strip():
                chunks.append(Chunk(content, location, sha256(content.encode()).hexdigest()))
            if len(chunks) > MAX_CHUNKS:
                raise AppError('Document exceeds the chunk limit.')
            if end == len(text):
                break
            next_start, length = end, 0
            while next_start > start + 1 and length + len(text[next_start - 1].encode()) <= overlap:
                next_start -= 1
                length += len(text[next_start].encode())
            start = next_start
    if not chunks:
        raise AppError('Document contains no indexable text.')
    return chunks
