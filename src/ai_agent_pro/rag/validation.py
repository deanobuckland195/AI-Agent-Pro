"""Validate names and bytes independently of the upload widget."""
from pathlib import PurePosixPath
from ai_agent_pro.config import AppError
from ai_agent_pro.safety import validate_text
from ai_agent_pro.rag.models import MAX_FILE_BYTES


def validate_upload(filename, data):
    filename = validate_text(filename, 'Filename', 180)
    if any(ord(c) < 32 or ord(c) == 127 for c in filename):
        raise AppError('Filename contains unsupported control characters.')
    if filename in ('.', '..') or any(c in filename for c in '/\\:'):
        raise AppError('Use a filename without directories or drive prefixes.')
    if any(ord(c) in range(0x202a, 0x202f) or ord(c) in range(0x2066, 0x206a) for c in filename):
        raise AppError('Filename contains unsupported direction controls.')
    kind = PurePosixPath(filename).suffix.lower()
    if kind not in ('.pdf', '.txt'):
        raise AppError('Only PDF and UTF-8 text files are supported.')
    if not isinstance(data, bytes) or not data or len(data) > MAX_FILE_BYTES:
        raise AppError('Upload a nonempty file no larger than 10 MiB.')
    if kind == '.pdf' and not data.startswith(b'%PDF-'):
        raise AppError('The file does not have a valid PDF signature.')
    if kind == '.txt':
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            raise AppError('Text files must use UTF-8 encoding.') from None
        validate_text(text, 'Document text', MAX_FILE_BYTES)
        if text.lstrip().startswith('%PDF-'):
            raise AppError('PDF content must use the .pdf extension.')
    return filename, kind[1:]
