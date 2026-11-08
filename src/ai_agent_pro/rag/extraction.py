"""Bounded extraction with PDF parsing outside the Streamlit process."""
import json
import os
from pathlib import Path
import subprocess
import sys
from ai_agent_pro.config import AppError
from ai_agent_pro.rag.models import MAX_TEXT_BYTES, Section
from ai_agent_pro.safety import redact, validate_text


def extract(kind, data):
    if kind == 'txt':
        text = data.decode('utf-8-sig')
        if len(text.encode('utf-8')) > MAX_TEXT_BYTES:
            raise AppError('Extracted document exceeds the 1 MB text limit.')
        validate_text(text, 'Document text', MAX_TEXT_BYTES)
        return [Section(redact(text))], None
    if kind != 'pdf':
        raise AppError('Unsupported document format.')
    # Whitelist the child environment: API keys and user configuration are not inherited.
    env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR') if k in os.environ}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        result = subprocess.run([sys.executable, '-m', 'ai_agent_pro.rag.extraction_worker'],
                                input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=20, env=env, check=False)
        if result.returncode or len(result.stdout) > MAX_TEXT_BYTES * 8:
            raise AppError('PDF extraction exceeded resource limits or failed.')
        body = json.loads(result.stdout)
        if 'error' in body:
            raise AppError('PDF is invalid, encrypted, exceeds extraction limits, or requires OCR.')
        sections = [Section(redact(validate_text(s['text'], 'Extracted text', MAX_TEXT_BYTES)), s['page'])
                    for s in body['sections']]
        if not sections or sum(len(s.text.encode('utf-8')) for s in sections) > MAX_TEXT_BYTES:
            raise AppError('PDF contains no usable text or exceeds extraction limits.')
        return sections, body.get('warning')
    except subprocess.TimeoutExpired:
        raise AppError('PDF extraction timed out; the parser was terminated.') from None
    except (OSError, ValueError, KeyError, TypeError):
        raise AppError('PDF could not be extracted safely.') from None
