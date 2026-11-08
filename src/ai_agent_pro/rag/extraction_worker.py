"""Resource-limited PDF parser. No credentials are passed to this process."""
import io
import json
import logging
import os
import sys
import threading
import time
from ai_agent_pro.rag.models import MAX_FILE_BYTES, MAX_PAGES, MAX_TEXT_BYTES


def parse_pdf(data):
    from pypdf import PdfReader
    logging.disable(logging.CRITICAL)
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise ValueError('Encrypted PDFs are unsupported.')
    if len(reader.pages) > MAX_PAGES:
        raise ValueError('PDF exceeds the 200-page limit.')
    sections, total, empty = [], 0, 0
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        total += len(text.encode('utf-8'))
        if total > MAX_TEXT_BYTES:
            raise ValueError('Extracted document exceeds the text limit.')
        if text.strip():
            sections.append({'text': text, 'page': number})
        else:
            empty += 1
    if not sections:
        raise ValueError('No extractable text. Scanned PDFs require OCR, which is not supported.')
    return {'sections': sections, 'warning': f'{empty} pages had no extractable text; OCR may be required.' if empty else None}


def main():
    # This is resource isolation, not an OS security sandbox. Never execute PDF actions.
    try:
        import resource
        import psutil
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        if sys.platform != 'darwin':
            resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
        # macOS rejects RLIMIT_AS. Monitor RSS on all supported platforms too.
        def watch_memory():
            process = psutil.Process()
            while True:
                if process.memory_info().rss > 512 * 1024**2:
                    os._exit(70)
                time.sleep(0.02)
        threading.Thread(target=watch_memory, daemon=True).start()
        data = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError('File exceeds the upload limit.')
        result = parse_pdf(data)
    except ValueError:
        # Only fixed application errors are surfaced by the parent; parser errors stay private.
        result = {'error': 'PDF is invalid, encrypted, too large, or requires OCR.'}
    except Exception:
        result = {'error': 'PDF could not be extracted safely.'}
    sys.stdout.write(json.dumps(result))


if __name__ == '__main__':
    main()
