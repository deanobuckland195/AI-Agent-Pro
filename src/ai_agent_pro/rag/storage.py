"""Validated upload storage. Originals live transactionally inside local SQLite."""
from ai_agent_pro.rag.validation import validate_upload


class DocumentStorage:
    def __init__(self, knowledge_repository):
        self.knowledge = knowledge_repository

    def upload(self, collection_id, filename, data):
        filename, kind = validate_upload(filename, data)
        return self.knowledge.add_document(collection_id, filename, kind, data)
