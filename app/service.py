"""Composition root: build the retriever from the knowledge base once, so the
API layer and the tests construct the service the same way."""
from __future__ import annotations

from app.chunking import chunk_documents
from app.documents import DOCUMENTS, Document
from app.retrieval import TfidfRetriever


def build_retriever(documents: list[Document] | None = None) -> TfidfRetriever:
    docs = documents if documents is not None else DOCUMENTS
    return TfidfRetriever(chunk_documents(docs))
