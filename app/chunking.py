"""Chunking: split each document into sentence-level chunks, the unit of
retrieval and the raw material of an extractive answer.

The seed documents are written one fact per line, so line + sentence splitting
yields tight, self-contained chunks. A chunk keeps a back-reference to its
parent document's id so citations can roll up to the document level.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.documents import Document

# Split a line into sentences on terminal punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A single retrievable sentence, tagged with its parent document."""

    chunk_id: str  # stable id, e.g. "doc_a::0"
    doc_id: str
    title: str
    text: str


def split_into_sentences(text: str) -> list[str]:
    """Split free text into sentences, respecting line breaks first."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for piece in _SENTENCE_SPLIT.split(line):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def chunk_document(doc: Document) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=sentence,
        )
        for i, sentence in enumerate(split_into_sentences(doc.content))
    ]


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks
