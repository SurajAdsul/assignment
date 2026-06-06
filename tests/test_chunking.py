"""Unit tests for sentence-level chunking."""
from app.documents import DOCUMENTS
from app.chunking import chunk_documents


def test_each_document_splits_into_its_sentences():
    chunks = chunk_documents(DOCUMENTS)
    # Every seed document has 3 sentences -> 12 chunks total.
    assert len(chunks) == 12


def test_chunk_ids_are_unique_and_namespaced():
    chunks = chunk_documents(DOCUMENTS)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith(f"{c.doc_id}::") for c in chunks)


def test_chunk_text_is_non_empty_and_carries_title():
    chunks = chunk_documents(DOCUMENTS)
    assert all(c.text.strip() for c in chunks)
    doc_a_titles = {c.title for c in chunks if c.doc_id == "doc_a"}
    assert doc_a_titles == {"Refund Policy"}
