"""Unit tests for the answer/grounding layer: confidence bands, coverage
demotion, the relative-cutoff anti-over-citation gate, and refusal."""
import pytest

from app.answer import (
    REFUSAL_TEXT,
    TAU,
    TAU_HIGH,
    _confidence,
    _coverage,
    answer_question,
)
from app.retrieval import ScoredChunk
from app.chunking import Chunk


def test_confidence_bands():
    # Strong score + full coverage -> high.
    assert _confidence(TAU_HIGH + 0.1, 1.0) == "high"
    # Strong score but partial coverage -> demoted to medium.
    assert _confidence(TAU_HIGH + 0.1, 0.2) == "medium"
    # Score between the floor and the high cut point -> medium.
    assert _confidence((TAU + TAU_HIGH) / 2, 1.0) == "medium"


def test_grounded_answer_is_structurally_grounded(retriever):
    res = answer_question("When do access tokens expire?", retriever, top_k=2)
    assert res.confidence in {"high", "medium"}
    assert {c["doc_id"] for c in res.citations} == {"doc_d"}
    assert "60 minutes" in res.answer
    # Every cited doc_id appears tagged in the answer text.
    for c in res.citations:
        assert f"[{c['doc_id']}]" in res.answer


def test_relative_cutoff_blocks_spurious_cross_doc_citation(retriever):
    # "request" collides with "API requests" (doc_b), but that chunk scores far
    # below the refund chunk, so it must not be cited.
    res = answer_question(
        "How long do I have to request a refund?", retriever, top_k=3
    )
    assert {c["doc_id"] for c in res.citations} == {"doc_a"}


def test_partial_overlap_is_not_high_confidence(retriever):
    # Only one of several content terms matches the corpus -> coverage demotes it.
    res = answer_question(
        "How do I configure refund webhooks and notification email settings?",
        retriever,
        top_k=2,
    )
    assert res.confidence == "medium"
    assert {c["doc_id"] for c in res.citations} == {"doc_a"}


def test_out_of_domain_question_refuses(retriever):
    res = answer_question("What is the capital of France?", retriever, top_k=2)
    assert res.confidence == "low"
    assert res.citations == []
    assert res.answer == REFUSAL_TEXT
    # Candidates still reported for observability.
    assert isinstance(res.used_chunks, list)


def test_coverage_full_and_partial():
    chunk = Chunk(chunk_id="doc_x::0", doc_id="doc_x", title="T", text="alpha beta")
    grounding = [ScoredChunk(chunk=chunk, score=0.9)]
    assert _coverage("alpha beta", grounding) == pytest.approx(1.0)
    # "gamma" is not covered -> 1 of 2 content terms covered.
    assert _coverage("alpha gamma", grounding) == pytest.approx(0.5)
