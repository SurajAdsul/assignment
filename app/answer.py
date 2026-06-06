"""Answering: turn retrieved chunks into a grounded answer, citations, and a
confidence label -- or a refusal when nothing clears the bar.

This module owns the anti-hallucination policy (see docs/adr/0001):
  * The best chunk must clear the absolute score floor TAU; if it doesn't, the
    service refuses instead of guessing.
  * The answer is composed only from the best chunk plus any *supporting* chunks
    that score within RELATIVE_CUTOFF of it -- so a spurious cross-document word
    collision (e.g. "request a refund" vs "API requests") never gets cited just
    because top_k asked for a second chunk. top_k is a ceiling, not a quota.
  * Each used sentence is tagged inline with its doc_id.

Why two gates instead of one: calibration showed a single absolute threshold
can't both (a) stay low enough to admit sparsely-phrased real questions and
(b) stay high enough to reject spurious second chunks -- the spurious matches
overlap the genuine ones in absolute terms but are far below their own query's
best score. So TAU governs "answer at all?" and RELATIVE_CUTOFF governs "which
supporting chunks join?". See scripts/calibrate.py and the README.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval import Retriever, ScoredChunk, normalize

# --- Calibrated thresholds (see scripts/calibrate.py output in the README) ---
# Calibration (12 grounded vs 6 out-of-domain questions): out-of-domain queries
# score exactly 0.0 (no shared vocabulary), grounded queries score 0.46-0.87.
# Score floor: chunks scoring below this are ignored; if none clear it, we refuse.
# 0.15 sits well above zero-overlap noise yet far below the grounded floor (0.46),
# leaving margin for legitimate but sparsely-phrased questions.
TAU: float = 0.15
# At/above this top score the answer is "high" confidence (subject to coverage).
# 0.40 keeps every full grounded match high; weak/partial lexical overlap
# (scores in [0.15, 0.40)) lands at medium.
TAU_HIGH: float = 0.40
# A supporting chunk joins the answer only if it scores at least this fraction of
# the best chunk's score. Separates genuine same-topic corroboration (close to the
# top score) from spurious cross-document single-word matches (far below it).
RELATIVE_CUTOFF: float = 0.6
# Fraction of the question's content terms a high-confidence answer must cover;
# below this we demote high -> medium (honest about partially-answered questions).
COVERAGE_MIN: float = 0.5

REFUSAL_TEXT = (
    "I don't have enough information in the knowledge base to answer that."
)


@dataclass(frozen=True)
class AnswerResult:
    """Domain result, independent of the HTTP/JSON layer."""

    answer: str
    citations: list[dict]  # [{doc_id, title}], deduped per document
    confidence: str  # "high" | "medium" | "low"
    used_chunks: list[dict]  # [{doc_id, title, chunk_id, text, score}], all candidates


def _serialize(scored: ScoredChunk) -> dict:
    c = scored.chunk
    return {
        "doc_id": c.doc_id,
        "title": c.title,
        "chunk_id": c.chunk_id,
        "text": c.text,
        "score": round(scored.score, 4),
    }


def _coverage(question: str, grounding: list[ScoredChunk]) -> float:
    """Fraction of the question's distinct content terms that appear in the
    chunks backing the answer. Out-of-vocabulary question terms count against
    coverage (they cannot be covered), which is what we want."""
    q_terms = set(normalize(question))
    if not q_terms:
        return 1.0
    covered: set[str] = set()
    for sc in grounding:
        covered |= set(normalize(f"{sc.chunk.title} {sc.chunk.text}"))
    return len(q_terms & covered) / len(q_terms)


def _confidence(top_score: float, coverage: float) -> str:
    # Reaches here only when at least one chunk cleared TAU, so never "low":
    # "low" is reserved for refusals, where the service declined to answer.
    if top_score >= TAU_HIGH:
        return "high" if coverage >= COVERAGE_MIN else "medium"
    return "medium"


def answer_question(
    question: str,
    retriever: Retriever,
    top_k: int = 2,
    *,
    tau: float = TAU,
) -> AnswerResult:
    retrieved = retriever.search(question, top_k)
    used_chunks = [_serialize(sc) for sc in retrieved]

    if not retrieved or retrieved[0].score < tau:
        # Refusal: rejected candidates stay in used_chunks for observability.
        return AnswerResult(
            answer=REFUSAL_TEXT,
            citations=[],
            confidence="low",
            used_chunks=used_chunks,
        )

    # The best chunk always grounds the answer; supporting chunks must be both
    # above the floor and within RELATIVE_CUTOFF of the best chunk's score.
    top_score = retrieved[0].score
    cutoff = max(tau, RELATIVE_CUTOFF * top_score)
    grounding = [sc for sc in retrieved if sc.score >= cutoff]

    answer = " ".join(f"{sc.chunk.text} [{sc.chunk.doc_id}]" for sc in grounding)

    citations: list[dict] = []
    seen: set[str] = set()
    for sc in grounding:  # grounding is sorted by score desc
        if sc.chunk.doc_id not in seen:
            seen.add(sc.chunk.doc_id)
            citations.append({"doc_id": sc.chunk.doc_id, "title": sc.chunk.title})

    confidence = _confidence(top_score, _coverage(question, grounding))
    return AnswerResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        used_chunks=used_chunks,
    )
