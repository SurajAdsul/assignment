"""FastAPI application exposing the question-answering endpoint."""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI

from app.answer import answer_question
from app.documents import DOCUMENTS
from app.models import AnswerRequest, AnswerResponse, Citation, UsedChunk
from app.service import build_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("docqa")

app = FastAPI(
    title="Document QA Service",
    version="1.0.0",
    description="Answers questions using only a fixed knowledge base, with citations.",
)

# Build the index once at startup; the knowledge base is fixed and in-memory.
_retriever = build_retriever()
logger.info("retriever ready: %d documents, %d chunks", len(DOCUMENTS), len(_retriever.chunks))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(DOCUMENTS),
        "chunks": len(_retriever.chunks),
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    start = time.perf_counter()
    result = answer_question(request.question, _retriever, request.top_k)
    latency_ms = (time.perf_counter() - start) * 1000.0

    top_score = result.used_chunks[0]["score"] if result.used_chunks else 0.0
    # Structured, no-PII log line: enough to monitor grounding/refusal rates and
    # latency in production without recording the raw question text.
    logger.info(
        "answer q_len=%d top_k=%d top_score=%.4f grounded=%s confidence=%s latency_ms=%.1f",
        len(request.question),
        request.top_k,
        top_score,
        bool(result.citations),
        result.confidence,
        latency_ms,
    )

    return AnswerResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        confidence=result.confidence,
        used_chunks=[UsedChunk(**u) for u in result.used_chunks],
    )
