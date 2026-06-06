"""HTTP request/response schemas. Pydantic does the input validation declaratively
so the contract self-documents via OpenAPI and invalid input returns 422."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AnswerRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The user's question. Answered only from the knowledge base.",
        examples=["How long do I have to request a refund?"],
    )
    top_k: int = Field(
        2,
        ge=1,
        le=3,
        description="Max number of chunks to retrieve. Default 2, max 3.",
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank or whitespace only")
        return value


class Citation(BaseModel):
    doc_id: str
    title: str


class UsedChunk(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    used_chunks: list[UsedChunk]
