"""The knowledge base: the fixed, closed set of documents the service may answer
from. Seed data is embedded here so the service is fully self-contained.

If this ever grows beyond a handful of documents, load it from disk/object
storage instead of a module constant -- nothing downstream assumes these literals.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One seed item in the knowledge base. The unit of citation."""

    doc_id: str
    title: str
    content: str


DOCUMENTS: list[Document] = [
    Document(
        doc_id="doc_a",
        title="Refund Policy",
        content=(
            "Customers may request a refund within 30 days of purchase.\n"
            "Refunds are not available for usage-based charges once compute has been consumed.\n"
            "Enterprise contracts follow the signed agreement terms."
        ),
    ),
    Document(
        doc_id="doc_b",
        title="API Rate Limits",
        content=(
            "Public API requests are limited to 100 requests per minute per API key.\n"
            "Burst traffic above the limit returns HTTP 429.\n"
            "Rate limits may be increased for enterprise customers after review."
        ),
    ),
    Document(
        doc_id="doc_c",
        title="Data Retention",
        content=(
            "Application logs are retained for 14 days.\n"
            "User-uploaded files are retained until deleted by the customer.\n"
            "Account deletion triggers permanent file removal within 7 days, "
            "except where legal retention is required."
        ),
    ),
    Document(
        doc_id="doc_d",
        title="Authentication",
        content=(
            "API access requires a bearer token.\n"
            "Tokens expire after 60 minutes.\n"
            "Refresh tokens may be exchanged for new access tokens."
        ),
    ),
]
