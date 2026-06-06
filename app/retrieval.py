"""Retrieval: rank chunks against a question with TF-IDF + cosine similarity.

Implemented from scratch (no scikit-learn) -- the math is small, the dependency
footprint stays tiny, and it starts instantly. Everything sits behind the
``Retriever`` protocol so it can be swapped for embeddings or a vector DB without
touching the answer layer. See docs/adr/0001 for the grounding stance this serves.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from app.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, hand-picked English stopword list. Kept deliberately short: we only
# drop function words that carry no topical signal, never domain terms.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "of",
        "to", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "do", "does", "did", "done", "can",
        "could", "may", "might", "must", "shall", "should", "will", "would",
        "have", "has", "had", "i", "you", "he", "she", "it", "we", "they",
        "this", "that", "these", "those", "my", "your", "our", "their", "its",
        "what", "when", "where", "who", "whom", "which", "why", "how", "whose",
        "per", "after", "once", "within", "above", "not", "no", "there", "here",
        "about", "into", "than", "up", "out", "over", "again", "only", "very",
        "just", "also", "me", "am", "any", "some", "such",
    }
)


def stem(token: str) -> str:
    """Conservative suffix stemmer.

    Deliberately crude -- it bridges the common morphology in this corpus
    (refund/refunds, token/tokens, limit/limits) without pulling in a real
    stemmer. It will occasionally over- or under-conflate (e.g. delete/deletion);
    on this corpus the recall win clearly outweighs the rare miss.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize(text: str) -> list[str]:
    """Lowercase, tokenize, drop stopwords, then stem. Numbers are kept --
    they are highly discriminative here ("30 days", "429", "60 minutes")."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        out.append(stem(raw))
    return out


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    """The retrieval seam. Swap the implementation, keep the answer layer."""

    def search(self, question: str, top_k: int) -> list[ScoredChunk]: ...


class TfidfRetriever:
    """TF-IDF vectors with smoothed IDF, ranked by cosine similarity.

    Each chunk is indexed on its own sentence *plus its document title*, so a
    topical question ("data retention") matches even when it shares no words with
    the specific sentence that answers it.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._chunk_terms = [normalize(f"{c.title} {c.text}") for c in chunks]
        self._build_index()

    def _build_index(self) -> None:
        n = len(self.chunks)
        df: dict[str, int] = {}
        for terms in self._chunk_terms:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        # Smoothed IDF (sklearn-style): log((n+1)/(df+1)) + 1, always positive.
        self.idf: dict[str, float] = {
            term: math.log((n + 1) / (d + 1)) + 1.0 for term, d in df.items()
        }
        self._chunk_vectors = [self._vectorize(terms) for terms in self._chunk_terms]

    def _vectorize(self, terms: list[str]) -> dict[str, float]:
        """TF-IDF weights for in-vocabulary terms, L2-normalized.

        Out-of-vocabulary terms are dropped (they have no document dimension), so
        an unknown word in a question never inflates the query norm and tanks the
        cosine score -- matching how a fitted vectorizer transforms a query.
        """
        tf: dict[str, float] = {}
        for term in terms:
            if term in self.idf:
                tf[term] = tf.get(term, 0.0) + 1.0
        vec = {term: count * self.idf[term] for term, count in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm > 0.0:
            vec = {term: w / norm for term, w in vec.items()}
        return vec

    def search(self, question: str, top_k: int) -> list[ScoredChunk]:
        query_vec = self._vectorize(normalize(question))
        scored = [
            ScoredChunk(chunk=chunk, score=_cosine(query_vec, chunk_vec))
            for chunk, chunk_vec in zip(self.chunks, self._chunk_vectors)
        ]
        # Stable tie-break on chunk_id keeps results deterministic across runs.
        scored.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
        return scored[:top_k]


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Dot product of two L2-normalized sparse vectors == cosine similarity."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())
