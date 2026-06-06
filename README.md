# Document QA Service

A small HTTP API that answers questions over a fixed, in-memory knowledge base
using retrieval-augmented generation — with **citations**, a **confidence**
label, and a hard rule that it **never asserts anything the documents don't say**.

The four seed documents (refund policy, API rate limits, data retention,
authentication) are embedded in `app/documents.py`, so the service is fully
self-contained: no network, no API keys, no external services.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the API (interactive docs at http://127.0.0.1:8000/docs)
uvicorn app.main:app --reload

# Run the tests
pytest
```

### Try it

```bash
# Grounded question
curl -s -X POST localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long do I have to request a refund?"}' | python3 -m json.tool

# Unanswerable question -> honest refusal
curl -s -X POST localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}' | python3 -m json.tool
```

---

## API

### `POST /answer`

**Request**

| field      | type           | required | notes                          |
|------------|----------------|----------|--------------------------------|
| `question` | string         | yes      | must be non-blank              |
| `top_k`    | integer        | no       | default `2`, range `1..3`      |

**Response**

```jsonc
{
  "answer": "Customers may request a refund within 30 days of purchase. [doc_a]",
  "citations": [{ "doc_id": "doc_a", "title": "Refund Policy" }],
  "confidence": "high",                 // high | medium | low
  "used_chunks": [                      // retrieved candidates + scores (observability)
    { "doc_id": "doc_a", "title": "Refund Policy", "chunk_id": "doc_a::0",
      "text": "Customers may request a refund within 30 days of purchase.", "score": 0.6336 },
    { "doc_id": "doc_b", "title": "API Rate Limits", "chunk_id": "doc_b::0",
      "text": "Public API requests are limited to 100 requests per minute per API key.", "score": 0.3473 }
  ]
}
```

Invalid input (`top_k` out of range, blank/missing `question`, wrong type)
returns **`422`** with Pydantic's standard `detail` payload.

`GET /health` returns service status and the document/chunk counts.

#### `citations` vs `used_chunks` — read this

These are intentionally different (the spec's field name "used" vs its
description "retrieved" forced the question):

- **`citations`** is the *grounding* signal: documents that cleared the score
  threshold and actually back the answer. **This is what you show users.**
- **`used_chunks`** is the *observability* signal: every candidate retrieval
  surfaced, with its score — including ones that were rejected. In the example
  above, `doc_b` is a candidate (`0.347`) but is **not** cited, because it's only
  a spurious word collision ("request a refund" vs "API **requests**"). Use this
  to debug and to monitor why the service answered or refused.

---

## How it works

```
question ─▶ retrieve (TF-IDF + cosine over sentence chunks)
         ─▶ ground   (apply thresholds: refuse? which chunks?)
         ─▶ answer   (stitch cited source sentences) ─▶ {answer, citations, confidence, used_chunks}
```

1. **Chunking** (`chunking.py`) — each document is split into one chunk per
   sentence. Tight chunks make for precise answers and fine-grained scoring.
2. **Retrieval** (`retrieval.py`) — hand-rolled **TF-IDF with cosine
   similarity**. Tokens are lowercased, stopword-filtered, and lightly stemmed
   (so `refund`/`refunds`, `token`/`tokens` match). Each chunk is indexed on its
   sentence *plus its document title*, so a topical question matches even when it
   shares no words with the exact answering sentence. It sits behind a
   `Retriever` protocol, so swapping in embeddings or a vector DB touches nothing
   else.
3. **Answering** (`answer.py`) — applies the grounding policy below and composes
   an **extractive** answer: the answer text *is* the retrieved source sentences,
   each tagged with `[doc_id]`. There is no language model, so the answer cannot
   contain a claim that isn't in the knowledge base. (See
   [`docs/adr/0001`](docs/adr/0001-grounding-over-fluency.md) for why.)

### Grounding & safeguards

Two thresholds with two distinct jobs (this split was driven by calibration —
see below):

- **`TAU = 0.15`** — the **refusal floor**. If the best chunk scores below it,
  the service returns the honest "I don't have enough information…" refusal
  rather than guessing. Out-of-domain questions share no vocabulary and score
  `0.0`, so they're cleanly caught.
- **`RELATIVE_CUTOFF = 0.6`** — the **citation gate**. A *supporting* chunk joins
  the answer only if it scores ≥ 60% of the best chunk. This stops a weak,
  off-topic chunk from being welded onto a good answer just because `top_k` asked
  for more. `top_k` is a **ceiling, not a quota**.

### Confidence heuristic

| label    | when                                                                 |
|----------|----------------------------------------------------------------------|
| `low`    | refusal — nothing cleared `TAU`                                      |
| `medium` | answered, but top score is weak (`< TAU_HIGH=0.40`) **or** the chunks cover < 50% of the question's content terms |
| `high`   | strong top score **and** good question coverage                      |

The coverage term is what keeps it honest on partially-answerable questions: a
question with five specific terms where only one matches the corpus gets a
`medium`, not a confident `high`.

---

## Calibration (how the thresholds were chosen)

Thresholds are not guessed. `scripts/calibrate.py` runs 12 grounded and 6
out-of-domain questions and prints the score distributions:

```bash
python -m scripts.calibrate
```

The result is a clean, wide gap:

- **Out-of-domain** questions scored **0.0000** (no shared vocabulary after
  stopword removal).
- **Grounded** questions scored **0.46 – 0.87**.

`TAU = 0.15` sits well above zero-noise yet far below the grounded floor (0.46),
leaving margin for legitimately sparse phrasing.

Calibration also surfaced a real bug: a *single* absolute threshold can't both be
low enough to admit real questions **and** high enough to reject spurious second
chunks — a spurious cross-document match (`0.347`) scored *below* genuine
same-topic corroboration in other queries (`0.44 – 0.48`). That's why the
citation gate is **relative to each query's best score**, not absolute.

---

## Assumptions

- **Closed-world.** The service answers *only* from the four seed documents.
  "I don't know" is a correct, expected outcome, not a failure.
- **Small, static corpus.** Seed data lives in a module and the index is built
  once at startup. For a larger or changing corpus, load documents from
  storage and rebuild/persist the index (nothing downstream assumes the literals).
- **English, lexical questions.** TF-IDF matches words, not meaning, which is
  appropriate for a 12-sentence corpus where questions share vocabulary with
  answers. See limitations.
- **Single process, in-memory.** No persistence or auth; this is a prototype.

## Tradeoffs & limitations

- **Extractive, not generative.** Answers are stitched-together source sentences,
  not fluent prose. This is a deliberate trade for a *guaranteed* no-hallucination
  property and zero external dependencies. The retrieval layer is isolated behind
  a `Retriever` protocol so an LLM generator could be added later — **but only
  one constrained to and verified against the retrieved chunks** (see ADR 0001).
- **Lexical retrieval misses synonyms.** "How long until my data is purged?"
  won't match "retained" / "removal" well. Embeddings would fix this at the cost
  of a heavy dependency — not worth it at this scale.
- **The stemmer is crude.** It bridges plurals and common suffixes but will miss
  pairs like `delete`/`deletion`. Conservative by design; the recall win on this
  corpus outweighs the occasional miss.
- **TF-IDF hand-rolled, not scikit-learn.** Keeps the dependency footprint tiny
  and the algorithm legible; behind the `Retriever` seam it's a drop-in swap.
- **Thresholds tuned on a tiny sample.** Defensible for a prototype; a real
  deployment would calibrate on logged production queries.

## If this were going to production

- **Retrieval quality.** Add embedding-based (dense) retrieval and combine it
  with the lexical score (hybrid search / reciprocal-rank fusion); a real
  stemmer or lemmatizer; query expansion. Re-rank the top candidates.
- **Observability.** The service already emits a structured, no-PII log line per
  request (`top_score`, `grounded`, `confidence`, `latency_ms`). In production
  I'd ship these as metrics: **refusal rate**, **confidence distribution**, and
  **score histograms** are the early-warning signals that the corpus or the
  thresholds have drifted. Trace IDs, and (with consent) sampled
  question/answer logging for offline eval.
- **Evaluation.** A labelled question→expected-doc set in CI to catch retrieval
  regressions; track grounding precision/recall as the corpus changes.
- **Hardening.** Auth, rate limiting, request size limits, and per-tenant
  knowledge bases.

---

## Project layout

```
app/
  documents.py   # seed knowledge base
  chunking.py    # sentence-level chunking
  retrieval.py   # TF-IDF + cosine, behind a Retriever protocol
  answer.py      # grounding policy, confidence, extractive composition
  models.py      # Pydantic request/response schemas (input validation)
  service.py     # composition root (build the retriever)
  main.py        # FastAPI app: POST /answer, GET /health
scripts/
  calibrate.py   # empirical threshold calibration
tests/           # pytest: api, answer, retrieval, chunking
CONTEXT.md       # domain glossary (ubiquitous language)
docs/adr/        # architecture decision records
```

`CONTEXT.md` is the glossary for the domain language used throughout the code
(Knowledge base, Document, Chunk, Grounded, Citation, Refusal, …); `docs/adr/`
records the one decision worth preserving — grounding over fluency.
