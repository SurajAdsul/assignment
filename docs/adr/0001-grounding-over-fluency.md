# Grounding over fluency: extractive answers, closed-world refusal, no LLM

We answer questions by stitching together verbatim sentences retrieved from the knowledge base (each tagged with its `doc_id`) rather than generating prose with an LLM, and we refuse ("not enough information") whenever no retrieved chunk clears the score threshold. We chose this because the product requirement is explicit that the service must never make unsupported claims, and an extractive, closed-world design makes hallucination structurally impossible — at the cost of less fluent answers and no ability to answer beyond the documents.

## Considered options

- **LLM with a grounding prompt** — fluent, real RAG, but reintroduces hallucination risk and adds a network/API-key dependency. Rejected for a prototype whose headline requirement is "no unsupported claims."
- **Extractive + closed-world refusal (chosen)** — the answer is its own evidence; grounding is guaranteed by construction and the whole thing runs locally and deterministically.

## Consequences

- **No general-knowledge fallback.** Do not "improve" the service by bolting an LLM onto the refusal path to answer out-of-domain questions — that silently breaks the core guarantee. If an LLM is ever added, it must be constrained to the retrieved chunks and verified against them, and this ADR should be superseded.
- Answers read as terse, quoted sentences, not conversational prose. That is the intended trade-off.
