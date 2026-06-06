# Document QA Service

A service that answers user questions using only a fixed set of internal documents, returning answers with citations and never asserting claims the documents don't support.

## Language

**Knowledge base**:
The fixed, closed set of seed documents the service is allowed to answer from. Closed-world — anything not in it is treated as unknown.
_Avoid_: corpus, dataset, index

**Document**:
One seed item in the knowledge base, identified by a stable `doc_id` (e.g. `doc_a`) and a title. The unit of citation — answers cite documents, not chunks.
_Avoid_: file, record, article

**Chunk**:
A single retrievable sentence extracted from a document; it carries its parent document's id. The unit of retrieval and the raw material of an extractive answer.
_Avoid_: passage, fragment, segment, snippet

**Answer**:
The service's response to a question, composed only of verbatim sentences drawn from the knowledge base. An answer never contains a statement absent from its sources.
_Avoid_: response, completion, generation

**Grounded**:
Property of an answer in which every statement traces back to a retrieved source. An ungrounded claim is a defect, not a stylistic choice.
_Avoid_: supported, sourced, factual

**Citation**:
A reference, by `doc_id`, to a document that cleared the score threshold and backs part of the answer. The authoritative "sources" signal; empty on a refusal.
_Avoid_: source, reference, footnote

**Used chunk**:
An entry in `used_chunks`: a retrieved candidate chunk reported with its retrieval score for observability — listed whether or not it ended up grounding the answer.
_Avoid_: result, hit, match

**Refusal**:
The response when no chunk clears the score threshold: an honest "not enough information" answer, empty citations, low confidence, and the rejected candidates still shown in `used_chunks`.
_Avoid_: fallback, error, miss

**Confidence**:
The service's self-assessed reliability of an answer — `high`, `medium`, or `low` — reflecting how strongly retrieval supports it and how much of the question it covers. A refusal is always `low`.
_Avoid_: score, certainty, probability
