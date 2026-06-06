"""API-level tests for POST /answer via FastAPI's TestClient.

Covers the three required cases (grounded answer, unanswerable question, invalid
input) plus the contract guarantees: citation/used_chunk shape, top_k as a
ceiling, and the no-over-citation regression.
"""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents"] == 4
    assert body["chunks"] == 12


# --- Required: a grounded answer -------------------------------------------------

def test_grounded_answer_refund(client):
    resp = client.post(
        "/answer", json={"question": "How long do I have to request a refund?"}
    )
    assert resp.status_code == 200
    body = resp.json()

    assert "30 days" in body["answer"]
    assert "[doc_a]" in body["answer"]
    assert body["confidence"] in {"high", "medium"}

    cited = {c["doc_id"] for c in body["citations"]}
    assert cited == {"doc_a"}  # exactly the refund doc, nothing spurious
    assert all(c["title"] for c in body["citations"])  # titles populated

    assert body["used_chunks"], "used_chunks must list retrieved candidates"
    for uc in body["used_chunks"]:
        assert {"doc_id", "title", "chunk_id", "text", "score"} <= uc.keys()


def test_grounded_answer_rate_limit(client):
    resp = client.post(
        "/answer", json={"question": "What happens to traffic above the API rate limit?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "429" in body["answer"]
    assert {c["doc_id"] for c in body["citations"]} == {"doc_b"}


def test_citations_are_a_subset_of_used_chunks(client):
    resp = client.post("/answer", json={"question": "When do access tokens expire?"})
    body = resp.json()
    used_ids = {uc["doc_id"] for uc in body["used_chunks"]}
    cited_ids = {c["doc_id"] for c in body["citations"]}
    assert cited_ids <= used_ids


# --- Required: an unanswerable question -----------------------------------------

def test_unanswerable_question_refuses(client):
    resp = client.post("/answer", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["citations"] == []
    assert body["confidence"] == "low"
    assert "enough information" in body["answer"].lower()
    # Rejected candidates are still surfaced for observability.
    assert isinstance(body["used_chunks"], list)


# --- Required: invalid input -----------------------------------------------------

def test_top_k_too_large_is_rejected(client):
    resp = client.post("/answer", json={"question": "anything", "top_k": 5})
    assert resp.status_code == 422


def test_top_k_zero_is_rejected(client):
    resp = client.post("/answer", json={"question": "anything", "top_k": 0})
    assert resp.status_code == 422


def test_blank_question_is_rejected(client):
    resp = client.post("/answer", json={"question": "   "})
    assert resp.status_code == 422


def test_missing_question_is_rejected(client):
    resp = client.post("/answer", json={"top_k": 2})
    assert resp.status_code == 422


def test_wrong_type_top_k_is_rejected(client):
    resp = client.post("/answer", json={"question": "anything", "top_k": "two"})
    assert resp.status_code == 422


# --- Contract: top_k is a ceiling ------------------------------------------------

def test_top_k_defaults_to_two(client):
    resp = client.post("/answer", json={"question": "When do access tokens expire?"})
    assert resp.status_code == 200
    assert len(resp.json()["used_chunks"]) <= 2


def test_top_k_one_returns_single_candidate(client):
    resp = client.post(
        "/answer", json={"question": "When do access tokens expire?", "top_k": 1}
    )
    assert resp.status_code == 200
    assert len(resp.json()["used_chunks"]) == 1
