"""Unit tests for normalization and TF-IDF retrieval."""
from app.retrieval import normalize, stem
from app.service import build_retriever


def test_stem_bridges_common_morphology():
    assert stem("refunds") == stem("refund") == "refund"
    assert stem("tokens") == "token"
    assert stem("limits") == "limit"
    assert stem("policies") == "policy"


def test_stem_leaves_double_s_and_short_words():
    assert stem("access") == "access"  # not "acces"
    assert stem("is") == "is"


def test_normalize_drops_stopwords_and_lowercases():
    terms = normalize("How long do I keep my Tokens?")
    assert "how" not in terms and "do" not in terms and "i" not in terms
    assert "token" in terms  # lowercased + stemmed


def test_search_ranks_expected_document_first():
    retriever = build_retriever()
    cases = {
        "How long do I have to request a refund?": "doc_a",
        "What is the API rate limit per key?": "doc_b",
        "How long are application logs retained?": "doc_c",
        "When do access tokens expire?": "doc_d",
    }
    for question, expected_doc in cases.items():
        top = retriever.search(question, top_k=1)[0]
        assert top.chunk.doc_id == expected_doc, question


def test_out_of_domain_query_scores_zero():
    retriever = build_retriever()
    top = retriever.search("What is the capital of France?", top_k=1)[0]
    assert top.score == 0.0


def test_search_respects_top_k_and_is_deterministic():
    retriever = build_retriever()
    first = retriever.search("refund policy", top_k=3)
    second = retriever.search("refund policy", top_k=3)
    assert len(first) == 3
    assert [s.chunk.chunk_id for s in first] == [s.chunk.chunk_id for s in second]
    assert [s.score for s in first] == [s.score for s in second]
