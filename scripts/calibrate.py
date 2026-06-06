"""Calibrate the retrieval thresholds (TAU, TAU_HIGH) empirically.

Prints the top-chunk cosine score for a batch of grounded questions (which
should clear the floor) and out-of-domain questions (which should fall below
it). Pick TAU in the gap between the two distributions, and TAU_HIGH where
clearly-strong matches sit. Run:  python -m scripts.calibrate
"""
from __future__ import annotations

from app.service import build_retriever

GROUNDED = [
    "How long do I have to request a refund?",
    "Are refunds available for compute usage charges?",
    "What does an enterprise contract follow?",
    "What is the API rate limit per key?",
    "What happens to burst traffic above the rate limit?",
    "Can enterprise customers get higher rate limits?",
    "How long are application logs retained?",
    "How long are user uploaded files kept?",
    "What happens to my files after account deletion?",
    "What kind of token does API access require?",
    "When do access tokens expire?",
    "Can I exchange a refresh token for a new access token?",
]

OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "Who won the world cup in 2018?",
    "How do I bake sourdough bread?",
    "What is the meaning of life?",
    "What is the weather tomorrow?",
    "How tall is Mount Everest?",
]


def main() -> None:
    retriever = build_retriever()

    def top(q: str) -> float:
        hits = retriever.search(q, top_k=1)
        return hits[0].score if hits else 0.0

    grounded_scores = [(q, top(q)) for q in GROUNDED]
    ood_scores = [(q, top(q)) for q in OUT_OF_DOMAIN]

    print("=== GROUNDED (should clear TAU) ===")
    for q, s in sorted(grounded_scores, key=lambda x: x[1]):
        print(f"  {s:.4f}  {q}")
    print("\n=== OUT-OF-DOMAIN (should fall below TAU) ===")
    for q, s in sorted(ood_scores, key=lambda x: x[1], reverse=True):
        print(f"  {s:.4f}  {q}")

    g_min = min(s for _, s in grounded_scores)
    o_max = max(s for _, s in ood_scores)
    g_sorted = sorted(s for _, s in grounded_scores)
    median = g_sorted[len(g_sorted) // 2]
    print("\n=== SUMMARY ===")
    print(f"  min grounded score      = {g_min:.4f}")
    print(f"  max out-of-domain score = {o_max:.4f}")
    print(f"  gap                     = [{o_max:.4f}, {g_min:.4f}]")
    print(f"  median grounded score   = {median:.4f}")
    print(f"  suggested TAU           = {(o_max + g_min) / 2:.4f}  (midpoint of gap)")
    print(f"  suggested TAU_HIGH ~    = {median:.4f}  (median grounded)")


if __name__ == "__main__":
    main()
