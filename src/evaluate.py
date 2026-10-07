"""
Evaluation Metrics Module
Bilingual Document Q&A (RAG) System

Computes retrieval metrics (Hit@k, MRR) and generation evaluation metrics
(Correctness, Unsupported-Answer Rate, Refusal Accuracy).
"""

from typing import List, Dict, Any, Set, Tuple
import numpy as np


def compute_hit_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Return 1.0 if any gold supporting chunk appears in top-k, else 0.0."""
    top_k_ids = retrieved_ids[:k]
    for cid in top_k_ids:
        if cid in gold_ids:
            return 1.0
    return 0.0


def compute_reciprocal_rank(retrieved_ids: List[str], gold_ids: Set[str]) -> float:
    """
    Compute 1 / rank of first relevant chunk (1-indexed).
    Returns 0.0 if no relevant chunk found.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retriever_fn,
    gold_set: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Any]:
    """
    Evaluate a retriever function on gold questions.
    Only answerable questions with gold supporting chunks are scored.
    """
    answerable_items = [q for q in gold_set if q["answerable"] and q["supporting_chunk_ids"]]
    if not answerable_items:
        raise ValueError("No answerable items with supporting chunk IDs found in gold set.")

    hits = {k: 0.0 for k in k_values}
    mrr_total = 0.0
    per_query_details = []

    for item in answerable_items:
        qid = item["id"]
        q_text = item["question"]
        lang = item["language"]
        gold_ids = set(item["supporting_chunk_ids"])

        # Retrieve top max(k_values)
        max_k = max(k_values)
        retrieved_chunks = retriever_fn(q_text, top_k=max_k)
        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

        # Calculate metrics
        q_hits = {}
        for k in k_values:
            hit = compute_hit_at_k(retrieved_ids, gold_ids, k)
            hits[k] += hit
            q_hits[f"hit@{k}"] = hit

        rr = compute_reciprocal_rank(retrieved_ids, gold_ids)
        mrr_total += rr

        per_query_details.append({
            "id": qid,
            "language": lang,
            "question": q_text,
            "gold_ids": list(gold_ids),
            "retrieved_ids": retrieved_ids,
            "reciprocal_rank": rr,
            **q_hits
        })

    total = len(answerable_items)
    results = {
        "total_evaluated_questions": total,
        "mrr": round(mrr_total / total, 4),
        "per_query": per_query_details
    }
    for k in k_values:
        results[f"hit@{k}"] = round(hits[k] / total, 4)

    # Calculate language breakdowns
    for l in ["ar", "en"]:
        l_items = [d for d in per_query_details if d["language"] == l]
        if l_items:
            l_total = len(l_items)
            results[f"{l}_mrr"] = round(sum(d["reciprocal_rank"] for d in l_items) / l_total, 4)
            for k in k_values:
                results[f"{l}_hit@{k}"] = round(sum(d[f"hit@{k}"] for d in l_items) / l_total, 4)

    return results
