"""
Retrieval Evaluation Script
Bilingual Document Q&A (RAG) System

Compares Baseline Dense Retriever vs. Improved Cross-Encoder Reranker
on the fixed evaluation/gold.jsonl dataset.
Calculates Hit@1, Hit@3, Hit@5, and MRR.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieve import BaselineRetriever
from src.rerank import ImprovedRetriever, Reranker
from src.evaluate import evaluate_retrieval


def load_gold_data(path: str = "evaluation/gold.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def main():
    print("=" * 70)
    print("BILINGUAL RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    gold_path = os.path.join("evaluation", "gold.jsonl")
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold evaluation file not found at {gold_path}")

    gold_questions = load_gold_data(gold_path)
    ans_count = len([q for q in gold_questions if q["answerable"]])
    unans_count = len([q for q in gold_questions if not q["answerable"]])
    print(f"Loaded {len(gold_questions)} questions from {gold_path}")
    print(f"  - Answerable: {ans_count}")
    print(f"  - Unanswerable: {unans_count}")

    # 1. Initialize Baseline
    print("\n--- Initializing Baseline Dense Retriever (BGE-M3 + FAISS) ---")
    t0 = time.time()
    baseline = BaselineRetriever()
    print(f"Baseline ready in {time.time() - t0:.2f}s")

    # 2. Evaluate Baseline
    print("Evaluating Baseline Retrieval...")
    baseline_eval = evaluate_retrieval(
        retriever_fn=lambda q, top_k: baseline.retrieve(q, top_k=top_k),
        gold_set=gold_questions,
        k_values=[1, 3, 5]
    )

    # 3. Initialize Improved (Reranker)
    print("\n--- Initializing Improved Retriever (Dense top-20 + BGE-Reranker top-5) ---")
    t1 = time.time()
    improved = ImprovedRetriever(baseline_retriever=baseline)
    print(f"Reranker ready in {time.time() - t1:.2f}s")

    # 4. Evaluate Improved
    print("Evaluating Improved Retrieval...")
    improved_eval = evaluate_retrieval(
        retriever_fn=lambda q, top_k: improved.retrieve(q, candidate_k=20, top_k=top_k),
        gold_set=gold_questions,
        k_values=[1, 3, 5]
    )

    # 5. Format Results Table
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'System':<25} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8}")
    print("-" * 70)
    print(f"{'Dense baseline':<25} | {baseline_eval['hit@1']:<8.4f} | {baseline_eval['hit@3']:<8.4f} | {baseline_eval['hit@5']:<8.4f} | {baseline_eval['mrr']:<8.4f}")
    print(f"{'Dense + reranker':<25} | {improved_eval['hit@1']:<8.4f} | {improved_eval['hit@3']:<8.4f} | {improved_eval['hit@5']:<8.4f} | {improved_eval['mrr']:<8.4f}")
    print("-" * 70)

    print("\nArabic Subset:")
    print(f"{'Dense baseline (AR)':<25} | {baseline_eval.get('ar_hit@1', 0):<8.4f} | {baseline_eval.get('ar_hit@3', 0):<8.4f} | {baseline_eval.get('ar_hit@5', 0):<8.4f} | {baseline_eval.get('ar_mrr', 0):<8.4f}")
    print(f"{'Dense + reranker (AR)':<25} | {improved_eval.get('ar_hit@1', 0):<8.4f} | {improved_eval.get('ar_hit@3', 0):<8.4f} | {improved_eval.get('ar_hit@5', 0):<8.4f} | {improved_eval.get('ar_mrr', 0):<8.4f}")

    print("\nEnglish Subset:")
    print(f"{'Dense baseline (EN)':<25} | {baseline_eval.get('en_hit@1', 0):<8.4f} | {baseline_eval.get('en_hit@3', 0):<8.4f} | {baseline_eval.get('en_hit@5', 0):<8.4f} | {baseline_eval.get('en_mrr', 0):<8.4f}")
    print(f"{'Dense + reranker (EN)':<25} | {improved_eval.get('en_hit@1', 0):<8.4f} | {improved_eval.get('en_hit@3', 0):<8.4f} | {improved_eval.get('en_hit@5', 0):<8.4f} | {improved_eval.get('en_mrr', 0):<8.4f}")

    # Save results to json
    results_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "evaluation/gold.jsonl",
        "total_answerable_evaluated": baseline_eval["total_evaluated_questions"],
        "baseline": {
            "hit@1": baseline_eval["hit@1"],
            "hit@3": baseline_eval["hit@3"],
            "hit@5": baseline_eval["hit@5"],
            "mrr": baseline_eval["mrr"],
            "ar_mrr": baseline_eval.get("ar_mrr"),
            "en_mrr": baseline_eval.get("en_mrr"),
            "per_query": baseline_eval["per_query"]
        },
        "improved": {
            "hit@1": improved_eval["hit@1"],
            "hit@3": improved_eval["hit@3"],
            "hit@5": improved_eval["hit@5"],
            "mrr": improved_eval["mrr"],
            "ar_mrr": improved_eval.get("ar_mrr"),
            "en_mrr": improved_eval.get("en_mrr"),
            "per_query": improved_eval["per_query"]
        }
    }

    results_file = os.path.join("evaluation", "retrieval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed retrieval results saved to {results_file}")


if __name__ == "__main__":
    main()
