import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieve import BaselineRetriever
from src.rerank import ImprovedRetriever

def main():
    base = BaselineRetriever()
    improved = ImprovedRetriever(base)

    with open("evaluation/gold.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    print("=== ANSWERABLE QUERIES SCORES ===")
    ans_dense, ans_rerank = [], []
    for q in questions:
        if q["answerable"]:
            dense_res = base.retrieve(q["question"], top_k=1)
            rerank_res = improved.retrieve(q["question"], candidate_k=20, top_k=1)
            d_top = dense_res[0]["score"] if dense_res else 0.0
            r_top = rerank_res[0]["reranker_score"] if rerank_res else -99
            ans_dense.append(d_top)
            ans_rerank.append(r_top)
            print(f"{q['id']} ({q['language']}): dense={d_top:.4f}, rerank={r_top:.4f}")

    print("\n=== UNANSWERABLE QUERIES SCORES ===")
    unans_dense, unans_rerank = [], []
    for q in questions:
        if not q["answerable"]:
            dense_res = base.retrieve(q["question"], top_k=1)
            rerank_res = improved.retrieve(q["question"], candidate_k=20, top_k=1)
            d_top = dense_res[0]["score"] if dense_res else 0.0
            r_top = rerank_res[0]["reranker_score"] if rerank_res else -99
            unans_dense.append(d_top)
            unans_rerank.append(r_top)
            print(f"{q['id']} ({q['language']}): dense={d_top:.4f}, rerank={r_top:.4f}")

    print("\nSummary Statistics:")
    print(f"Answerable dense mean: {sum(ans_dense)/len(ans_dense):.4f}, min: {min(ans_dense):.4f}, max: {max(ans_dense):.4f}")
    print(f"Answerable rerank mean: {sum(ans_rerank)/len(ans_rerank):.4f}, min: {min(ans_rerank):.4f}, max: {max(ans_rerank):.4f}")
    print(f"Unanswerable dense mean: {sum(unans_dense)/len(unans_dense):.4f}, min: {min(unans_dense):.4f}, max: {max(unans_dense):.4f}")
    print(f"Unanswerable rerank mean: {sum(unans_rerank)/len(unans_rerank):.4f}, min: {min(unans_rerank):.4f}, max: {max(unans_rerank):.4f}")

if __name__ == "__main__":
    main()
