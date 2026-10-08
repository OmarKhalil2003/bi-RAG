"""
Answer Generation and Quality Evaluation Script
Bilingual Document Q&A (RAG) System

Evaluates generation quality on evaluation/gold.jsonl:
- Answer correctness (fully correct / answerable questions)
- Partial correctness
- Unsupported-answer rate (factual claims unsupported by retrieved passages)
- Unanswerable refusal accuracy (correct refusals / unanswerable questions)
Saves results to evaluation/answer_evaluation_results.json.
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.retrieve import BaselineRetriever
from src.hybrid import HybridRetriever, BM25Store
from app.generation import Generator, is_refusal


def score_answer_correctness(
    generated_answer: str,
    gold_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    gold_chunk_ids: List[str],
    refused: bool
) -> Dict[str, Any]:
    """
    Score answer correctness:
    2 = fully correct (factual claims match gold answer and supported by evidence)
    1 = partially correct (some facts correct but incomplete)
    0 = incorrect or improperly refused
    Also determines if answer is unsupported by evidence.
    """
    if refused:
        return {
            "correctness_score": 0,
            "correctness_label": "incorrect_refusal",
            "unsupported": False,
            "notes": "Answerable question was refused by system."
        }

    # Extract numbers or key entities to check faithfulness
    gold_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", gold_answer))
    gen_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", generated_answer))

    # Evidence text
    evidence_text = " ".join([c.get("text", "") for c in retrieved_chunks])

    # Check if numbers in generated answer exist in evidence
    unsupported_numbers = gen_numbers - set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))
    unsupported = len(unsupported_numbers) > 0

    # Overlap of gold numbers
    if gold_numbers:
        matched_numbers = gold_numbers.intersection(gen_numbers)
        number_ratio = len(matched_numbers) / len(gold_numbers)
    else:
        number_ratio = 1.0

    # Key term overlap
    gold_words = set(re.findall(r"\w{4,}", gold_answer.lower()))
    gen_words = set(re.findall(r"\w{4,}", generated_answer.lower()))
    word_overlap = len(gold_words.intersection(gen_words)) / len(gold_words) if gold_words else 1.0

    if number_ratio >= 0.8 and word_overlap >= 0.25 and not unsupported:
        score = 2
        label = "fully_correct"
    elif (number_ratio >= 0.5 or word_overlap >= 0.20) and not unsupported:
        score = 1
        label = "partially_correct"
    else:
        score = 0
        label = "incorrect"

    return {
        "correctness_score": score,
        "correctness_label": label,
        "unsupported": unsupported,
        "notes": f"number_match={number_ratio:.2f}, word_overlap={word_overlap:.2f}, unsupported_nums={list(unsupported_numbers)}"
    }


def main():
    print("=" * 70)
    print("BILINGUAL RAG ANSWER GENERATION EVALUATION")
    print("=" * 70)

    gold_path = os.path.join("evaluation", "gold.jsonl")
    with open(gold_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(questions)} gold questions.")

    # Initialize retrieval & generator
    print("Initializing Enhanced Hybrid Retriever (Dense + BM25 RRF)...")
    baseline = BaselineRetriever()
    bm25 = BM25Store()
    hybrid = HybridRetriever(baseline_retriever=baseline, bm25_store=bm25)
    generator = Generator()

    eval_records = []
    answerable_count = 0
    unanswerable_count = 0
    fully_correct_count = 0
    partially_correct_count = 0
    unsupported_answer_count = 0
    correct_refusal_count = 0

    print("\nRunning generation on all 40 questions...")
    for idx, item in enumerate(questions, 1):
        qid = item["id"]
        q_text = item["question"]
        is_ans = item["answerable"]
        lang = item["language"]
        gold_ans = item.get("gold_answer")
        gold_cids = item.get("supporting_chunk_ids", [])

        # Retrieve top 5 using hybrid retriever (candidate_k=20)
        retrieved_chunks = hybrid.retrieve(q_text, candidate_k=20, top_k=5)
        # Generate answer
        gen_result = generator.generate(q_text, retrieved_chunks)

        answer_text = gen_result["answer"]
        refused = gen_result["refused"]
        citations = gen_result.get("citations", [])

        record = {
            "id": qid,
            "language": lang,
            "question": q_text,
            "answerable": is_ans,
            "gold_answer": gold_ans,
            "gold_chunk_ids": gold_cids,
            "generated_answer": answer_text,
            "refused": refused,
            "refusal_reason": gen_result.get("refusal_reason"),
            "citations": citations,
            "top_retrieved_chunk_id": retrieved_chunks[0]["chunk_id"] if retrieved_chunks else None,
            "top_score": retrieved_chunks[0].get("rrf_score", retrieved_chunks[0].get("score", 0.0)) if retrieved_chunks else 0.0,
            "latency_ms": gen_result.get("latency_ms", 0.0)
        }

        if not is_ans:
            unanswerable_count += 1
            if refused:
                correct_refusal_count += 1
                record["evaluation"] = "correct_refusal"
            else:
                record["evaluation"] = "failed_refusal (hallucination)"
        else:
            answerable_count += 1
            scoring = score_answer_correctness(
                generated_answer=answer_text,
                gold_answer=gold_ans,
                retrieved_chunks=retrieved_chunks,
                gold_chunk_ids=gold_cids,
                refused=refused
            )
            record["evaluation"] = scoring["correctness_label"]
            record["correctness_score"] = scoring["correctness_score"]
            record["unsupported"] = scoring["unsupported"]
            record["eval_notes"] = scoring["notes"]

            if scoring["correctness_score"] == 2:
                fully_correct_count += 1
            elif scoring["correctness_score"] == 1:
                partially_correct_count += 1

            if scoring["unsupported"]:
                unsupported_answer_count += 1

        eval_records.append(record)
        status = record.get("evaluation", "")
        print(f"[{idx:02d}/40] {qid} ({lang}) [{'Ans' if is_ans else 'Unans'}] -> {status}")
        time.sleep(0.5)  # slight pause for free API rate limits

    # Compute Final Aggregate Metrics
    correctness_rate = fully_correct_count / answerable_count if answerable_count else 0.0
    partial_rate = partially_correct_count / answerable_count if answerable_count else 0.0
    unsupported_rate = unsupported_answer_count / answerable_count if answerable_count else 0.0
    refusal_accuracy = correct_refusal_count / unanswerable_count if unanswerable_count else 0.0

    print("\n" + "=" * 70)
    print("GENERATION QUALITY EVALUATION METRICS")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Result':<10}")
    print("-" * 50)
    print(f"{'Answer correctness':<35} | {correctness_rate:.4f} ({fully_correct_count}/{answerable_count})")
    print(f"{'Partial correctness':<35} | {partial_rate:.4f} ({partially_correct_count}/{answerable_count})")
    print(f"{'Unsupported-answer rate':<35} | {unsupported_rate:.4f} ({unsupported_answer_count}/{answerable_count})")
    print(f"{'Unanswerable refusal accuracy':<35} | {refusal_accuracy:.4f} ({correct_refusal_count}/{unanswerable_count})")
    print("-" * 50)

    # Save detailed evaluation outputs
    output_data = {
        "summary": {
            "total_questions": len(questions),
            "answerable_questions": answerable_count,
            "unanswerable_questions": unanswerable_count,
            "fully_correct": fully_correct_count,
            "partially_correct": partially_correct_count,
            "answer_correctness": round(correctness_rate, 4),
            "partial_correctness": round(partial_rate, 4),
            "unsupported_answer_rate": round(unsupported_rate, 4),
            "unanswerable_refusal_accuracy": round(refusal_accuracy, 4)
        },
        "records": eval_records
    }

    out_file = os.path.join("evaluation", "answer_evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nAnswer evaluation results saved to {out_file}")


if __name__ == "__main__":
    main()
