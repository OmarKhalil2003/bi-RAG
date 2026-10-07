"""
Unit Tests for Retrieval Pipelines (Baseline and Improved)
"""

import os
import pytest
from src.index import VectorStore
from src.retrieve import BaselineRetriever
from src.rerank import ImprovedRetriever


@pytest.fixture(scope="module")
def baseline_retriever():
    assert os.path.exists("data/index.faiss"), "data/index.faiss must exist"
    assert os.path.exists("data/chunks.jsonl"), "data/chunks.jsonl must exist"
    return BaselineRetriever()


@pytest.fixture(scope="module")
def improved_retriever(baseline_retriever):
    return ImprovedRetriever(baseline_retriever)


def test_vector_store_loads():
    store = VectorStore("data/index.faiss", "data/chunks.jsonl")
    assert store.index.ntotal > 0
    assert len(store.chunks) == store.index.ntotal


def test_baseline_retrieval_returns_k_results(baseline_retriever):
    results = baseline_retriever.retrieve("audit log retention period", top_k=5)
    assert len(results) == 5
    for item in results:
        assert "chunk_id" in item
        assert "document_id" in item
        assert "score" in item
        assert "text" in item
        assert len(item["chunk_id"]) > 0


def test_improved_retrieval_returns_k_results(improved_retriever):
    results = improved_retriever.retrieve(
        "ما هي المبادئ الأخلاقية للذكاء الاصطناعي؟",
        candidate_k=10,
        top_k=3
    )
    assert len(results) == 3
    for item in results:
        assert "chunk_id" in item
        assert "reranker_score" in item
        assert "dense_score" in item
        assert item["chunk_id"].startswith("doc_ar_") or item["chunk_id"].startswith("doc_en_")


def test_retrieved_chunk_ids_are_valid(baseline_retriever):
    results = baseline_retriever.retrieve("cybersecurity incident response", top_k=3)
    chunk_ids = [r["chunk_id"] for r in results]
    assert all("_p" in cid and "_c" in cid for cid in chunk_ids)
