"""
Hybrid Retrieval Module (The Single Improvement)
Bilingual Document Q&A (RAG) System

Implements multilingual Hybrid Search combining:
1. Dense semantic vector retrieval (BGE-M3 + FAISS IndexFlatIP)
2. Sparse lexical keyword retrieval (BM25Okapi over tokenized multilingual chunks)
3. Reciprocal Rank Fusion (RRF) combining dense semantic recall with exact sparse lexical precision.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.retrieve import BaselineRetriever
from src.preprocess import normalize_query, strip_chunk_header


def tokenize_multilingual(text: str) -> List[str]:
    """
    Multilingual tokenizer extracting Arabic and Latin alphanumeric tokens.
    Applies canonical normalization (stripping diacritics, tatweel, and standardizing numerals).
    """
    clean_text = normalize_query(text).lower()
    tokens = re.findall(r"[\u0621-\u064Aa-z0-9]+", clean_text)
    return tokens


class BM25Store:
    """
    Sparse lexical index using BM25Okapi over the corpus chunks.
    """

    def __init__(self, chunks_path: str = "data/chunks.jsonl"):
        self.chunks_path = chunks_path
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        print(f"Building BM25 sparse index from '{self.chunks_path}'...")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        self.corpus_tokens = [tokenize_multilingual(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"BM25 index built with {len(self.chunks)} documents.")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search BM25 sparse index and return top_k candidates with scores."""
        query_tokens = tokenize_multilingual(query)
        if not query_tokens or self.bm25 is None:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(raw_scores[idx])
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = score
            chunk["bm25_rank"] = rank
            chunk["raw_text"] = strip_chunk_header(chunk.get("text", ""))
            results.append(chunk)

        return results


class HybridRetriever:
    """
    Enhanced Hybrid Retriever:
    Combines Dense Vector Retrieval (BGE-M3 + FAISS) with Sparse BM25 via Reciprocal Rank Fusion (RRF).
    Differs from baseline strictly by adding sparse lexical BM25 retrieval and rank fusion.
    """

    def __init__(
        self,
        baseline_retriever: BaselineRetriever,
        bm25_store: Optional[BM25Store] = None,
        k_rrf: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0
    ):
        self.baseline = baseline_retriever
        self.bm25 = bm25_store if bm25_store is not None else BM25Store()
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search:
        1. Dense retrieval of candidate_k items.
        2. Sparse BM25 retrieval of candidate_k items.
        3. Reciprocal Rank Fusion (RRF) to combine ranks.
        4. Return top_k fused candidates.
        """
        # 1. Dense retrieval
        dense_results = self.baseline.retrieve(query, top_k=candidate_k)
        dense_ranks = {r["chunk_id"]: (i, r) for i, r in enumerate(dense_results, 1)}

        # 2. Sparse BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        bm25_ranks = {r["chunk_id"]: (i, r) for i, r in enumerate(bm25_results, 1)}

        # 3. Reciprocal Rank Fusion
        all_chunk_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        fused_items = {}

        for cid in all_chunk_ids:
            dense_rank, dense_item = dense_ranks.get(cid, (None, None))
            bm25_rank, bm25_item = bm25_ranks.get(cid, (None, None))

            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += self.dense_weight / (self.k_rrf + dense_rank)
            if bm25_rank is not None:
                rrf_score += self.bm25_weight / (self.k_rrf + bm25_rank)

            # Base chunk metadata from whichever source retrieved it
            source_item = dense_item if dense_item is not None else bm25_item
            merged = dict(source_item)
            merged["dense_score"] = float(dense_item.get("score", 0.0)) if dense_item else 0.0
            merged["dense_rank"] = dense_rank
            merged["bm25_score"] = float(bm25_item.get("bm25_score", 0.0)) if bm25_item else 0.0
            merged["bm25_rank"] = bm25_rank
            merged["rrf_score"] = rrf_score
            merged["score"] = rrf_score  # Primary sort key
            fused_items[cid] = merged

        # Sort descending by RRF score
        sorted_results = sorted(fused_items.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Apply min_score filter if requested
        if min_score > 0.0:
            sorted_results = [r for r in sorted_results if r["rrf_score"] >= min_score]

        # Assign final hybrid ranks
        for rank, item in enumerate(sorted_results[:top_k], start=1):
            item["rank"] = rank

        return sorted_results[:top_k]
