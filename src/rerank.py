"""
Reranking Module (The Single Improvement)
Bilingual Document Q&A (RAG) System

Implements multilingual cross-encoder reranking over dense retrieval candidates:
query -> dense retrieve top-20 -> BGE-reranker-v2-m3 -> top-5.
"""

import os
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import CrossEncoder

from src.retrieve import BaselineRetriever

DEFAULT_RERANKER_NAME = os.getenv("RERANKER_NAME", "BAAI/bge-reranker-v2-m3")


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_NAME, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        print(f"Loading cross-encoder reranker '{model_name}' on device '{self.device}'...")
        self.model = CrossEncoder(model_name, device=self.device)
        print("Reranker loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunks for a query using cross-encoder scores.
        Returns top_k chunks sorted by reranker score descending.
        """
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["dense_score"] = cand.get("score")
            item["reranker_score"] = float(score)
            item["score"] = float(score)
            reranked.append(item)

        # Sort descending by reranker score
        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:top_k]


class ImprovedRetriever:
    """
    Improved Retriever:
    Dense retrieval of candidate_k (20) -> BGE Cross-Encoder Reranker -> top_k (5).
    Differs from baseline ONLY by adding cross-encoder reranking.
    """

    def __init__(
        self,
        baseline_retriever: BaselineRetriever,
        reranker: Reranker = None
    ):
        self.baseline = baseline_retriever
        self.reranker = reranker if reranker is not None else Reranker()

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve candidate_k dense results, rerank with cross-encoder, return top_k."""
        candidates = self.baseline.retrieve(query, top_k=candidate_k)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked
