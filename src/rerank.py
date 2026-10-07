"""
Reranking Module (The Single Improvement)
Bilingual Document Q&A (RAG) System

Implements multilingual cross-encoder reranking over dense retrieval candidates:
query -> dense retrieve top-M candidates -> BGE-reranker-v2-m3 -> calibrated scoring & dynamic pruning -> top-k.
"""

import os
import math
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import CrossEncoder

from src.retrieve import BaselineRetriever
from src.preprocess import normalize_query

DEFAULT_RERANKER_NAME = os.getenv("RERANKER_NAME", "BAAI/bge-reranker-v2-m3")


def calibrate_reranker_score(raw_score: float) -> float:
    """
    Ensure reranker scores are calibrated probabilities bounded in [0.0, 1.0].
    Applies numerically stable sigmoid if model emits unnormalized logits.
    """
    if 0.0 <= raw_score <= 1.0:
        return float(raw_score)
    try:
        if raw_score >= 0:
            return float(1.0 / (1.0 + math.exp(-raw_score)))
        else:
            exp_val = math.exp(raw_score)
            return float(exp_val / (1.0 + exp_val))
    except OverflowError:
        return 1.0 if raw_score > 0 else 0.0


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_NAME, device: Optional[str] = None):
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
        top_k: int = 5,
        min_score: float = 0.0,
        relative_margin: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks using cross-attention between query and passage.
        Applies score calibration and dynamic evidence pruning.
        """
        if not candidates:
            return []

        cleaned_query = normalize_query(query)
        pairs = [[cleaned_query, c["text"]] for c in candidates]
        raw_scores = self.model.predict(pairs)

        reranked = []
        for cand, raw_s in zip(candidates, raw_scores):
            calibrated = calibrate_reranker_score(float(raw_s))
            item = dict(cand)
            item["dense_score"] = float(cand.get("score", 0.0))
            item["raw_reranker_score"] = float(raw_s)
            item["reranker_score"] = calibrated
            item["score"] = calibrated
            reranked.append(item)

        # Sort descending by calibrated cross-encoder score
        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

        if not reranked:
            return []

        top_score = reranked[0]["reranker_score"]

        # Dynamic evidence pruning: filter out distant distractors below relative margin or absolute floor
        pruned = []
        for rank, item in enumerate(reranked, start=1):
            if min_score > 0.0 and item["reranker_score"] < min_score:
                continue
            if relative_margin > 0.0 and rank > 1:
                if item["reranker_score"] < (top_score * relative_margin):
                    continue
            item["rank"] = rank
            pruned.append(item)
            if len(pruned) >= top_k:
                break

        # Fallback to top_k if all were filtered out by margin
        if not pruned and reranked:
            pruned = reranked[:top_k]

        return pruned


class ImprovedRetriever:
    """
    Improved Retriever:
    Dense retrieval of candidate_k (20) -> BGE Cross-Encoder Reranker -> top_k (5).
    Differs from baseline strictly by adding multilingual cross-encoder reranking with score calibration.
    """

    def __init__(
        self,
        baseline_retriever: BaselineRetriever,
        reranker: Optional[Reranker] = None
    ):
        self.baseline = baseline_retriever
        self.reranker = reranker if reranker is not None else Reranker()

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
        min_score: float = 0.0,
        relative_margin: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve candidate_k dense results, rerank with cross-encoder, return top_k."""
        candidates = self.baseline.retrieve(query, top_k=candidate_k)
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            min_score=min_score,
            relative_margin=relative_margin
        )
        return reranked
