"""
Retrieval Service Wrapper
Bilingual Document Q&A (RAG) System
"""

import os
from typing import List, Dict, Any, Optional

from src.retrieve import BaselineRetriever
from src.hybrid import HybridRetriever, BM25Store
from src.embed import Embedder
from src.index import VectorStore


class RetrievalService:
    """Wrapper managing vector store, baseline dense retriever, and hybrid retriever."""

    def __init__(
        self,
        index_path: str = "data/index.faiss",
        chunks_path: str = "data/chunks.jsonl",
        use_hybrid: bool = True
    ):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.use_hybrid = use_hybrid

        self.embedder = Embedder()
        self.vector_store = VectorStore(index_path, chunks_path)
        self.baseline = BaselineRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store
        )
        if self.use_hybrid:
            self.bm25_store = BM25Store(chunks_path=chunks_path)
            self.hybrid = HybridRetriever(
                baseline_retriever=self.baseline,
                bm25_store=self.bm25_store
            )
        else:
            self.bm25_store = None
            self.hybrid = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform retrieval using enhanced hybrid retriever if available, else baseline dense."""
        if self.use_hybrid and self.hybrid is not None:
            return self.hybrid.retrieve(
                query=query,
                candidate_k=candidate_k,
                top_k=top_k,
                min_score=min_score
            )
        else:
            return self.baseline.retrieve(query=query, top_k=top_k, min_score=min_score)
