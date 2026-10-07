"""
Retrieval Service Wrapper
Bilingual Document Q&A (RAG) System
"""

import os
from typing import List, Dict, Any, Optional

from src.retrieve import BaselineRetriever
from src.rerank import ImprovedRetriever, Reranker
from src.embed import Embedder
from src.index import VectorStore


class RetrievalService:
    """Wrapper managing vector store, baseline retriever, and reranker."""

    def __init__(
        self,
        index_path: str = "data/index.faiss",
        chunks_path: str = "data/chunks.jsonl",
        use_reranking: bool = True
    ):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.use_reranking = use_reranking

        self.embedder = Embedder()
        self.vector_store = VectorStore(index_path, chunks_path)
        self.baseline = BaselineRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store
        )
        if self.use_reranking:
            self.improved = ImprovedRetriever(
                baseline_retriever=self.baseline,
                reranker=Reranker()
            )
        else:
            self.improved = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform retrieval using improved retriever if available, else baseline."""
        if self.use_reranking and self.improved is not None:
            return self.improved.retrieve(query, candidate_k=candidate_k, top_k=top_k)
        else:
            return self.baseline.retrieve(query, top_k=top_k)
