"""
Baseline Retrieval Module
Bilingual Document Q&A (RAG) System

Implements dense retrieval baseline using BGE-M3 and FAISS cosine similarity.
"""

from typing import List, Dict, Any
from src.embed import Embedder
from src.index import VectorStore


class BaselineRetriever:
    """
    Baseline Dense Retriever:
    query -> BGE-M3 normalized embedding -> FAISS IndexFlatIP -> top-k chunks.
    """

    def __init__(
        self,
        embedder: Embedder = None,
        vector_store: VectorStore = None,
        index_path: str = "data/index.faiss",
        chunks_path: str = "data/chunks.jsonl"
    ):
        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = vector_store if vector_store is not None else VectorStore(index_path, chunks_path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks for a given query."""
        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k)
        return results
