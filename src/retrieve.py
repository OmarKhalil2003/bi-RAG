"""
Baseline Retrieval Module
Bilingual Document Q&A (RAG) System

Implements dense retrieval baseline with canonical query preprocessing,
BGE-M3 multilingual dense embeddings, and FAISS exact inner-product search (cosine similarity).
"""

from typing import List, Dict, Any, Optional
from src.embed import Embedder
from src.index import VectorStore
from src.preprocess import normalize_query, strip_chunk_header


class BaselineRetriever:
    """
    Baseline Dense Retriever:
    query -> canonical normalization -> BGE-M3 normalized embedding -> FAISS IndexFlatIP -> top-k chunks.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        index_path: str = "data/index.faiss",
        chunks_path: str = "data/chunks.jsonl"
    ):
        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = vector_store if vector_store is not None else VectorStore(index_path, chunks_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for a given query with score calibration and metadata framing.
        """
        cleaned_query = normalize_query(query)
        if not cleaned_query:
            return []

        query_vec = self.embedder.embed_query(cleaned_query)
        raw_results = self.vector_store.search(query_vec, top_k=top_k)

        calibrated_results = []
        for rank, res in enumerate(raw_results, start=1):
            raw_score = float(res.get("score", 0.0))
            # Bound cosine similarity in [0.0, 1.0]
            bounded_score = max(0.0, min(1.0, raw_score))
            if bounded_score < min_score:
                continue

            item = dict(res)
            item["score"] = bounded_score
            item["rank"] = rank
            item["raw_text"] = strip_chunk_header(res.get("text", ""))
            calibrated_results.append(item)

        return calibrated_results
