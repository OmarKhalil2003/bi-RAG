"""
FAISS Indexing Module
Bilingual Document Q&A (RAG) System

Builds and persists a FAISS vector index (IndexFlatIP for cosine similarity)
and loads index and chunk mappings for retrieval.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np

from src.embed import Embedder, load_chunks


def build_faiss_index(
    chunks_path: str = "data/chunks.jsonl",
    index_output_path: str = "data/index.faiss",
    embedder: Embedder = None,
    batch_size: int = 16
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Build FAISS IndexFlatIP index over all chunks from JSONL.
    Cosine similarity is exact because embeddings are L2 normalized.
    """
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    if embedder is None:
        embedder = Embedder()

    texts = [c["text"] for c in chunks]
    print(f"Generating normalized embeddings for {len(texts)} chunks...")
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)

    dimension = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    print(f"Index successfully written to {index_output_path} (total vectors: {index.ntotal}).")

    return index, chunks


class VectorStore:
    """Helper class to load and query FAISS index alongside chunks metadata."""

    def __init__(
        self,
        index_path: str = "data/index.faiss",
        chunks_path: str = "data/chunks.jsonl"
    ):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at {index_path}. Build it first.")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}.")

        self.index = faiss.read_index(index_path)
        self.chunks = load_chunks(chunks_path)
        if self.index.ntotal != len(self.chunks):
            print(f"Warning: Index vector count ({self.index.ntotal}) differs from chunks count ({len(self.chunks)}).")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search FAISS index with normalized query vector.
        Returns top_k chunks with similarity scores.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        query_vector = query_vector.astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
