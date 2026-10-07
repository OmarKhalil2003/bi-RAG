"""
Embedding Module
Bilingual Document Q&A (RAG) System

Generates normalized dense embeddings using BAAI/bge-m3.
Normalized embeddings enable exact cosine similarity via FAISS IndexFlatIP.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = os.getenv("MODEL_NAME", "BAAI/bge-m3")


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        print(f"Loading embedding model '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Embedding model loaded successfully.")

    def embed_texts(self, texts: List[str], batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of texts and return normalized float32 numpy array.
        Cosine similarity equals inner product of normalized vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into a 1D normalized float32 array."""
        emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return emb[0].astype(np.float32)


def load_chunks(chunks_path: str = "data/chunks.jsonl") -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
