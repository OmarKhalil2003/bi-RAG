"""
Build Index Script
Bilingual Document Q&A (RAG) System

Executes the complete index building pipeline:
1. Verifies/runs ingestion from raw PDFs to data/chunks.jsonl
2. Generates normalized BGE-M3 embeddings
3. Builds and persists the FAISS vector index at data/index.faiss
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import run_ingestion
from src.index import build_faiss_index, VectorStore
from src.embed import Embedder


def main():
    print("=" * 60)
    print("Step 1: Checking corpus chunks...")
    chunks_path = os.path.join("data", "chunks.jsonl")
    if not os.path.exists(chunks_path) or os.path.getsize(chunks_path) == 0:
        print("Chunks not found. Running ingestion pipeline...")
        stats = run_ingestion()
        print(f"Ingestion generated {stats['total_chunks']} chunks.")
    else:
        print(f"Existing chunks found at {chunks_path}.")

    print("\n" + "=" * 60)
    print("Step 2: Building FAISS index with BGE-M3 embeddings...")
    start_time = time.time()
    embedder = Embedder()
    index_path = os.path.join("data", "index.faiss")
    index, chunks = build_faiss_index(
        chunks_path=chunks_path,
        index_output_path=index_path,
        embedder=embedder
    )
    elapsed = round(time.time() - start_time, 2)
    print(f"FAISS index built with {index.ntotal} vectors in {elapsed}s.")

    print("\n" + "=" * 60)
    print("Step 3: Validating vector store retrieval...")
    store = VectorStore(index_path=index_path, chunks_path=chunks_path)
    test_queries = [
        "What is the required retention period for audit logs?",
        "ما هي فترات الاحتفاظ الإلزامية بسجلات التدقيق الأمني؟"
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        q_vec = embedder.embed_query(q)
        results = store.search(q_vec, top_k=2)
        for r in results:
            print(f"  -> [{r['chunk_id']}] (score: {r['score']:.4f}, doc: {r['document_id']}): {r['text'][:90]}...")

    print("\n" + "=" * 60)
    print("Index build completed successfully!")


if __name__ == "__main__":
    main()
