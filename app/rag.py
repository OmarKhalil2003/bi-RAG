"""
RAG Pipeline Orchestrator
Bilingual Document Q&A (RAG) System

Coordinates retrieval and generation:
Question -> Multilingual Retrieval (Dense top-20 + BGE Reranker top-5)
         -> Confidence Gate & Evidence Assembly
         -> Language-preserving Generation with Citations / Refusal.
"""

from typing import Dict, Any, List, Optional
from app.retrieval import RetrievalService
from app.generation import Generator


class RAGPipeline:
    def __init__(
        self,
        retrieval_service: RetrievalService = None,
        generator: Generator = None,
        top_k: int = 5,
        candidate_k: int = 20
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.generator = generator or Generator()
        self.top_k = top_k
        self.candidate_k = candidate_k

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Execute full RAG pipeline for a given question."""
        # 1. Retrieve evidence
        retrieved_chunks = self.retrieval.search(
            query=question,
            top_k=self.top_k,
            candidate_k=self.candidate_k
        )

        # 2. Generate answer with citations or confidence refusal
        generation_result = self.generator.generate(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        # 3. Assemble response payload
        return {
            "answer": generation_result["answer"],
            "sources": generation_result.get("sources", []),
            "refused": generation_result.get("refused", False),
            "refusal_reason": generation_result.get("refusal_reason"),
            "citations": generation_result.get("citations", []),
            "latency_ms": generation_result.get("latency_ms"),
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
            "top_score": retrieved_chunks[0].get("reranker_score", retrieved_chunks[0].get("score", 0.0)) if retrieved_chunks else 0.0
        }
