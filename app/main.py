"""
FastAPI Application Entrypoint
Bilingual Document Q&A (RAG) System

Exposes POST /query with Pydantic validation, structured logging,
and Docker packaging support.
"""

import os
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# Synchronize HF_API_KEY into standard HF_TOKEN if needed
if not os.getenv("HF_TOKEN") and os.getenv("HF_API_KEY"):
    os.environ["HF_TOKEN"] = os.getenv("HF_API_KEY")

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryRequest, QueryResponse, SourceItem, HealthResponse
from app.rag import RAGPipeline
from app.logging_config import setup_logger

logger = setup_logger()

# Global pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on application startup."""
    global rag_pipeline
    logger.info("Initializing Bilingual RAG Pipeline...", extra={"request_id": "startup"})
    try:
        rag_pipeline = RAGPipeline()
        logger.info("Bilingual RAG Pipeline ready for inference.", extra={"request_id": "startup"})
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}", extra={"request_id": "startup"})
    yield
    logger.info("Shutting down Bilingual RAG Service.", extra={"request_id": "shutdown"})


app = FastAPI(
    title="Bilingual Document Q&A (RAG) API",
    description="Arabic and English RAG System using BGE-M3, FAISS, BM25 Hybrid Search, and LLM Generation with Passage Citations.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint providing index status and active models."""
    global rag_pipeline
    index_loaded = rag_pipeline is not None and rag_pipeline.retrieval.vector_store is not None
    total_vecs = rag_pipeline.retrieval.vector_store.index.ntotal if index_loaded else 0

    return HealthResponse(
        status="healthy" if index_loaded else "degraded",
        index_loaded=index_loaded,
        total_vectors=total_vecs,
        models={
            "embedding": os.getenv("MODEL_NAME", "BAAI/bge-m3"),
            "sparse": "BM25Okapi",
            "vector_index": "FAISS IndexFlatIP",
            "generator": os.getenv("GENERATION_MODEL", "openrouter/free")
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(query_req: QueryRequest, request: Request):
    """
    Query endpoint for Bilingual Document Q&A.
    Accepts question in Arabic or English, retrieves evidence passages,
    and returns answer with citations or explicit refusal.
    """
    global rag_pipeline
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized. Please ensure index is built."
        )

    # Generate unique request ID for structured logging
    req_id = str(uuid.uuid4())
    start_time = time.time()
    q_len = len(query_req.question)

    logger.info(
        f"Incoming query (length={q_len} chars)",
        extra={"request_id": req_id}
    )

    try:
        result = rag_pipeline.answer_question(query_req.question)
        latency = round((time.time() - start_time) * 1000, 2)

        # Structured log without exposing sensitive tokens or logging full user questions by default
        logger.info(
            f"Query processed: chunks={result['retrieved_chunk_ids']}, "
            f"top_score={result['top_score']:.4f}, "
            f"refused={result['refused']}, "
            f"latency={latency}ms",
            extra={"request_id": req_id}
        )

        sources_payload = [
            SourceItem(
                chunk_id=s["chunk_id"],
                title=s.get("title", ""),
                page=s.get("page", 1),
                source_url=s.get("source_url", "")
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources_payload,
            refused=result.get("refused", False),
            latency_ms=latency
        )

    except Exception as e:
        latency = round((time.time() - start_time) * 1000, 2)
        logger.error(
            f"Error processing query after {latency}ms: {str(e)}",
            extra={"request_id": req_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the question."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
