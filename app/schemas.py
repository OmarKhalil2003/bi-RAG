"""
Pydantic Schemas for FastAPI Service
Bilingual Document Q&A (RAG) System
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask in either Arabic or English."
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or only whitespace.")
        return v.strip()


class SourceItem(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier of the supporting passage chunk.")
    title: str = Field(..., description="Title of the source document.")
    page: int = Field(..., description="Page number of the source document.")
    source_url: str = Field(..., description="Public reference URL of the source document.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question in the question's language with source citations, or refusal.")
    sources: List[SourceItem] = Field(default_factory=list, description="List of cited source passages.")
    refused: Optional[bool] = Field(default=False, description="Whether the question was refused.")
    latency_ms: Optional[float] = Field(default=None, description="End-to-end request latency in milliseconds.")


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_vectors: int
    models: dict
