from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    doc_id: str
    doc_title: Optional[str] = None
    source: Optional[str] = None
    page_number: Optional[int] = None
    created_at: Optional[datetime] = None
    section: Optional[str] = None
    extra: Dict[str, Any] = {}


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    chunk_index: int


class IndexStats(BaseModel):
    num_documents: int
    num_chunks: int
    dim: int


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]


class SourceCitation(BaseModel):
    source_id: int
    doc_title: Optional[str] = None
    source: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    score: float


class AnswerRequest(BaseModel):
    query: str
    top_k: int | None = None
    include_chunks: bool = False


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    retrieval_results: Optional[List[RetrievedChunk]] = None
