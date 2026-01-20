from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import settings
from .generation import answer_question
from .models import AnswerRequest, AnswerResponse, QueryRequest, QueryResponse
from .query_pipeline import run_query
from .vector_store import load_index


app = FastAPI(title="RAG System", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/index/stats")
def index_stats() -> dict:
    try:
        idx = load_index("default")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found. Run indexing first.")

    dim = idx.index.d
    num_chunks = idx.index.ntotal
    doc_ids = {c.metadata.doc_id for c in idx.chunks}
    return {"num_documents": len(doc_ids), "num_chunks": num_chunks, "dim": dim}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return run_query(req)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found. Run indexing first.")


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    try:
        return answer_question(req)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found. Run indexing first.")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def get_app() -> FastAPI:
    return app
