from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .config import settings
from .models import DocumentChunk, DocumentMetadata, RetrievedChunk


@dataclass
class FaissIndex:
    index: faiss.IndexFlatIP
    chunks: List[DocumentChunk]


def _metadata_to_dict(md: DocumentMetadata) -> dict:
    return {
        "doc_id": md.doc_id,
        "doc_title": md.doc_title,
        "source": md.source,
        "page_number": md.page_number,
        "created_at": md.created_at.isoformat() if md.created_at else None,
        "section": md.section,
        "extra": md.extra,
    }


def _metadata_from_dict(d: dict) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=d["doc_id"],
        doc_title=d.get("doc_title"),
        source=d.get("source"),
        page_number=d.get("page_number"),
        created_at=None,  # Simplify: omit parsing for now
        section=d.get("section"),
        extra=d.get("extra") or {},
    )


def save_index(index: FaissIndex, name: str = "default") -> None:
    index_dir = settings.paths.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / f"{name}.faiss"
    meta_path = index_dir / f"{name}.jsonl"

    faiss.write_index(index.index, str(faiss_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for chunk in index.chunks:
            rec = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "metadata": _metadata_to_dict(chunk.metadata),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_index(name: str = "default") -> FaissIndex:
    index_dir = settings.paths.index_dir
    faiss_path = index_dir / f"{name}.faiss"
    meta_path = index_dir / f"{name}.jsonl"

    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index '{name}' not found in {index_dir}")

    faiss_index = faiss.read_index(str(faiss_path))

    chunks: List[DocumentChunk] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            md = _metadata_from_dict(rec["metadata"])
            chunk = DocumentChunk(
                chunk_id=rec["chunk_id"],
                text=rec["text"],
                metadata=md,
                chunk_index=rec["chunk_index"],
            )
            chunks.append(chunk)

    return FaissIndex(index=faiss_index, chunks=chunks)


def build_faiss_index(embeddings: np.ndarray, chunks: List[DocumentChunk]) -> FaissIndex:
    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embeddings and chunks length mismatch")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return FaissIndex(index=index, chunks=chunks)


def search_index(
    faiss_index: FaissIndex,
    query_vec: np.ndarray,
    top_k: int,
) -> List[RetrievedChunk]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]

    scores, indices = faiss_index.index.search(query_vec, top_k)
    results: List[RetrievedChunk] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0 or idx >= len(faiss_index.chunks):
            continue
        chunk = faiss_index.chunks[idx]
        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=float(score),
                rank=rank,
            )
        )
    return results

