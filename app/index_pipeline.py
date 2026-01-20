from __future__ import annotations

from typing import List

from .chunking import chunk_documents
from .config import settings
from .embeddings import embed_texts
from .loader import load_text_files
from .models import DocumentChunk
from .vector_store import build_faiss_index, save_index


def build_index_from_dir(data_dir: str | None = None) -> List[DocumentChunk]:
    root = settings.paths.data_dir if data_dir is None else settings.paths.root_dir / data_dir
    docs = list(load_text_files(root))
    chunks = chunk_documents(docs)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    faiss_index = build_faiss_index(embeddings, chunks)
    save_index(faiss_index, name="default")
    return chunks

