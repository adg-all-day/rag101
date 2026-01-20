from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from .config import settings
from .embeddings import embed_texts, get_embedding_model
from .models import QueryRequest, QueryResponse
from .vector_store import FaissIndex, load_index, search_index


@lru_cache(maxsize=1)
def _load_default_index() -> FaissIndex:
    return load_index("default")


def run_query(req: QueryRequest) -> QueryResponse:
    index = _load_default_index()
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    query_vec = embed_texts([req.query])[0]
    if query_vec.shape[0] != dim:
        raise ValueError("Query embedding dimension mismatch")

    top_k = req.top_k or settings.retrieval.top_k
    results = search_index(index, np.asarray(query_vec, dtype="float32"), top_k=top_k)
    return QueryResponse(query=req.query, results=results)

