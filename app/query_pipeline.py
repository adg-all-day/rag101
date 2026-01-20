from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import settings
from .embeddings import embed_texts, get_embedding_model
from .models import QueryRequest, QueryResponse, RetrievedChunk
from .vector_store import FaissIndex, load_index


@lru_cache(maxsize=1)
def _load_default_index() -> FaissIndex:
    return load_index("default")


_bm25_retriever: Optional[BM25Okapi] = None
_reranker_model: Optional[CrossEncoder] = None
_query_embedding_cache: Dict[str, np.ndarray] = {}
_query_result_cache: Dict[Tuple[str, int], QueryResponse] = {}


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _get_bm25(index: FaissIndex) -> BM25Okapi:
    global _bm25_retriever
    if _bm25_retriever is None:
        corpus_tokens = [_tokenize(chunk.text) for chunk in index.chunks]
        _bm25_retriever = BM25Okapi(corpus_tokens)
    return _bm25_retriever


def _get_query_embedding(query: str) -> np.ndarray:
    if settings.cache.enable_embedding_cache and query in _query_embedding_cache:
        return _query_embedding_cache[query]

    vec = embed_texts([query])[0]
    if settings.cache.enable_embedding_cache:
        if len(_query_embedding_cache) >= settings.cache.embedding_cache_size:
            # Simple eviction: remove an arbitrary item.
            _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
        _query_embedding_cache[query] = vec
    return vec


def _dense_search(index: FaissIndex, query_vec: np.ndarray, top_k: int) -> Dict[int, float]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    scores, indices = index.index.search(query_vec.astype("float32"), top_k)
    dense_scores: Dict[int, float] = {}
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        dense_scores[int(idx)] = float(score)
    return dense_scores


def _sparse_search(index: FaissIndex, query: str, top_k: int) -> Dict[int, float]:
    bm25 = _get_bm25(index)
    scores = bm25.get_scores(_tokenize(query))
    if len(scores) == 0:
        return {}

    # Take top_k indices by BM25 score.
    top_indices = np.argsort(scores)[::-1][:top_k]
    return {int(i): float(scores[i]) for i in top_indices if scores[i] > 0}


def _normalize_scores(scores: Dict[int, float], shift: bool = False) -> Dict[int, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype="float32")
    if shift:
        # For cosine similarities in [-1, 1], shift to [0, 1].
        values = (values + 1.0) / 2.0
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v > min_v:
        values = (values - min_v) / (max_v - min_v)
    else:
        values = np.ones_like(values)
    return {idx: float(v) for idx, v in zip(scores.keys(), values)}


def _hybrid_search(
    index: FaissIndex,
    query: str,
    query_vec: np.ndarray,
    top_k: int,
) -> List[RetrievedChunk]:
    dense_raw = _dense_search(index, query_vec, top_k)
    sparse_raw = _sparse_search(index, query, top_k)

    dense_norm = _normalize_scores(dense_raw, shift=True)
    sparse_norm = _normalize_scores(sparse_raw, shift=False)

    w_dense = settings.retrieval.hybrid_dense_weight
    w_sparse = settings.retrieval.hybrid_sparse_weight

    all_indices = set(dense_norm.keys()) | set(sparse_norm.keys())
    combined_scores: Dict[int, float] = {}
    for idx in all_indices:
        d = dense_norm.get(idx, 0.0)
        s = sparse_norm.get(idx, 0.0)
        combined_scores[idx] = w_dense * d + w_sparse * s

    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[RetrievedChunk] = []
    for rank, (idx, score) in enumerate(sorted_indices, start=1):
        chunk = index.chunks[idx]
        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=float(score),
                rank=rank,
            )
        )
    return results


def _dense_only_search(
    index: FaissIndex,
    query_vec: np.ndarray,
    top_k: int,
) -> List[RetrievedChunk]:
    dense_raw = _dense_search(index, query_vec, top_k)
    # Sort by raw cosine similarity.
    sorted_indices = sorted(dense_raw.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[RetrievedChunk] = []
    for rank, (idx, score) in enumerate(sorted_indices, start=1):
        chunk = index.chunks[idx]
        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=float(score),
                rank=rank,
            )
        )
    return results


def _get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.retrieval.rerank_model_name)
    return _reranker_model


def _rerank_results(
    query: str,
    candidates: List[RetrievedChunk],
    top_k: int,
) -> List[RetrievedChunk]:
    if not candidates:
        return candidates

    reranker = _get_reranker()
    pairs = [(query, c.chunk.text) for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    reranked: List[RetrievedChunk] = []
    for rank, (cand, score) in enumerate(scored[:top_k], start=1):
        reranked.append(
            RetrievedChunk(
                chunk=cand.chunk,
                score=float(score),
                rank=rank,
            )
        )
    return reranked


def run_query(req: QueryRequest) -> QueryResponse:
    index = _load_default_index()
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    top_k = req.top_k or settings.retrieval.top_k

    cache_key = (req.query, top_k)
    if settings.cache.enable_query_cache and cache_key in _query_result_cache:
        return _query_result_cache[cache_key]

    query_vec = _get_query_embedding(req.query)
    if query_vec.shape[0] != dim:
        raise ValueError("Query embedding dimension mismatch")

    if settings.retrieval.use_hybrid:
        results = _hybrid_search(index, req.query, np.asarray(query_vec, dtype="float32"), top_k=top_k)
    else:
        results = _dense_only_search(index, np.asarray(query_vec, dtype="float32"), top_k=top_k)

    if settings.retrieval.use_rerank:
        rerank_top_k = min(settings.retrieval.rerank_top_k, len(results))
        results = _rerank_results(req.query, results, top_k=rerank_top_k)

    response = QueryResponse(query=req.query, results=results)

    if settings.cache.enable_query_cache:
        if len(_query_result_cache) >= settings.cache.query_cache_size:
            _query_result_cache.pop(next(iter(_query_result_cache)))
        _query_result_cache[cache_key] = response

    return response

