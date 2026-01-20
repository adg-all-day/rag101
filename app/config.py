from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class Paths(BaseModel):
    root_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root_dir / "data"
    index_dir: Path = root_dir / "indices"


class EmbeddingConfig(BaseModel):
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


class RetrievalConfig(BaseModel):
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    use_hybrid: bool = os.getenv("USE_HYBRID", "false").lower() == "true"
    hybrid_dense_weight: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
    hybrid_sparse_weight: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))
    use_rerank: bool = os.getenv("USE_RERANK", "false").lower() == "true"
    rerank_model_name: str = os.getenv(
        "RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


class CacheConfig(BaseModel):
    enable_query_cache: bool = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
    enable_embedding_cache: bool = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
    query_cache_size: int = int(os.getenv("QUERY_CACHE_SIZE", "1024"))
    embedding_cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1024"))


class ApiConfig(BaseModel):
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


class Settings(BaseModel):
    paths: Paths = Paths()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    cache: CacheConfig = CacheConfig()
    api: ApiConfig = ApiConfig()


settings = Settings()
