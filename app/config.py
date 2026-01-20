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


class ApiConfig(BaseModel):
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


class Settings(BaseModel):
    paths: Paths = Paths()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    api: ApiConfig = ApiConfig()


settings = Settings()

