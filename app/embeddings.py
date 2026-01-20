from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embeddings.model_name)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = get_embedding_model()
    texts_list: List[str] = list(texts)
    if not texts_list:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

    batch_size = settings.embeddings.batch_size
    embeddings: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts_list), batch_size), desc="Embedding", unit="batch"):
        batch = texts_list[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb.astype("float32"))

    return np.vstack(embeddings)

