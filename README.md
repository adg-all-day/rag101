# RAG System (Minimal Production-Style Skeleton)

This project implements a minimal but realistic RAG system matching the high-level architecture you described. It provides:

- Offline **indexing pipeline**: load documents → chunk → embed → FAISS vector index
- Online **query pipeline**: embed query → retrieve top-k chunks → return structured results
- A small **FastAPI** service to query the index.

## 1. Setup

```bash
cd rag101  # or the cloned repo directory
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -e .
```

> Note: The project uses `sentence-transformers` and `faiss-cpu`, which will download a small embedding model on first run.

## 2. Prepare Documents

Put `.txt` or `.md` files under the `data/` directory, for example:

```text
data/
  intro_to_ai.md
  notes/
    rag_overview.txt
```

You can extend the loader in `app/loader.py` to support PDFs, HTML, etc.

## 3. Build the Index

```bash
python scripts/build_index.py
```

This will:

- Load documents from `data/`
- Chunk them with simple paragraph-based chunking
- Generate embeddings with a sentence-transformers model
- Build a cosine-similarity FAISS index under `indices/default.*`

## 4. Run the API

```bash
uvicorn app.main:app --reload
```

Then query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this corpus about?", "top_k": 5}'
```

To get a generated answer with citations:

```bash
export OPENAI_API_KEY=sk-...   # or set according to LLM_API_KEY_ENV
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this corpus about?", "top_k": 5}'
```

To also see the raw retrieved chunks alongside the answer, set `include_chunks`:

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this corpus about?", "top_k": 5, "include_chunks": true}'
```

You can also check basic stats:

```bash
curl http://localhost:8000/index/stats
```

## 5. Configuration

Environment variables (optional):

- `EMBEDDING_MODEL`: sentence-transformers model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_BATCH_SIZE`: batch size for embedding (default: `32`)
- `RETRIEVAL_TOP_K`: default `top_k` if not passed (default: `10`)
- `RERANK_TOP_K`: how many retrieved chunks to rerank when reranking is enabled (default: `5`)

Hybrid / reranking toggles:

- `USE_HYBRID`: set to `true` to enable hybrid dense + BM25 retrieval (default: `false`)
- `HYBRID_DENSE_WEIGHT`: weight for dense cosine similarity in fusion (default: `0.7`)
- `HYBRID_SPARSE_WEIGHT`: weight for BM25 score in fusion (default: `0.3`)
- `USE_RERANK`: set to `true` to enable cross-encoder reranking (default: `false`)
- `RERANK_MODEL`: sentence-transformers cross-encoder model name (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)

Caching:

- `ENABLE_QUERY_CACHE`: cache full query responses in-memory (default: `true`)
- `ENABLE_EMBEDDING_CACHE`: cache query embeddings in-memory (default: `true`)
- `QUERY_CACHE_SIZE`: max cached queries (default: `1024`)
- `EMBEDDING_CACHE_SIZE`: max cached query embeddings (default: `1024`)

LLM:

- `LLM_PROVIDER`: currently `openai` is supported (default: `openai`)
- `LLM_MODEL`: chat model for generation (default: `gpt-4o-mini`)
- `LLM_TEMPERATURE`: sampling temperature for answers (default: `0.1`)
- `LLM_MAX_TOKENS`: max tokens for the answer (default: `512`)
- `LLM_API_KEY_ENV`: name of env var holding the API key (default: `OPENAI_API_KEY`)

You can create a `.env` file in the project root to set them.

## 6. Extending Toward Full Architecture

This skeleton is intentionally focused on the **retrieval** half of RAG:

- Hybrid search: dense FAISS + BM25 (enabled via `USE_HYBRID=true`)
- Optional cross-encoder reranking for higher precision (`USE_RERANK=true`)
- Simple in-process caching for embeddings and query responses
- You can still add an LLM and prompts in a new `app/generation.py`, richer document loaders, and evaluation / monitoring modules.

The current layout and abstractions (`loader`, `chunking`, `embeddings`, `vector_store`, `query_pipeline`) are designed so you can plug in advanced components incrementally.
