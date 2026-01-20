# RAG System (Minimal Production-Style Skeleton)

This project implements a minimal but realistic RAG system matching the high-level architecture you described. It provides:

- Offline **indexing pipeline**: load documents → chunk → embed → FAISS vector index
- Online **query pipeline**: embed query → retrieve top-k chunks → return structured results
- A small **FastAPI** service to query the index.

## 1. Setup

```bash
cd rag
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

You can also check basic stats:

```bash
curl http://localhost:8000/index/stats
```

## 5. Configuration

Environment variables (optional):

- `EMBEDDING_MODEL`: sentence-transformers model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_BATCH_SIZE`: batch size for embedding (default: `32`)
- `RETRIEVAL_TOP_K`: default `top_k` if not passed (default: `10`)

You can create a `.env` file in the project root to set them.

## 6. Extending Toward Full Architecture

This skeleton is intentionally focused on the **retrieval** half of RAG:

- Add an LLM and generation prompts in a new `app/generation.py`.
- Implement hybrid search (BM25 + vectors) and reranking modules.
- Add richer document loaders (PDF, HTML, web, APIs).
- Implement caching, evaluation, and monitoring as separate services or modules.

The current layout and abstractions (`loader`, `chunking`, `embeddings`, `vector_store`, `query_pipeline`) are designed so you can plug in those advanced components incrementally.

