# cAST-style repository RAG CLI

A code retrieval-augmented generation tool that uses **AST-aware chunking** and **hybrid search** (FAISS vector + BM25 + RRF fusion) to answer questions about any codebase.

## How it works

1. **Ingest** — Clone a repo (or point to a local one), chunk source files using cAST (AST split-then-merge for Python, line-based fallback for other languages), embed chunks with `all-MiniLM-L6-v2`, and save a FAISS HNSW index to disk.
2. **Ask** — Load a saved index and run hybrid retrieval: FAISS vector similarity + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF). Optionally generate an answer with Gemini.

No compilation or build step is needed — AST parsing reads raw source text directly.

## Setup

```bash
uv sync
```

## CLI usage

After `uv sync`, the `cast-rag` command is available:

```bash
cast-rag --help
```

> If you're outside the virtualenv, prefix with `uv run`:
> ```bash
> uv run cast-rag --help
> ```
>
> Or use `python main.py` directly:
> ```bash
> uv run python main.py --help
> ```

### Ingest a repository

From a git URL (clones automatically):

```bash
cast-rag ingest https://github.com/user/repo.git
```

From a local path:

```bash
cast-rag ingest /path/to/local/repo --name my-project
```

Options:
- `--name` — Custom index name (default: derived from URL/path)
- `--strategy` — `cast` (default) or `fixed`

### Query a saved index (hybrid search)

```bash
cast-rag ask --index my-project --query "how does authentication work"
```

### Query a local repo directly (BM25 only, no index saved)

```bash
cast-rag ask --repo /path/to/repo --query "how is auth signature verified"
```

### Generate an answer with Gemini

```bash
export GEMINI_API_KEY=your-key-here
cast-rag ask --index my-project --query "how does caching work" --answer
```

Or pass the key directly:

```bash
cast-rag ask --index my-project --query "how does caching work" --answer --gemini-key YOUR_KEY
```

### Manage indexes

```bash
# List all saved indexes
cast-rag list

# Delete an index
cast-rag delete my-project
```

### Run the built-in experiment

```bash
cast-rag experiment
```

### JSON output

Add `--json` to any `ask` command:

```bash
cast-rag ask --index my-project --query "cache embeddings" --json
```

## Architecture

```
Git URL / Local Path
    |
    v
git clone (if URL)
    |
    v
load_repository() — walk files, filter by extension
    |
    v
CastChunker (Python: AST split-then-merge) / FixedChunker (line-based)
    |
    v
all-MiniLM-L6-v2 embeddings (local, no API key)
    |
    v
FAISS HNSW index — saved to ~/.cast-rag/indexes/<name>/
    |
    v
Hybrid search: FAISS vector + BM25 keyword, fused with RRF
    |
    v
(optional) Gemini generates answer from top chunks
```

## Files

- `cast_rag.py` — Chunkers, repository loader, BM25 retrieval, ingest/query pipeline, Gemini answer generation.
- `vector_store.py` — FAISS HNSW index, sentence-transformers embeddings, hybrid RRF search.
- `cli.py` — CLI commands: `ingest`, `ask`, `list`, `delete`, `experiment`.
- `main.py` — Entrypoint that delegates to `cli.main()`.
- `test_cast_rag.py` — Unit tests.
- `pyproject.toml` — Project metadata and dependencies.

## Dependencies

- `sentence-transformers` — Local embeddings (all-MiniLM-L6-v2, 384-dim)
- `faiss-cpu` — Vector similarity search (HNSW index)
- `google-generativeai` — Gemini answer generation (optional, needs API key)


uv run python main.py ask --index streamlit --query "how does streamlit run work?" 

export GEMINI_API_KEY=...
uv run cast-rag ask --index streamlit --query "how does streamlit run work?" --answer
