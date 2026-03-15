# cAST-style repository RAG CLI

This repository now includes a real CLI so you can point at a local code repository and run retrieval with a cAST-style chunking pipeline.

## What you can do

- Run a built-in synthetic experiment.
- Query any local repository path with:
  - `cast` strategy (cAST-inspired):
    - Python files: AST split-then-merge chunking with a non-whitespace char budget.
    - Non-Python text/code files: fallback chunking.
  - `fixed` strategy (baseline): fixed line chunks.

## Setup (`uv`)

```bash
uv sync
```

(Uses `pyproject.toml` with no external runtime dependencies.)

## CLI usage

Run experiment:

```bash
uv run python main.py experiment
```

Query a repository:

```bash
uv run python main.py ask --repo /path/to/repo --query "how is auth signature verified" --strategy cast --top-k 5
```

JSON output:

```bash
uv run python main.py ask --repo /path/to/repo --query "cache embeddings" --strategy cast --top-k 5 --json
```

## Files

- `main.py` — CLI entrypoint (`experiment` and `ask` commands).
- `cast_rag.py` — chunkers, repository loader, BM25 retrieval, experiment helpers.
- `test_cast_rag.py` — unit tests including repository query flow.
- `pyproject.toml` — project metadata and script entrypoint.
