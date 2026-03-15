# cAST-inspired Code RAG experiment

This repository contains a compact experiment inspired by the paper **"cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree"** and the referenced implementation repository.

## What is implemented

A simple side-by-side retrieval experiment over synthetic Python repositories:

- **Baseline (`FixedChunker`)**: fixed-size line chunking.
- **cAST-inspired (`CastChunker`)**:
  - Parse source with Python AST.
  - Apply **recursive split-then-merge** over AST sibling nodes.
  - Enforce chunk budget by **non-whitespace character count**.

Then both chunk sets are indexed with a BM25-like retriever and evaluated with toy **Recall@3**.

## Why this aligns better with cAST

Compared to earlier versions, this now explicitly mirrors the key algorithmic ideas from the paper text:

1. **Syntax-aware boundaries** from AST nodes.
2. **Split-then-merge recursion** instead of only flat splitting.
3. **Character-density chunk budget** (non-whitespace chars), not line count.
4. **Direct fixed-size baseline comparison** in one script.

## Files

- `cast_rag.py` — chunkers, retrieval index, synthetic corpus, evaluator, runnable experiment.
- `test_cast_rag.py` — tests for chunk budget constraints and retrieval behavior.

## Run

```bash
python cast_rag.py
python -m unittest -v
```

## Notes

- I attempted `git pull --ff-only`, but this branch has no upstream tracking configured in the current environment.
- The code is self-contained and dependency-free (standard library only).
