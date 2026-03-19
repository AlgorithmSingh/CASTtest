from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .cast import CastChunker, FixedChunker
from .common import Chunk, build_synthetic_code_corpus, evaluate_retrieval, load_repository
from .rlm import RecursiveChunker, recursive_query_repository

INDEX_ROOT = Path.home() / ".cast-rag" / "indexes"
REPO_ROOT = Path.home() / ".cast-rag" / "repos"


def build_chunks_for_repo(repo_path: str, strategy: str = "cast") -> List[Chunk]:
    files = load_repository(repo_path)
    if strategy == "fixed":
        return FixedChunker().chunk(files)
    if strategy == "rlm":
        return RecursiveChunker().chunk(files)
    return CastChunker().chunk(files)


def query_repository(repo_path: str, query: str, strategy: str = "cast", top_k: int = 5) -> Dict[str, object]:
    if strategy == "rlm":
        return recursive_query_repository(repo_path, query, top_k=top_k)

    chunks = build_chunks_for_repo(repo_path, strategy=strategy)
    from .common import BM25Retriever

    retriever = BM25Retriever(chunks)
    rows = retriever.retrieve(query, top_k=top_k)
    return {
        "query": query,
        "strategy": strategy,
        "num_files": len({c.file_id for c in chunks}),
        "num_chunks": len(chunks),
        "results": [
            {"file_id": c.file_id, "chunk_id": c.chunk_id, "score": round(score, 4), "text": c.text}
            for c, score in rows
        ],
    }


def ingest_repository(source: str, name: str | None = None, strategy: str = "cast") -> Dict[str, object]:
    import json as _json
    import subprocess

    from vector_store import EmbeddingModel, build_index

    is_url = source.startswith("http://") or source.startswith("https://") or source.startswith("git@")

    if is_url:
        if name is None:
            name = source.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        clone_dir = REPO_ROOT / name
        if clone_dir.exists():
            subprocess.run(["git", "-C", str(clone_dir), "pull", "--ff-only"], check=False)
        else:
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", source, str(clone_dir)], check=True)
        repo_path = str(clone_dir)
    else:
        repo_path = str(Path(source).resolve())
        if name is None:
            name = Path(repo_path).name

    chunks = build_chunks_for_repo(repo_path, strategy=strategy)
    if not chunks:
        raise ValueError(f"No chunks produced from {repo_path}")

    print(f"Embedding {len(chunks)} chunks with all-MiniLM-L6-v2...")
    embedder = EmbeddingModel()
    vector_index, _ = build_index(chunks, embedder)

    index_dir = str(INDEX_ROOT / name)
    vector_index.save(index_dir)

    meta = {
        "name": name,
        "source": source,
        "repo_path": repo_path,
        "strategy": strategy,
        "num_files": len({c.file_id for c in chunks}),
        "num_chunks": len(chunks),
    }
    (Path(index_dir) / "meta.json").write_text(_json.dumps(meta, indent=2))
    return meta


def query_index(name: str, query: str, top_k: int = 5) -> Dict[str, object]:
    import json as _json

    from vector_store import EmbeddingModel, VectorIndex, hybrid_search

    index_dir = INDEX_ROOT / name
    if not index_dir.exists():
        raise ValueError(f"No index found for '{name}'. Run 'ingest' first.")

    meta = _json.loads((index_dir / "meta.json").read_text())
    vector_index = VectorIndex.load(str(index_dir))
    embedder = EmbeddingModel()
    results = hybrid_search(query=query, chunks=vector_index.chunks, vector_index=vector_index, embedder=embedder, top_k=top_k)

    return {
        "query": query,
        "index": name,
        "strategy": meta.get("strategy", "cast"),
        "num_files": meta.get("num_files", 0),
        "num_chunks": meta.get("num_chunks", 0),
        "retrieval": "hybrid (vector + BM25 + RRF)",
        "results": [
            {
                "file_id": result.chunk.file_id,
                "chunk_id": result.chunk.chunk_id,
                "score": result.score,
                "source": result.source,
                "text": result.chunk.text,
            }
            for result in results
        ],
    }


def generate_answer(query: str, chunks: List[Dict[str, object]], api_key: str, model: str = "gemini-2.5-flash") -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"--- Chunk {i} (file: {chunk['file_id']}, score: {chunk['score']}) ---\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful code assistant. Use the following code chunks retrieved from a repository "
        "to answer the user's question. Reference specific files and code when possible. "
        "If the chunks don't contain enough information, say so.\n\n"
        f"## Retrieved code chunks\n\n{context}\n\n"
        f"## Question\n\n{query}"
    )

    client = genai.GenerativeModel(model)
    response = client.generate_content(prompt)
    return response.text


def run_experiment() -> None:
    files, cases = build_synthetic_code_corpus()
    fixed_chunks = FixedChunker(lines_per_chunk=8).chunk(files)
    cast_chunks = CastChunker(max_chunk_size=220).chunk(files)
    rlm_chunks = RecursiveChunker(max_chunk_size=220).chunk(files)

    fixed_metrics = evaluate_retrieval(fixed_chunks, cases, k=3)
    cast_metrics = evaluate_retrieval(cast_chunks, cases, k=3)
    rlm_metrics = evaluate_retrieval(rlm_chunks, cases, k=3)

    print("=== Repository retrieval experiment ===")
    print(f"Fixed chunks: {len(fixed_chunks)}")
    print(f"cAST chunks:  {len(cast_chunks)}")
    print(f"RLM chunks:   {len(rlm_chunks)}")
    print(f"Fixed Recall@3: {fixed_metrics['hits']}/{fixed_metrics['total']} ({fixed_metrics['recall_at_k']:.2f})")
    print(f"cAST Recall@3:  {cast_metrics['hits']}/{cast_metrics['total']} ({cast_metrics['recall_at_k']:.2f})")
    print(f"RLM Recall@3:   {rlm_metrics['hits']}/{rlm_metrics['total']} ({rlm_metrics['recall_at_k']:.2f})")
