"""cAST-style retrieval utilities and runnable experiment/CLI helpers.

Core features:
- Fixed-size baseline chunking.
- cAST-inspired split-then-merge chunking for Python via stdlib AST.
- Repository ingestion for real local codebases.
- BM25-like retrieval over generated chunks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import ast
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".m",
    ".scala",
    ".kt",
    ".swift",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
SKIP_DIRS = {".git", "node_modules", "dist", "build", "target", "venv", ".venv", "__pycache__"}
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "to",
    "of",
    "and",
    "in",
    "for",
    "on",
    "how",
    "what",
    "does",
    "do",
    "with",
    "by",
    "that",
}


def normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def non_ws_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


@dataclass(frozen=True)
class CodeFile:
    file_id: str
    language: str
    source: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_id: str
    strategy: str
    text: str


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_file_id: str


class FixedChunker:
    def __init__(self, lines_per_chunk: int = 16):
        self.lines_per_chunk = lines_per_chunk

    def chunk(self, files: Sequence[CodeFile]) -> List[Chunk]:
        out: List[Chunk] = []
        for f in files:
            lines = normalize_whitespace(f.source).splitlines()
            part = 0
            for i in range(0, len(lines), self.lines_per_chunk):
                block = "\n".join(lines[i : i + self.lines_per_chunk]).strip()
                if block:
                    out.append(Chunk(f"fixed::{f.file_id}::{part}", f.file_id, "fixed", block))
                    part += 1
        return out


class CastChunker:
    """cAST-inspired chunker: recursive split-then-merge for Python; char-budget fallback otherwise."""

    def __init__(self, max_chunk_size: int = 1200, fallback_lines_per_chunk: int = 24):
        self.max_chunk_size = max_chunk_size
        self.fallback_lines_per_chunk = fallback_lines_per_chunk

    def chunk(self, files: Sequence[CodeFile]) -> List[Chunk]:
        out: List[Chunk] = []
        for f in files:
            if f.language == "python":
                out.extend(self._chunk_python_file(f))
            else:
                out.extend(self._chunk_fallback_file(f))
        return out

    def _chunk_python_file(self, f: CodeFile) -> List[Chunk]:
        src = normalize_whitespace(f.source)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return self._chunk_fallback_file(f)

        lines = src.splitlines()
        node_texts = self._split_then_merge(lines, list(tree.body))
        chunks: List[Chunk] = []
        for i, txt in enumerate(node_texts):
            cleaned = txt.strip()
            if cleaned:
                chunks.append(Chunk(f"cast::{f.file_id}::{i}", f.file_id, "cast", cleaned))
        return chunks

    def _chunk_fallback_file(self, f: CodeFile) -> List[Chunk]:
        lines = normalize_whitespace(f.source).splitlines()
        out: List[Chunk] = []
        part = 0
        for i in range(0, len(lines), self.fallback_lines_per_chunk):
            block = "\n".join(lines[i : i + self.fallback_lines_per_chunk]).strip()
            if block:
                out.append(Chunk(f"cast-fallback::{f.file_id}::{part}", f.file_id, "cast", block))
                part += 1
        return out

    def _node_span(self, node: ast.AST, lines: Sequence[str]) -> str:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        start = getattr(node, "lineno") - 1
        end = getattr(node, "end_lineno")
        return "\n".join(lines[start:end])

    def _split_then_merge(self, lines: Sequence[str], nodes: List[ast.AST]) -> List[str]:
        chunks: List[str] = []
        current_parts: List[str] = []
        current_size = 0

        def flush_current() -> None:
            nonlocal current_parts, current_size
            if current_parts:
                chunks.append("\n".join(current_parts).strip())
                current_parts = []
                current_size = 0

        for node in nodes:
            text = self._node_span(node, lines).strip()
            if not text:
                continue
            size = non_ws_len(text)

            if size > self.max_chunk_size:
                flush_current()
                child_nodes = [
                    n for n in ast.iter_child_nodes(node) if hasattr(n, "lineno") and hasattr(n, "end_lineno")
                ]
                if child_nodes:
                    chunks.extend(self._split_then_merge(lines, child_nodes))
                else:
                    chunks.extend(self._hard_cut(text))
                continue

            if current_size + size > self.max_chunk_size:
                flush_current()
            current_parts.append(text)
            current_size += size

        flush_current()
        return chunks

    def _hard_cut(self, text: str) -> List[str]:
        lines = text.splitlines()
        out: List[str] = []
        buf: List[str] = []
        size = 0
        for ln in lines:
            ln_size = non_ws_len(ln)
            if buf and size + ln_size > self.max_chunk_size:
                out.append("\n".join(buf).strip())
                buf = []
                size = 0
            buf.append(ln)
            size += ln_size
        if buf:
            out.append("\n".join(buf).strip())
        return out


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.tokens = {c.chunk_id: tokenize(c.text) for c in self.chunks}
        self.tf = {cid: Counter(toks) for cid, toks in self.tokens.items()}
        self.length = {cid: len(toks) for cid, toks in self.tokens.items()}
        self.avg_len = sum(self.length.values()) / max(len(self.length), 1)
        self.df = Counter()
        for toks in self.tokens.values():
            self.df.update(set(toks))

    def score(self, query_tokens: Iterable[str], chunk_id: str, k1: float = 1.2, b: float = 0.75) -> float:
        tf = self.tf[chunk_id]
        dl = self.length[chunk_id]
        n = max(len(self.chunks), 1)
        s = 0.0
        for term in query_tokens:
            n_q = self.df.get(term, 0)
            idf = math.log(1 + (n - n_q + 0.5) / (n_q + 0.5))
            f = tf.get(term, 0)
            denom = f + k1 * (1 - b + b * dl / max(self.avg_len, 1e-8))
            if denom > 0:
                s += idf * (f * (k1 + 1)) / denom
        return s

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q = tokenize(query)
        scored: List[Tuple[Chunk, float]] = []
        for c in self.chunks:
            sc = self.score(q, c.chunk_id)
            if sc > 0:
                scored.append((c, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def detect_language(path: Path) -> str:
    return "python" if path.suffix.lower() == ".py" else "text"


def load_repository(repo_path: str, max_file_bytes: int = 300_000) -> List[CodeFile]:
    root = Path(repo_path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {repo_path}")

    files: List[CodeFile] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if p.stat().st_size > max_file_bytes:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue

        rel = str(p.relative_to(root))
        files.append(CodeFile(file_id=rel, language=detect_language(p), source=text))

    return files


def build_chunks_for_repo(repo_path: str, strategy: str = "cast") -> List[Chunk]:
    files = load_repository(repo_path)
    if strategy == "fixed":
        return FixedChunker().chunk(files)
    return CastChunker().chunk(files)


def query_repository(repo_path: str, query: str, strategy: str = "cast", top_k: int = 5) -> Dict[str, object]:
    chunks = build_chunks_for_repo(repo_path, strategy=strategy)
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


INDEX_ROOT = Path.home() / ".cast-rag" / "indexes"
REPO_ROOT = Path.home() / ".cast-rag" / "repos"


def ingest_repository(
    source: str,
    name: str | None = None,
    strategy: str = "cast",
) -> Dict[str, object]:
    """Clone (if git URL) or read a local repo, chunk it, embed it, and save a FAISS index.

    Returns metadata about the ingested index.
    """
    import subprocess

    from vector_store import EmbeddingModel, build_index

    # Determine if source is a git URL or local path
    is_url = source.startswith("http://") or source.startswith("https://") or source.startswith("git@")

    if is_url:
        if name is None:
            # Derive name from URL: https://github.com/user/repo.git -> repo
            name = source.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        clone_dir = REPO_ROOT / name
        if clone_dir.exists():
            # Pull latest
            subprocess.run(["git", "-C", str(clone_dir), "pull", "--ff-only"], check=False)
        else:
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", source, str(clone_dir)], check=True)
        repo_path = str(clone_dir)
    else:
        repo_path = str(Path(source).resolve())
        if name is None:
            name = Path(repo_path).name

    # Load and chunk
    chunks = build_chunks_for_repo(repo_path, strategy=strategy)
    if not chunks:
        raise ValueError(f"No chunks produced from {repo_path}")

    # Embed and build FAISS index
    print(f"Embedding {len(chunks)} chunks with all-MiniLM-L6-v2...")
    embedder = EmbeddingModel()
    vector_index, _ = build_index(chunks, embedder)

    # Save index
    index_dir = str(INDEX_ROOT / name)
    vector_index.save(index_dir)

    # Save metadata
    import json as _json

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


def query_index(
    name: str,
    query: str,
    top_k: int = 5,
) -> Dict[str, object]:
    """Query a previously ingested index using hybrid search (vector + BM25 + RRF)."""
    import json as _json

    from vector_store import EmbeddingModel, VectorIndex, hybrid_search

    index_dir = INDEX_ROOT / name
    if not index_dir.exists():
        raise ValueError(f"No index found for '{name}'. Run 'ingest' first.")

    meta = _json.loads((index_dir / "meta.json").read_text())
    vector_index = VectorIndex.load(str(index_dir))
    embedder = EmbeddingModel()

    results = hybrid_search(
        query=query,
        chunks=vector_index.chunks,
        vector_index=vector_index,
        embedder=embedder,
        top_k=top_k,
    )

    return {
        "query": query,
        "index": name,
        "strategy": meta.get("strategy", "cast"),
        "num_files": meta.get("num_files", 0),
        "num_chunks": meta.get("num_chunks", 0),
        "retrieval": "hybrid (vector + BM25 + RRF)",
        "results": [
            {
                "file_id": r.chunk.file_id,
                "chunk_id": r.chunk.chunk_id,
                "score": r.score,
                "source": r.source,
                "text": r.chunk.text,
            }
            for r in results
        ],
    }


def generate_answer(query: str, chunks: List[Dict[str, object]], api_key: str, model: str = "gemini-2.0-flash") -> str:
    """Send retrieved chunks + query to Gemini and return the generated answer."""
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


def build_synthetic_code_corpus() -> Tuple[List[CodeFile], List[QueryCase]]:
    files = [
        CodeFile(
            "repo_stats.py",
            "python",
            '''
from statistics import mean, median

def normalize(values):
    total = sum(values)
    if total == 0:
        return values
    return [v / total for v in values]


def compute_stats(values):
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return {"mean": 0, "median": 0, "count": 0}
    return {
        "mean": mean(cleaned),
        "median": median(cleaned),
        "count": len(cleaned),
    }
''',
        ),
        CodeFile(
            "repo_cache.py",
            "python",
            '''
class EmbeddingCache:
    def __init__(self):
        self._items = {}

    def get(self, key):
        return self._items.get(key)

    def set(self, key, value):
        self._items[key] = value


def get_or_embed(cache, model, text):
    cached = cache.get(text)
    if cached is not None:
        return cached
    vec = model.embed(text)
    cache.set(text, vec)
    return vec
''',
        ),
        CodeFile(
            "repo_auth.py",
            "python",
            '''
import hmac
import hashlib

def sign_message(secret, payload):
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def verify_signature(secret, payload, signature):
    expected = sign_message(secret, payload)
    return hmac.compare_digest(expected, signature)
''',
        ),
    ]

    cases = [
        QueryCase("function that computes mean and median statistics", "repo_stats.py"),
        QueryCase("cache embeddings and reuse vectors to reduce latency", "repo_cache.py"),
        QueryCase("verify message signature using hmac", "repo_auth.py"),
    ]
    return files, cases


def evaluate_retrieval(chunks: Sequence[Chunk], cases: Sequence[QueryCase], k: int = 3) -> Dict[str, object]:
    retriever = BM25Retriever(chunks)
    hits = 0
    rows = []
    for case in cases:
        preds = retriever.retrieve(case.query, top_k=k)
        predicted_files = [ch.file_id for ch, _ in preds]
        hit = case.expected_file_id in predicted_files
        hits += int(hit)
        rows.append({"query": case.query, "expected": case.expected_file_id, "predicted": predicted_files, "hit": hit})

    return {"hits": hits, "total": len(cases), "recall_at_k": hits / max(len(cases), 1), "rows": rows}


def run_experiment() -> None:
    files, cases = build_synthetic_code_corpus()
    fixed_chunks = FixedChunker(lines_per_chunk=8).chunk(files)
    cast_chunks = CastChunker(max_chunk_size=220).chunk(files)

    fixed_metrics = evaluate_retrieval(fixed_chunks, cases, k=3)
    cast_metrics = evaluate_retrieval(cast_chunks, cases, k=3)

    print("=== cAST-inspired retrieval experiment ===")
    print(f"Fixed chunks: {len(fixed_chunks)}")
    print(f"cAST chunks:  {len(cast_chunks)}")
    print(f"Fixed Recall@3: {fixed_metrics['hits']}/{fixed_metrics['total']} ({fixed_metrics['recall_at_k']:.2f})")
    print(f"cAST Recall@3:  {cast_metrics['hits']}/{cast_metrics['total']} ({cast_metrics['recall_at_k']:.2f})")


if __name__ == "__main__":
    run_experiment()
