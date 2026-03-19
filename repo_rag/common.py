from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    source: str


def normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def non_ws_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


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
