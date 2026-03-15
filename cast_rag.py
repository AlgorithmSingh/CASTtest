"""cAST-inspired code RAG experiment (self-contained).

This implementation mirrors key ideas from the cAST paper in a lightweight form:
- Syntax-aware chunking using Python AST nodes.
- Recursive split-then-merge chunk construction.
- Chunk budget measured by non-whitespace characters.
- Fixed-size baseline chunker for comparison.

The script builds synthetic Python files, chunks them with both strategies, indexes
chunks with a BM25-like retriever, and reports retrieval metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import ast
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
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
    """Naive baseline: chunk by fixed line count."""

    def __init__(self, lines_per_chunk: int = 8):
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
    """cAST-inspired recursive split-then-merge chunker for Python code.

    - Parse source into AST.
    - Split large nodes recursively.
    - Greedily merge adjacent sibling nodes under max size.
    - Size metric is non-whitespace character count.
    """

    def __init__(self, max_chunk_size: int = 220):
        self.max_chunk_size = max_chunk_size

    def chunk(self, files: Sequence[CodeFile]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for f in files:
            if f.language != "python":
                continue
            src = normalize_whitespace(f.source)
            tree = ast.parse(src)
            lines = src.splitlines()
            node_texts = self._chunk_module_nodes(lines, tree.body)
            for i, txt in enumerate(node_texts):
                cleaned = txt.strip()
                if cleaned:
                    all_chunks.append(Chunk(f"cast::{f.file_id}::{i}", f.file_id, "cast", cleaned))
        return all_chunks

    def _node_span(self, node: ast.AST, lines: Sequence[str]) -> str:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        start = getattr(node, "lineno") - 1
        end = getattr(node, "end_lineno")
        return "\n".join(lines[start:end])

    def _chunk_module_nodes(self, lines: Sequence[str], nodes: Sequence[ast.AST]) -> List[str]:
        return self._split_then_merge(lines, list(nodes))

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

            # If this single node is too large, recurse into children.
            if size > self.max_chunk_size:
                flush_current()
                child_nodes = [n for n in ast.iter_child_nodes(node) if hasattr(n, "lineno") and hasattr(n, "end_lineno")]
                if child_nodes:
                    subchunks = self._split_then_merge(lines, child_nodes)
                    chunks.extend(subchunks)
                else:
                    # Fallback: hard-cut this oversized unit by lines.
                    chunks.extend(self._hard_cut(text))
                continue

            # Greedy sibling merge under size budget.
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


def build_synthetic_code_corpus() -> Tuple[List[CodeFile], List[QueryCase]]:
    files = [
        CodeFile(
            "repo_stats",
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
            "repo_cache",
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
            "repo_auth",
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
        QueryCase("function that computes mean and median statistics", "repo_stats"),
        QueryCase("cache embeddings and reuse vectors to reduce latency", "repo_cache"),
        QueryCase("verify message signature using hmac", "repo_auth"),
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
        rows.append({
            "query": case.query,
            "expected": case.expected_file_id,
            "predicted": predicted_files,
            "hit": hit,
        })

    precision = hits / max(len(cases) * k, 1)  # coarse toy precision@k proxy
    recall = hits / max(len(cases), 1)
    return {
        "hits": hits,
        "total": len(cases),
        "precision_at_k_proxy": precision,
        "recall_at_k": recall,
        "rows": rows,
    }


def run_experiment() -> None:
    files, cases = build_synthetic_code_corpus()
    fixed_chunks = FixedChunker(lines_per_chunk=8).chunk(files)
    cast_chunks = CastChunker(max_chunk_size=220).chunk(files)

    fixed_metrics = evaluate_retrieval(fixed_chunks, cases, k=3)
    cast_metrics = evaluate_retrieval(cast_chunks, cases, k=3)

    print("=== cAST-inspired retrieval experiment ===")
    print(f"Fixed chunks: {len(fixed_chunks)}")
    print(f"cAST chunks:  {len(cast_chunks)}")
    print(
        f"Fixed Recall@3: {fixed_metrics['hits']}/{fixed_metrics['total']} "
        f"({fixed_metrics['recall_at_k']:.2f})"
    )
    print(
        f"cAST Recall@3:  {cast_metrics['hits']}/{cast_metrics['total']} "
        f"({cast_metrics['recall_at_k']:.2f})"
    )

    print("\nPer-query (cAST):")
    for row in cast_metrics["rows"]:
        icon = "✅" if row["hit"] else "❌"
        print(f"{icon} {row['query']} -> {row['predicted']} (expected {row['expected']})")


if __name__ == "__main__":
    run_experiment()
