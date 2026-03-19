from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .common import BM25Retriever, Chunk, CodeFile, RetrievalHit, non_ws_len, normalize_whitespace


@dataclass(frozen=True)
class RepositoryHandle:
    file_id: str
    language: str
    metadata: str
    source: str


class RecursiveChunker:
    """RLM-inspired chunker that treats files as an environment and recursively zooms in."""

    def __init__(self, max_chunk_size: int = 900, fallback_lines_per_chunk: int = 20):
        self.max_chunk_size = max_chunk_size
        self.fallback_lines_per_chunk = fallback_lines_per_chunk

    def chunk(self, files: Sequence[CodeFile]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for code_file in files:
            handle = self._build_handle(code_file)
            chunks.append(
                Chunk(
                    chunk_id=f"rlm-meta::{code_file.file_id}",
                    file_id=code_file.file_id,
                    strategy="rlm",
                    text=handle.metadata,
                )
            )
            if code_file.language == "python":
                chunks.extend(self._chunk_python(code_file, handle))
            else:
                chunks.extend(self._chunk_fallback(code_file, handle))
        return chunks

    def _build_handle(self, code_file: CodeFile) -> RepositoryHandle:
        src = normalize_whitespace(code_file.source)
        lines = src.splitlines()
        symbols = self._python_symbols(src) if code_file.language == "python" else []
        preview = " ".join(line.strip() for line in lines[:4] if line.strip())[:160]
        symbol_text = ", ".join(symbols[:12]) if symbols else "(no top-level symbols)"
        metadata = (
            f"FILE {code_file.file_id}\n"
            f"LANGUAGE {code_file.language}\n"
            f"LINES {len(lines)}\n"
            f"SYMBOLS {symbol_text}\n"
            f"PREVIEW {preview}"
        )
        return RepositoryHandle(code_file.file_id, code_file.language, metadata, src)

    def _python_symbols(self, source: str) -> List[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        out: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.append(node.name)
        return out

    def _chunk_python(self, code_file: CodeFile, handle: RepositoryHandle) -> List[Chunk]:
        src = handle.source
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return self._chunk_fallback(code_file, handle)

        lines = src.splitlines()
        chunks: List[Chunk] = []
        for index, node in enumerate(tree.body):
            chunks.extend(self._chunk_python_node(code_file, lines, node, parent_path=[code_file.file_id, str(index)]))
        return chunks

    def _chunk_python_node(
        self,
        code_file: CodeFile,
        lines: Sequence[str],
        node: ast.AST,
        parent_path: List[str],
        header_chain: Iterable[str] = (),
    ) -> List[Chunk]:
        text = self._node_span(node, lines).strip()
        if not text:
            return []

        name = getattr(node, "name", node.__class__.__name__)
        header = f"PATH {' > '.join([*header_chain, name])}" if header_chain else f"PATH {name}"
        rendered = f"FILE {code_file.file_id}\n{header}\n{text}".strip()
        size = non_ws_len(rendered)
        if size <= self.max_chunk_size:
            chunk_id = f"rlm::{code_file.file_id}::{'/'.join(parent_path)}"
            return [Chunk(chunk_id=chunk_id, file_id=code_file.file_id, strategy="rlm", text=rendered)]

        child_nodes = [
            child for child in ast.iter_child_nodes(node) if hasattr(child, "lineno") and hasattr(child, "end_lineno")
        ]
        if not child_nodes:
            return self._hard_cut(code_file.file_id, rendered, parent_path)

        out: List[Chunk] = []
        for child_index, child in enumerate(child_nodes):
            out.extend(
                self._chunk_python_node(
                    code_file,
                    lines,
                    child,
                    parent_path=[*parent_path, str(child_index)],
                    header_chain=[*header_chain, name],
                )
            )
        return out

    def _chunk_fallback(self, code_file: CodeFile, handle: RepositoryHandle) -> List[Chunk]:
        lines = handle.source.splitlines()
        out: List[Chunk] = []
        for i in range(0, len(lines), self.fallback_lines_per_chunk):
            block = "\n".join(lines[i : i + self.fallback_lines_per_chunk]).strip()
            if not block:
                continue
            text = f"FILE {code_file.file_id}\nWINDOW {i + 1}-{min(i + self.fallback_lines_per_chunk, len(lines))}\n{block}"
            out.append(Chunk(f"rlm-fallback::{code_file.file_id}::{i // self.fallback_lines_per_chunk}", code_file.file_id, "rlm", text))
        return out

    def _node_span(self, node: ast.AST, lines: Sequence[str]) -> str:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        return "\n".join(lines[getattr(node, 'lineno') - 1 : getattr(node, 'end_lineno')])

    def _hard_cut(self, file_id: str, text: str, parent_path: List[str]) -> List[Chunk]:
        lines = text.splitlines()
        out: List[Chunk] = []
        buffer: List[str] = []
        size = 0
        piece = 0
        for line in lines:
            line_size = non_ws_len(line)
            if buffer and size + line_size > self.max_chunk_size:
                out.append(Chunk(f"rlm::{file_id}::{'/'.join(parent_path)}::{piece}", file_id, "rlm", "\n".join(buffer).strip()))
                buffer = []
                size = 0
                piece += 1
            buffer.append(line)
            size += line_size
        if buffer:
            out.append(Chunk(f"rlm::{file_id}::{'/'.join(parent_path)}::{piece}", file_id, "rlm", "\n".join(buffer).strip()))
        return out


class RecursiveRepositoryRetriever:
    """Two-stage repository search inspired by the RLM paper's environment/recursive decomposition."""

    def __init__(self, files: Sequence[CodeFile], max_chunk_size: int = 900):
        self.files = list(files)
        self.chunker = RecursiveChunker(max_chunk_size=max_chunk_size)
        self.handles = [self.chunker._build_handle(code_file) for code_file in self.files]
        self.file_chunks = [
            Chunk(f"rlm-file::{handle.file_id}", handle.file_id, "rlm", handle.metadata) for handle in self.handles
        ]
        self.file_ranker = BM25Retriever(self.file_chunks)

    def retrieve(self, query: str, top_k: int = 5, per_file_depth: int = 4) -> List[RetrievalHit]:
        candidate_files = self.file_ranker.retrieve(query, top_k=max(top_k, per_file_depth))
        ranked: List[RetrievalHit] = []
        seen_chunk_ids: set[str] = set()

        for file_chunk, file_score in candidate_files:
            code_file = next(f for f in self.files if f.file_id == file_chunk.file_id)
            local_chunks = self.chunker.chunk([code_file])
            chunk_ranker = BM25Retriever(local_chunks)
            for chunk, local_score in chunk_ranker.retrieve(query, top_k=per_file_depth):
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                combined = round(file_score * 0.35 + local_score * 0.65, 6)
                ranked.append(RetrievalHit(chunk=chunk, score=combined, source="rlm-recursive"))

        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]


def summarize_repository_environment(files: Sequence[CodeFile]) -> str:
    handles = [RecursiveChunker()._build_handle(code_file) for code_file in files]
    return "\n\n".join(handle.metadata for handle in handles)


def recursive_query_repository(repo_path: str, query: str, top_k: int = 5) -> dict[str, object]:
    from .common import load_repository

    files = load_repository(repo_path)
    retriever = RecursiveRepositoryRetriever(files)
    results = retriever.retrieve(query, top_k=top_k)
    return {
        "query": query,
        "strategy": "rlm",
        "retrieval": "recursive environment search",
        "num_files": len(files),
        "num_chunks": len(RecursiveChunker().chunk(files)),
        "environment": summarize_repository_environment(files[: min(len(files), 8)]),
        "results": [
            {
                "file_id": hit.chunk.file_id,
                "chunk_id": hit.chunk.chunk_id,
                "score": round(hit.score, 4),
                "source": hit.source,
                "text": hit.chunk.text,
            }
            for hit in results
        ],
    }


def recursive_query_chunks(chunks: Sequence[Chunk], query: str, top_k: int = 5) -> List[RetrievalHit]:
    ranker = BM25Retriever(chunks)
    return [RetrievalHit(chunk=chunk, score=score, source="rlm-recursive") for chunk, score in ranker.retrieve(query, top_k=top_k)]
