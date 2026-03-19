from __future__ import annotations

import ast
from typing import List, Sequence

from .common import Chunk, CodeFile, non_ws_len, normalize_whitespace


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
    """cAST-inspired chunker: recursive split-then-merge for Python; line fallback otherwise."""

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
