"""FAISS vector store with sentence-transformers embeddings and hybrid BM25+vector search.

Uses all-MiniLM-L6-v2 for local embeddings (no API key needed) and FAISS HNSW
for fast approximate nearest-neighbor search.  Hybrid retrieval fuses BM25 keyword
scores with vector similarity via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from cast_rag import BM25Retriever, Chunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RRF_K = 60  # RRF smoothing constant


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    source: str  # "vector", "bm25", or "hybrid"


class EmbeddingModel:
    """Wraps sentence-transformers for encoding text to vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]


class VectorIndex:
    """FAISS HNSW index with metadata for chunk lookup."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 50
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), top_k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            similarity = 1.0 / (1.0 + max(float(dist), 0.0))
            results.append(SearchResult(chunk=self.chunks[idx], score=similarity, source="vector"))
        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str) -> "VectorIndex":
        path = Path(directory)
        idx = cls.__new__(cls)
        idx.index = faiss.read_index(str(path / "index.faiss"))
        idx.dimension = idx.index.d
        with open(path / "chunks.pkl", "rb") as f:
            idx.chunks = pickle.load(f)
        return idx


def hybrid_search(
    query: str,
    chunks: List[Chunk],
    vector_index: VectorIndex,
    embedder: EmbeddingModel,
    top_k: int = 5,
) -> List[SearchResult]:
    """Hybrid retrieval: FAISS vector search + BM25 keyword search fused with RRF."""
    k_retrieve = max(10, top_k * 2)

    # Vector search
    query_vec = embedder.embed_query(query)
    vector_results = vector_index.search(query_vec, top_k=k_retrieve)

    # BM25 search
    bm25 = BM25Retriever(chunks)
    bm25_hits = bm25.retrieve(query, top_k=k_retrieve)
    bm25_results = [
        SearchResult(chunk=c, score=s, source="bm25") for c, s in bm25_hits
    ]

    # RRF fusion
    fused_scores: Dict[str, float] = {}
    canonical: Dict[str, SearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk.chunk_id
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        canonical.setdefault(cid, result)

    for rank, result in enumerate(bm25_results, start=1):
        cid = result.chunk.chunk_id
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        canonical.setdefault(cid, result)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    return [
        SearchResult(
            chunk=canonical[cid].chunk,
            score=round(fused_scores[cid], 6),
            source="hybrid",
        )
        for cid in ranked_ids[:top_k]
    ]


def build_index(
    chunks: List[Chunk],
    embedder: Optional[EmbeddingModel] = None,
) -> Tuple[VectorIndex, EmbeddingModel]:
    """Embed chunks and build a FAISS HNSW index."""
    if embedder is None:
        embedder = EmbeddingModel()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    vi = VectorIndex(dimension=embeddings.shape[1])
    vi.add(chunks, embeddings)
    return vi, embedder
