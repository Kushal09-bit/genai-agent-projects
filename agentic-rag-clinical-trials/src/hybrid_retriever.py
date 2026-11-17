"""
day3: hybrid retrieval over the chunks that ingestion_pipeline.py built -
BM25 (lexical) + dense (embedding cosine sim), fused, then MMR for the
final pick so the top-k isn't just 5 near-duplicate chunks of the same
paragraph.

corpus here is small (a handful of papers), so this just loads every chunk
into memory once and scores against the whole thing - no need for an ANN
index on top of chroma for this size.
"""

import re

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    HYBRID_ALPHA,
    MAX_CHUNK_CHARS,
    MMR_LAMBDA,
    RETRIEVAL_CANDIDATE_POOL,
    RETRIEVAL_TOP_K,
)

_WORD_RE = re.compile(r"\w+")


def _tokenize(text):
    return _WORD_RE.findall(text.lower())


def _normalize(scores):
    """min-max to [0, 1] so bm25 (unbounded) and cosine sim (-1..1) can be added together."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _cosine_sim_matrix(query_vec, matrix):
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normalized = matrix / norms
    return normalized @ query_vec


def _truncate(text):
    # keeps what goes back to the model small enough for groq's free-tier
    # TPM cap - see MAX_CHUNK_CHARS in config.py. full chunk's still in
    # chroma untouched, this only trims what retrieve_chunks hands back.
    if len(text) <= MAX_CHUNK_CHARS:
        return text
    return text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."


class HybridRetriever:
    def __init__(self, chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"no '{collection_name}' collection at {chroma_path} - "
                "run `python -m src.ingestion_pipeline` first to build it."
            ) from e

        got = self.collection.get(include=["documents", "metadatas", "embeddings"])
        self.ids = got["ids"]
        self.documents = got["documents"]
        self.metadatas = got["metadatas"]
        self.embeddings = np.array(got["embeddings"])

        if len(self.ids) == 0:
            raise RuntimeError(f"'{collection_name}' collection at {chroma_path} is empty.")

        self._bm25 = BM25Okapi([_tokenize(doc) for doc in self.documents])
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )

    def retrieve(
        self,
        query,
        top_k=RETRIEVAL_TOP_K,
        candidate_pool=RETRIEVAL_CANDIDATE_POOL,
        alpha=HYBRID_ALPHA,
        mmr_lambda=MMR_LAMBDA,
    ):
        """
        returns up to top_k chunks as dicts, ranked by MMR over the fused
        bm25+dense score. each dict has: id, text (truncated to
        MAX_CHUNK_CHARS), document_id, heading_path, score.
        """
        query_embedding = np.array(self._embedder([query])[0])

        dense = _cosine_sim_matrix(query_embedding, self.embeddings)
        bm25 = np.array(self._bm25.get_scores(_tokenize(query)))

        combined = alpha * _normalize(dense) + (1 - alpha) * _normalize(bm25)

        pool_size = min(candidate_pool, len(combined))
        pool_idx = np.argsort(-combined)[:pool_size]

        selected = self._mmr(pool_idx, combined, min(top_k, pool_size), mmr_lambda)

        return [
            {
                "id": self.ids[i],
                "text": _truncate(self.documents[i]),
                "document_id": self.metadatas[i].get("document_id"),
                "heading_path": self.metadatas[i].get("heading_path"),
                "score": round(float(combined[i]), 4),
            }
            for i in selected
        ]

    def _mmr(self, pool_idx, relevance, k, mmr_lambda):
        """greedy MMR: each step, pick whatever maximizes relevance minus
        similarity to what's already been picked, so the result set doesn't
        just fill up with near-duplicate chunks."""
        pool_idx = list(pool_idx)
        selected = [pool_idx.pop(0)]  # top relevance match always goes first

        while pool_idx and len(selected) < k:
            best_idx, best_score = None, -np.inf
            for idx in pool_idx:
                diversity_penalty = max(
                    self._pairwise_sim(idx, s) for s in selected
                )
                mmr_score = mmr_lambda * relevance[idx] - (1 - mmr_lambda) * diversity_penalty
                if mmr_score > best_score:
                    best_idx, best_score = idx, mmr_score
            selected.append(best_idx)
            pool_idx.remove(best_idx)

        return selected

    def _pairwise_sim(self, i, j):
        a, b = self.embeddings[i], self.embeddings[j]
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


_retriever = None


def get_retriever() -> "HybridRetriever":
    """process-wide singleton - the mcp server and linkage_lookup both need
    a retriever, no reason to load the embedding model into memory twice."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
