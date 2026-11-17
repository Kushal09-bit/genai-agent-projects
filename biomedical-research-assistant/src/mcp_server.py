import sys
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

_real_stdout = sys.stdout
sys.stdout = sys.stderr

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pubmed_chunks"
SIMILARITY_THRESHOLD = 1.3  # ChromaDB default distance (L2); lower = more similar
RERANK_CANDIDATES = 25  # how many embedding-similarity candidates to rerank
RERANK_SCORE_THRESHOLD = 0.0  # cross-encoder relevance logit cutoff

# Set up ChromaDB connection once, when the server starts
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=_embedder)

# Reranks embedding-similarity candidates by actual query-passage relevance.
# Embedding similarity alone can rank the true answer below the cutoff when
# it's phrased differently than the query (e.g. a question asking for
# "response rate" won't necessarily rank a chunk phrased as "4-point
# SELENA-SLEDAI score reduction" in its top results, even though that IS
# the answer) — the cross-encoder scores relevance directly instead of
# relying on vector proximity, recovering those cases.
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

sys.stdout = _real_stdout


def retrieve_ranked_chunks(query: str, n_results: int = 8) -> list[dict]:
    """Two-stage retrieval: fetch a wider embedding-similarity candidate
    pool from Chroma, then rerank by relevance with a cross-encoder before
    taking the final top n_results. Shared by the retrieve_chunks MCP tool
    and src/evaluate.py, so eval scores reflect what the agent actually
    retrieves.
    """
    candidate_k = max(n_results * 3, RERANK_CANDIDATES)
    results = _collection.query(
        query_texts=[query], n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    if not results["documents"] or not results["documents"][0]:
        return []

    # Coarse pre-filter: drop candidates the embedding model itself thinks
    # are wildly off-topic before spending reranker compute on them.
    candidates = [
        (doc, meta)
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if dist <= SIMILARITY_THRESHOLD
    ]
    if not candidates:
        return []

    pairs = [[query, doc] for doc, _ in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    print(
        f"[DEBUG] rerank scores for query '{query}': "
        f"{[round(float(s), 3) for _, s in ranked[:n_results]]}",
        file=sys.stderr,
    )

    chunks = []
    for (doc, meta), score in ranked[:n_results]:
        if score < RERANK_SCORE_THRESHOLD:
            break  # sorted descending — everything after is worse too
        chunks.append({
            "text": doc,
            "document_id": meta["document_id"],
            "page_number": meta["page_number"],
            "relevance_score": float(score),
        })
    return chunks


from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pubmed-retrieval")


@mcp.tool()
def retrieve_chunks(query: str, n_results: int = 8) -> str:
    chunks = retrieve_ranked_chunks(query, n_results)
    return json.dumps({"chunks": chunks})


if __name__ == "__main__":
    mcp.run()
