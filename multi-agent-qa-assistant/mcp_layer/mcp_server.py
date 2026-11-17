"""
The tool server: owns the SQL database and the biomedical vector store,
and exposes three tools two different ways from the same process --

  1. MCP, at http://<host>:8000/mcp -- what agents/sql_node.py and
     agents/rag_node.py actually use, via mcp_layer/mcp_client.py.
  2. Plain REST, at http://<host>:8000/tools/... -- a FastAPI wrapper
     added on top so the same tools can be hit directly (curl, Postman,
     a browser) without speaking the MCP protocol. The agents don't use
     these routes; they're an extra access point, not a replacement.

Both the MCP app and the REST routes are mounted on one FastAPI app and
served by one uvicorn process on one port -- see the bottom of this file.
"""

import sys
import os
import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder


_real_stdout = sys.stdout
sys.stdout = sys.stderr

# Two directories up: mcp_layer/mcp_server.py -> mcp_layer/ -> repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_REPO_ROOT, "chroma_db")
COLLECTION_NAME = "pubmed_chunks"

SIMILARITY_THRESHOLD = 1.8
RERANK_CANDIDATES = 25
RERANK_SCORE_THRESHOLD = 0.0

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_collection = _chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=_embedder)
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

sys.stdout = _real_stdout


def retrieve_ranked_chunks(query: str, n_results: int = 8) -> list[dict]:
    """Two-stage retrieval: a fast Chroma similarity search over the whole
    corpus, narrowed down to n_results by a slower, more accurate
    cross-encoder rerank of the top candidates."""
    candidate_k = max(n_results * 3, RERANK_CANDIDATES)
    results = _collection.query(
        query_texts=[query], n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    if not results["documents"] or not results["documents"][0]:
        return []

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

    chunks = []
    for (doc, meta), score in ranked[:n_results]:
        if score < RERANK_SCORE_THRESHOLD:
            break
        chunks.append({
            "text": doc,
            "document_id": meta["document_id"],
            "page_number": meta["page_number"],
            "relevance_score": float(score),
        })
    return chunks


# sql_db/ and sql_guardrails.py live at the repo root, but this file is
# launched as `python mcp_layer/mcp_server.py`, so Python only puts
# mcp_layer/ itself on sys.path -- not the repo root. Add the repo root
# explicitly so the two imports below resolve.
sys.path.insert(0, _REPO_ROOT)

from sqlalchemy import text as sql_text
from sql_db.metadata_extractor import MetadataExtractor
from sql_guardrails import SQLValidator, OutputGuardrail as SQLOutputGuardrail, ResultSanitizer

DB_URL = os.getenv(
    "CHINOOK_DATABASE_URL",
    "sqlite:///" + os.path.join(_REPO_ROOT, "sql_db", "data", "chinook.sqlite"),
)


_extractor = MetadataExtractor(DB_URL)
_sql_metadata = _extractor.extract_all()
_sql_engine = _extractor.engine
_sql_validator = SQLValidator(allowed_tables=set(_sql_metadata.keys()))
_sql_output_guardrail = SQLOutputGuardrail(permitted_tables=set(_sql_metadata.keys()))
_result_sanitizer = ResultSanitizer()


def _schema_json(metadata: dict) -> str:
    return json.dumps(
        {t: [{"name": c.name, "type": c.type} for c in m.columns] for t, m in metadata.items()},
        default=str,
    )


def _lov_json(metadata: dict) -> str:
    return json.dumps(
        {f"{t}.{c.name}": c.sample_values for t, m in metadata.items() for c in m.columns if c.sample_values},
        default=str,
    )


def _execute_sql_impl(sql: str) -> str:
    
    valid, reason = _sql_validator.validate(sql)
    if not valid:
        return f"INVALID: {reason}"

    valid, reason, sql = _sql_output_guardrail.check(sql)
    if not valid:
        return f"INVALID: {reason}"

    try:
        with _sql_engine.connect() as conn:
            result = conn.execute(sql_text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
    except Exception as e:
        return f"EXECUTION_ERROR: {e}"

    columns, rows, note = _result_sanitizer.sanitize(columns, rows)

    payload = {
        "sql": sql,
        "columns": columns,
        "rows": [list(r) for r in rows[:20]],
        "row_count": len(rows),
        "note": note,
    }
    return json.dumps(payload, default=str)


from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "multi-source-qa",
    host=os.getenv("MCP_SERVER_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_SERVER_PORT", "8000")),
)


@mcp.tool()
def retrieve_chunks(query: str, n_results: int = 8) -> str:
   
    chunks = retrieve_ranked_chunks(query, n_results)
    return json.dumps({"chunks": chunks})


@mcp.tool()
def get_schema_and_samples() -> str:
    
    schema = _schema_json(_sql_metadata)
    samples = _lov_json(_sql_metadata)
    return f"Schema:\n{schema}\n\nExample values for key columns:\n{samples}"


@mcp.tool()
def execute_sql(sql: str) -> str:
    
    return _execute_sql_impl(sql)


# ---------------------------------------------------------------------------
# FastAPI wrapper: same tools, plain REST, for anyone who wants to call them
# directly instead of through the MCP protocol. The agents (sql_node.py,
# rag_node.py) are untouched by this -- they still go through mcp_client.py
# and MCP, exactly as before.
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


@asynccontextmanager
async def _lifespan(_: "FastAPI"):
    
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Multi-Source QA -- Tool Server", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    """Plain liveness check -- confirms the container is up and both the
    MCP app and this REST layer are serving requests."""
    return {"status": "ok"}


class RetrieveChunksRequest(BaseModel):
    query: str
    n_results: int = 8


@app.post("/tools/retrieve_chunks")
def rest_retrieve_chunks(body: RetrieveChunksRequest) -> dict:
    """REST equivalent of the retrieve_chunks MCP tool above."""
    return {"chunks": retrieve_ranked_chunks(body.query, body.n_results)}


@app.get("/tools/schema")
def rest_get_schema_and_samples() -> dict:
    """REST equivalent of the get_schema_and_samples MCP tool above."""
    return {
        "schema": json.loads(_schema_json(_sql_metadata)),
        "sample_values": json.loads(_lov_json(_sql_metadata)),
    }


class ExecuteSQLRequest(BaseModel):
    sql: str


@app.post("/tools/execute_sql")
def rest_execute_sql(body: ExecuteSQLRequest) -> dict:
    """REST equivalent of the execute_sql MCP tool above. Validation
    failures and execution errors come back as a 400 with the reason in
    `detail`, rather than the "INVALID: ..." / "EXECUTION_ERROR: ..."
    prefixed strings the MCP tool returns (that convention exists so an
    LLM can read and self-correct from plain text; a REST client is
    better served by a real HTTP status code)."""
    result = _execute_sql_impl(body.sql)
    if result.startswith("INVALID:") or result.startswith("EXECUTION_ERROR:"):
        raise HTTPException(status_code=400, detail=result)
    return json.loads(result)


# Mounted LAST and on purpose: Starlette matches routes in registration
# order, and a Mount at "/" matches every path. If this were registered
# before the /health and /tools/... routes above, it would swallow every
# request (including /health) before FastAPI ever reached those routes,
# and callers would see the MCP sub-app's own 404 instead. Registering it
# last means the specific routes above match first, and only requests that
# don't match any of them (i.e. the actual /mcp traffic) fall through to it.
app.mount("/", mcp.streamable_http_app())


if __name__ == "__main__":
    
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("MCP_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_SERVER_PORT", "8000")),
    )
