"""
day6: HTTP API in front of the agent - /query, /health, basic per-IP rate
limiting. this is the "backend" container in the docker plan (agent + mcp
client) - streamlit (app.py) is a separate process/container that just
calls this over HTTP, same as any other client would.

the agent gets built ONCE at startup (it spawns the mcp server subprocess
and loads the embedding model - too expensive to redo per request) and
reused for every request after that. thread_id is how a client keeps a
conversation going across multiple /query calls - pass back whatever the
first response gave you.

run with: uvicorn src.api:app --host 0.0.0.0 --port 8000
(needs chroma_db/ already built, same as the REPL - see src/ingestion_pipeline.py)
"""

import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.agent.agent import _flush_langfuse, ask, build_agent

RATE_LIMIT = 10  # requests
RATE_WINDOW = 60  # seconds - simple fixed-ish window per client ip, in-memory
_hits = defaultdict(deque)

_agent = None  # set once in lifespan(), reused for every request


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await build_agent()
    yield
    # langfuse batches traces in the background - a container that just
    # exits can beat that flush to the punch and drop the last few, so
    # flush explicitly on shutdown instead of relying on it. no-op if
    # langfuse tracing was never turned on (see _get_langfuse_handler).
    _flush_langfuse()
    _agent = None


app = FastAPI(title="agenticrag API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    thread_id: str | None = None  # omit to start a new conversation


class QueryResponse(BaseModel):
    answer: str
    thread_id: str  # echoed back - reuse it on your next call to keep context


def _rate_limit_ok(client_ip: str) -> bool:
    now = time.time()
    hits = _hits[client_ip]
    while hits and now - hits[0] > RATE_WINDOW:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return False
    hits.append(now)
    return True


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": _agent is not None}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"rate limit: max {RATE_LIMIT} requests per {RATE_WINDOW}s",
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question can't be empty")

    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        answer = await ask(_agent, req.question, thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent failed: {e}") from e

    return QueryResponse(answer=answer, thread_id=thread_id)
