"""
FastAPI backend wrapping the LangGraph multi-agent graph (graph.py).

This is the piece that used to run inline inside the Streamlit process
(app.py used to `from graph import build_graph` and call it directly).
Now it lives here instead, as its own service -- dockerized via
docker/Dockerfile.backend / docker-compose.yml -- and app.py talks to it
over plain HTTP instead of importing it. app.py itself no longer needs
langgraph, langchain, or any LLM/embedding libraries at all.

One side benefit: the old bug where every Streamlit *process* needed
`@st.cache_resource` to share one graph/store instance across browser
sessions is now moot -- there's exactly one of this backend process
running, so there's exactly one graph/store, full stop.

Run with: uvicorn backend_api:app --host 0.0.0.0 --port 8001
(or via Docker -- see docker/Dockerfile.backend)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from graph import build_graph
from tracing import langfuse_handler

# Built once, when the process starts -- not per-request. Holds the
# LangGraph checkpointer (short-term, per-thread_id history) and the
# long-term memory Store (per-user_id, shared across every conversation),
# for the lifetime of this process.
_graph = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _graph
    _graph = build_graph()
    yield


app = FastAPI(title="Multi-Source Agentic Q&A -- Backend", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Plain liveness check."""
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str
    thread_id: str  # this conversation only -- see state.py's short-term history
    user_id: str = "anonymous"  # long-term identity, across all conversations


class AskResponse(BaseModel):
    route: str
    final_answer: str


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    """Runs one turn through the full graph (guardrail -> load_memory ->
    supervisor -> sql/rag/both/off_topic -> aggregate) and returns the
    route taken plus the final answer -- exactly what app.py used to get
    back from a direct `graph.invoke(...)` call."""
    config = {
        "configurable": {"thread_id": body.thread_id, "user_id": body.user_id},
        "callbacks": [langfuse_handler],
    }
    result = _graph.invoke({"question": body.question}, config)
    return AskResponse(route=result["route"], final_answer=result["final_answer"])


class HistoryItem(BaseModel):
    route: str
    question: str


@app.get("/history/{user_id}", response_model=list[HistoryItem])
def history(user_id: str) -> list[HistoryItem]:
    """This user's long-term Q&A history, across every conversation --
    powers app.py's sidebar. Reads the same Store aggregator_node writes
    to at the end of every turn."""
    items = _graph.store.search((user_id, "qa_history"))
    ordered = sorted(items, key=lambda i: i.created_at)
    return [HistoryItem(route=i.value["route"], question=i.value["question"]) for i in ordered]
