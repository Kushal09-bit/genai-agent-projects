import os
import re

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from src import aact_db
from src.config import (
    AGENT_MODEL,
    GEMINI_MODEL,
    MCP_SERVER_BIND_HOST,
    MCP_SERVER_PORT,
    MCP_TRANSPORT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
)
from src.hybrid_retriever import get_retriever

mcp = FastMCP("agenticrag-tools", host=MCP_SERVER_BIND_HOST, port=MCP_SERVER_PORT)


# ---------- day3: retrieval ----------

@mcp.tool()
def retrieve_chunks(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """
    Hybrid (BM25 + dense embeddings, fused, then MMR-diversified) retrieval
    over the ingested biomedical papers. Returns the top_k most relevant
    chunks, each with its text, source document, section heading path, and
    a relevance score.
    """
    return get_retriever().retrieve(query, top_k=top_k)


# ---------- day4: structured AACT tools ----------

@mcp.tool()
@aact_db.cached
def schema_tool(table_name: str = "") -> list[dict]:
    """
    Look up AACT's real table/column structure - call this before writing
    SQL for a table you haven't already checked this conversation, don't
    guess column names.

    With no table_name: lists every table in the ctgov schema.
    With a table_name: lists that table's columns and their data types.
    """
    if table_name:
        rows = aact_db.list_columns(table_name)
        return rows if rows else [{"error": f"no table '{table_name}' in the ctgov schema"}]
    return aact_db.list_tables()


@mcp.tool()
@aact_db.cached
def values_tool(table_name: str, column_name: str, limit: int = 20) -> list:
    """
    Look up what a column's real values look like - call this before
    filtering on a categorical column (like overall_status or phase) so
    you're matching real values, not guesses.

    Returns up to `limit` distinct values, most common first.
    """
    try:
        values = aact_db.list_values(table_name, column_name, limit)
    except ValueError as e:
        return [{"error": str(e)}]
    # same empty-content issue as sql_exec_tool below - keep this non-empty too
    return values if values else [{"result": "column has no non-null values"}]


@mcp.tool()
def sql_gen_tool(sql: str) -> dict:
    """
    Validate a SELECT query before running it with sql_exec_tool. Checks
    it's read-only, single-statement, and adds a LIMIT if you forgot one.
    Always run your SQL through this before sql_exec_tool - it'll tell you
    exactly what's wrong if something's off.
    """
    try:
        return {"sql": aact_db.validate_sql(sql), "ok": True}
    except ValueError as e:
        return {"error": str(e), "ok": False}


@mcp.tool()
@aact_db.cached
def sql_exec_tool(sql: str) -> list[dict]:
    """
    Run a validated SELECT against AACT and return the rows. Re-validates
    the query itself (read-only, single-statement) even if you already
    called sql_gen_tool - so it's always safe to call directly too.
    """
    try:
        safe_sql = aact_db.validate_sql(sql)
    except ValueError as e:
        return [{"error": str(e)}]
    try:
        rows = aact_db.run_query(safe_sql)
    except Exception as e:
        return [{"error": f"query failed: {e}"}]
    # a query that legitimately matches zero rows (e.g. "how many are
    # currently recruiting" when none are) used to come back as a bare []
    # - groq's api rejects a tool message whose content is empty outright
    # (has to be a non-empty string or a non-empty list), so that crashed
    # the whole request instead of just answering "0". spelling the empty
    # case out keeps the tool message non-empty and makes it unambiguous.
    return rows if rows else [{"result": "query ran fine, 0 rows matched"}]


# ---------- day4: cross-source pivot ----------

_NCT_ID_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
_NCT_ID_FINDALL_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


@mcp.tool()
@aact_db.cached
def linkage_lookup(nct_id: str) -> dict:
    """
    Cross-source pivot: given a clinical trial's NCT ID, returns its AACT
    record (title, status, phase, conditions) plus any chunks from the
    ingested papers that look related. Use this whenever a question names
    a specific NCT ID.

    heads up: the ingested papers are general research papers, not
    registered-trial publications, so related_paper_chunks is a best-effort
    topical match, not a confirmed reference to this trial.
    """
    nct_id = nct_id.strip().upper()
    if not _NCT_ID_RE.match(nct_id):
        return {"error": f"'{nct_id}' doesn't look like an NCT ID (expected NCT + 8 digits)"}

    study = aact_db.get_study(nct_id)
    if not study:
        return {"error": f"no study {nct_id} in AACT"}
    study["conditions"] = aact_db.get_conditions(nct_id)

    pivot_query = " ".join(filter(None, [study.get("brief_title"), *study["conditions"]]))
    related_chunks = get_retriever().retrieve(pivot_query, top_k=3) if pivot_query else []

    return {
        "study": study,
        "related_paper_chunks": related_chunks,
        "note": (
            "related_paper_chunks is a best-effort topical match, not a confirmed "
            "reference to this trial - check before treating it as a citation"
        ),
    }


# ---------- day4: entity extraction ----------
# NCT IDs are regex-extracted (the format's fixed). conditions/interventions
# are free text, so those go through a small structured-output call on the
# same model the agent's already using.

class _ExtractedEntities(BaseModel):
    conditions: list[str] = Field(default_factory=list, description="disease/condition names mentioned")
    interventions: list[str] = Field(
        default_factory=list, description="drugs, devices, or procedures mentioned"
    )


_extractor = None


def _build_extractor_llm():
    """same provider switch as agent.py's _build_llm() - kept duplicated
    here rather than imported from src.agent.agent, since this file is
    meant to run as its own process/container eventually (see the module
    docstring - agent.py's process never imports aact_db/hybrid_retriever
    directly, this one shouldn't reach into src/agent/ for anything either).

    without this, extract_entities was silently still calling groq no
    matter which provider OLLAMA_API_KEY/GOOGLE_API_KEY had switched the
    main agent to - init_chat_model(AGENT_MODEL) always resolves to groq's
    default model regardless of env, so every extract_entities call kept
    hitting groq's still-exhausted quota and silently degrading to
    regex-only NCT extraction (see the except Exception below), even in
    the middle of testing ollama/gemini specifically to get away from
    that same quota."""
    if os.getenv("OLLAMA_API_KEY"):
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
        )
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_MODEL)

    from langchain.chat_models import init_chat_model

    return init_chat_model(AGENT_MODEL)


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = _build_extractor_llm().with_structured_output(_ExtractedEntities)
    return _extractor


@mcp.tool()
def extract_entities(text: str) -> dict:
    """
    Pull out clinical trial NCT IDs, condition names, and intervention
    (drug/device/procedure) names mentioned in a chunk of text. Use this on
    retrieved paper chunks to figure out what to look up in AACT next.
    """
    nct_ids = sorted({m.upper() for m in _NCT_ID_FINDALL_RE.findall(text)})

    try:
        result = _get_extractor().invoke(
            f"Extract every disease/condition and every drug/device/procedure "
            f"intervention mentioned in this text:\n\n{text}"
        )
        conditions, interventions = result.conditions, result.interventions
    except Exception as e:
        return {
            "nct_ids": nct_ids,
            "conditions": [],
            "interventions": [],
            "warning": f"model-based extraction failed, only regex NCT IDs returned: {e}",
        }

    return {"nct_ids": nct_ids, "conditions": conditions, "interventions": interventions}


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if MCP_TRANSPORT == "http" else "stdio")
