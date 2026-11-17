# agentic-rag-clinical-trials

A single ReAct agent that answers questions about clinical trials by mixing two sources:
- **structured** - the public [AACT](https://aact.ctti-clinicaltrials.org/) clinical trials registry (Postgres), queried with SQL
- **unstructured** - a small set of biomedical research papers, searched with RAG

The agent decides on its own which tools to call, and can pivot across both (e.g. find a trial in AACT, then pull what the papers say about it).

## How it's put together

- **MCP server** (`src/mcp_server.py`) - all the tools live here:
  - `retrieve_chunks` - hybrid retrieval (BM25 + dense embeddings, fused, then MMR so the top-k isn't near-duplicates) - see `src/hybrid_retriever.py`
  - `schema_tool`, `values_tool` - look up real AACT tables/columns/values before writing SQL
  - `sql_gen_tool` -> `sql_exec_tool` - validate the SQL first, then run it (read-only, row cap, statement timeout)
  - `linkage_lookup`, `extract_entities` - link a trial's NCT id to the papers
- **Agent** (`src/agent/`) - LangGraph ReAct agent with an iteration cap, working memory per thread, long-term memory across threads (LangGraph Store), and guardrails on input, tool args and output (blocks injection attempts, redacts leaked secrets)
- **Skills** (`skills/`) - markdown skill files the agent loads for SQL querying and cross-source pivoting
- **Ingestion** (`src/ingestion_pipeline.py`) - chunks the PDFs in `docs/` into ChromaDB
- **Backend** (`src/api.py`) - FastAPI `/query` endpoint with rate limiting
- **Frontend** (`app.py`) - Streamlit chat UI that only talks to the backend over HTTP
- **Eval** (`eval/ragas_eval.py`) - Ragas (faithfulness, answer relevancy, context precision/recall), results in `eval/ragas_results.json`
- **Docker** (`docker/`) - 3 services (mcp-server, backend, frontend) with healthchecks so they start in the right order

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements/base.txt -r requirements/backend.txt -r requirements/frontend.txt -r requirements/mcp_server.txt
cp .env.example .env   # fill in keys + your AACT login (free signup)
```

Put a few PubMed PDFs in `docs/` (not included here), then build the vector store:

```bash
python -m src.ingestion_pipeline
```

## Run

Locally:
```bash
uvicorn src.api:app --port 8000   # backend
streamlit run app.py              # frontend
```

Or everything in docker:
```bash
cd docker && docker compose up --build
```

Eval:
```bash
python -m eval.ragas_eval
```
