# multi-agent-qa-assistant

Multi-source Q&A assistant built with LangGraph. One question can need data from two very different places:
- **SQL** - the Chinook music store database (artists, albums, tracks, invoices, customers)
- **RAG** - a small corpus of PubMed biomedical papers

A supervisor decides where each question should go, the sub-agents do the work, and an aggregator stitches the answer together.

## Flow

```
question -> guardrail -> load memory -> supervisor --sql--> sql agent ----------> aggregator -> answer
                  |                         |--rag--> rag agent ----------> aggregator
               blocked                      |--both-> sql agent -> rag agent -> aggregator
                                            '--off_topic-------------------> polite refusal
```

- **Guardrails** (`guardrails.py`, `sql_guardrails.py`) - catches prompt injection on the way in, validates generated SQL (read-only, allowed tables only) before it runs
- **Supervisor** (`supervisor.py`) - one LLM routing call per turn: `sql`, `rag`, `both` or `off_topic`
- **Sub-agents** (`agents/`) - SQL agent and RAG agent, both calling tools through MCP
- **MCP server** (`mcp_layer/mcp_server.py`) - owns the DB and the vector store, exposes `retrieve_chunks`, `get_schema_and_samples` and `execute_sql` as MCP tools (plus plain REST endpoints)
- **Memory** (`memory.py`) - short-term via LangGraph checkpointer, long-term per user via LangGraph Store (semantic search over past turns)
- **Backend** (`backend_api.py`) - FastAPI wrapping the graph
- **Frontend** (`app.py`) - thin Streamlit client that just talks to the backend over HTTP
- **Tracing** - Langfuse (`tracing.py`)
- **Eval** - Ragas over the multi-agent pipeline (`eval/eval_ragas.py`)
- **Docker** - mcp-server + backend as separate containers (`docker-compose.yml`)

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements/backend.txt -r requirements/mcp-server.txt -r requirements/client.txt
cp .env.example .env   # add your groq key + langfuse keys
```

Chinook DB is already included (`sql_db/data/chinook.sqlite`). For the RAG side, drop a few PubMed PDFs into `data/pubmed_docs/` (not included here) and build the vector store:

```bash
pip install -r requirements/rag-ingestion.txt
python ingest_biomedical_corpus.py
```

## Run

```bash
docker compose up --build      # mcp-server + backend
streamlit run app.py           # chat UI
```

Eval:
```bash
pip install -r requirements/eval.txt
python -m eval.eval_ragas
```
