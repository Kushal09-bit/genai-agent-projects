# Biomedical Research Assistant

A little RAG chatbot that answers questions using a small corpus of PubMed papers (lupus, Lassa fever, pneumonia, OCD/neurophysiology, pain research, HIV/pregnancy prevention, pediatrics). It only answers from what's actually in the PDFs, cites its sources, and has guardrails so it doesn't hallucinate, give medical advice, or get jailbroken.

## What's actually in here

- **Ingestion** (`src/ingest.py`) — reads the PDFs and builds a Chroma vector store. Uses OCR (pdf2image + pytesseract) instead of plain text extraction, because a bunch of the source PDFs have broken font encoding that turns into garbage text otherwise.
- **Retrieval** (`src/mcp_server.py`) — an MCP server exposing a `retrieve_chunks` tool. Does a wider first-pass vector search, then reranks the candidates with a cross-encoder so chunks phrased differently than the question still get found.
- **Agent** (`src/agent.py`) — a LangGraph ReAct agent (Groq, `openai/gpt-oss-120b`) that calls the retrieval tool and answers with citations. Has real conversation memory (`MemorySaver` checkpointer), so follow-up questions like "what page was that on?" actually work.
- **Guardrails** (`src/guardrails.py`) — input side blocks prompt injection, jailbreak attempts, off-topic questions, and requests for personal medical advice (it's context-aware, so it won't block a legit follow-up just because it looks weird on its own). Output side enforces that every answer has a citation and double-checks the answer is actually grounded in the retrieved text before it goes out.
- **Streamlit app** (`streamlit_app.py`) — the actual chat UI.
- **CLI** (`main.py`) — same thing but in your terminal, if you don't want to bother with the browser.
- **Eval** (`src/evaluate.py`) — RAGAS evaluation (faithfulness, answer relevancy, context precision). Generates its own test questions from random chunks in the corpus rather than using hand-written ones, so there's nothing to keep in sync by hand.
- **Docker** — `Dockerfile` + `docker-compose.yml`, builds and runs the whole thing in a container (includes the OCR system dependencies, which are easy to forget).

## Setup

1. Create a venv and install deps:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Fill in `.env` with your keys — `ANTHROPIC_API_KEY` (this is actually where the Groq key goes, yes it's a confusing name, it's from an earlier version of this project) plus the Langfuse keys if you want tracing.
3. Build the vector store:
   ```bash
   python src/ingest.py
   ```
   Only needs to be done once, or again if you add/change PDFs.

## Running it

Chat UI:
```bash
streamlit run streamlit_app.py
```

Terminal instead:
```bash
python main.py
```

Docker (does everything above for you):
```bash
docker compose up
```

Run the eval:
```bash
python -m src.evaluate
```

## Heads up about Groq's free tier

The free/on-demand tier has a 200k-tokens-per-day cap, and every single question burns through a handful of Groq calls (input classifier, the agent itself, output grounding check). It's easy to blow through the daily limit if you're testing a lot in one sitting — if the bot suddenly starts refusing everything, that's probably why, not a bug. Either wait for it to reset or grab a fresh key.
