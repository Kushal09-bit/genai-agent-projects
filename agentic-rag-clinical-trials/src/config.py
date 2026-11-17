

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DOCS_DIR = REPO_ROOT / "docs"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")
COLLECTION_NAME = "pubmed_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
MAX_HEADING_WORDS = 8


RETRIEVAL_TOP_K = 3  # was 5 - dropped to keep retrieve_chunks results small


RETRIEVAL_CANDIDATE_POOL = 20  # how many candidates go into the MMR pass
HYBRID_ALPHA = 0.5  # 0 = pure bm25, 1 = pure dense
MMR_LAMBDA = 0.5  # 0 = max diversity, 1 = max relevance
MAX_CHUNK_CHARS = 900  # ~roughly 220 tokens - retrieve_chunks trims each




AGENT_MODEL = os.getenv("AGENT_MODEL", "groq:openai/gpt-oss-120b")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))  # hard cap, 6 is the reference value


MAX_HISTORY_TOKENS = int(os.getenv("AGENT_MAX_HISTORY_TOKENS", "12000"))


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # cheap +

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")


AACT_DB_HOST = os.getenv("AACT_DB_HOST")
AACT_DB_PORT = os.getenv("AACT_DB_PORT", "5432")
AACT_DB_NAME = os.getenv("AACT_DB_NAME")
AACT_DB_USER = os.getenv("AACT_DB_USER")
AACT_DB_PASSWORD = os.getenv("AACT_DB_PASSWORD")
AACT_SCHEMA = "ctgov"

AACT_STATEMENT_TIMEOUT_MS = 10_000  # it's a shared public db, don't hang it
AACT_MAX_ROWS = 200  # hard cap slapped onto any query that doesn't already LIMIT

SKILLS_DIR = REPO_ROOT / "skills"


MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # "stdio" | "http"
MCP_SERVER_BIND_HOST = os.getenv("MCP_SERVER_BIND_HOST", "0.0.0.0")  # what the server itself binds to
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://localhost:{MCP_SERVER_PORT}/mcp")
