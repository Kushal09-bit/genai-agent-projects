"""
day5: the actual agent. single ReAct-style agent (not a supervisor/router) -
built with langchain.agents.create_agent, NOT langgraph.prebuilt's
create_react_agent, since that one's deprecated in LangGraph v1 and on its
way out in v2. create_agent compiles to a LangGraph graph either way - this
file adds a checkpointer + a per-session thread_id so the REPL below is an
actual multi-turn chatbot (it remembers earlier turns in the same run),
not a fresh stateless call for every question.

every tool - retrieval (day3) and all six AACT tools (day4) - comes from
one consolidated MCP server (src/mcp_server.py), reached through the
client wrapper in src/mcp_client.py (kept separate on purpose - that's the
file that'll change when the mcp server moves into its own docker
container later, agent.py itself won't need to). this process never
imports psycopg2 or touches a db connection directly - MCP is the only
boundary in.

two layers of memory: the checkpointer above is per-thread (one REPL/API
session), the store below is long-term (src/agent/long_term_memory.py) -
confirmed facts get promoted into it after every turn and semantically
recalled on future turns, even in a brand new thread.

run with: python -m src.agent.agent   (from the repo root, needs
chroma_db/ already built - see src/ingestion_pipeline.py)
"""

import asyncio
import logging
import os
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.config import (
    AACT_DB_HOST,
    AGENT_MODEL,
    GEMINI_MODEL,
    MAX_ITERATIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SKILLS_DIR,
)
from src.mcp_client import get_tools as get_mcp_tools
from src.agent.working_memory import WorkingMemoryMiddleware
from src.agent.long_term_memory import EMBED_DIMS, LongTermMemoryMiddleware, embed_fn
from src.agent.guardrails import GuardrailMiddleware

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("agenticrag.agent")  # same logger working_memory.py / long_term_memory.py use

BASE_SYSTEM_PROMPT = """You are a research assistant answering multi-hop biomedical
questions by dynamically choosing tools across a structured source (the AACT
clinical trials registry) and an unstructured source (a set of ingested
research papers).

Break multi-part questions down and call tools as many times as you need to,
with different queries/args, to cover every part. Don't repeat a call you've
already made with the same arguments - if you already have the answer to
something, use it.

Always cite where an answer came from (which document/section, or which
AACT table/NCT ID). If your tools don't actually answer the question, say
so - don't make something up.
"""


def _parse_skill(text: str) -> dict:
    """splits a skill file's yaml frontmatter (name/description/allowed-tools,
    between the --- lines) from its actual instructions below. hand-rolled,
    not a real yaml parser - frontmatter here is flat enough (a couple of
    strings + one [a, b, c] list) that pulling in a yaml dependency just for
    this felt like overkill. falls back to treating the whole file as body
    if it doesn't start with ---, so an old-style skill file with no
    frontmatter still loads fine, just with no metadata."""
    if not text.startswith("---"):
        return {"name": None, "description": None, "allowed_tools": [], "body": text.strip()}

    end = text.find("\n---", 3)
    if end == -1:
        return {"name": None, "description": None, "allowed_tools": [], "body": text.strip()}

    front, body = text[3:end].strip(), text[end + 4:].strip()
    meta = {"name": None, "description": None, "allowed_tools": []}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "name":
            meta["name"] = value
        elif key == "description":
            meta["description"] = value
        elif key == "allowed-tools":
            meta["allowed_tools"] = [t.strip() for t in value.strip("[]").split(",") if t.strip()]
    meta["body"] = body
    return meta


def _load_skills(*names, known_tools=frozenset()):
    """skills are markdown files under skills/ - a small yaml frontmatter
    (name, description, allowed-tools) followed by the actual instructions.
    only the body goes into the system prompt; the frontmatter is metadata -
    logged at startup so it's obvious which skills loaded, and checked
    against the agent's real toolset so a typo'd tool name in allowed-tools
    shows up as a warning instead of silently doing nothing. missing skill
    files are skipped, not fatal."""
    blocks = []
    for filename in names:
        path = SKILLS_DIR / filename
        if not path.exists():
            continue
        skill = _parse_skill(path.read_text())
        label = skill["name"] or filename
        unknown = [t for t in skill["allowed_tools"] if known_tools and t not in known_tools]
        if unknown:
            logger.warning("skill %s declares allowed-tools not in the actual toolset: %s", label, unknown)
        logger.info("loaded skill: %s - %s", label, skill["description"] or "(no description)")
        blocks.append(skill["body"])
    return blocks


def _build_system_prompt(known_tools=frozenset()) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    # AACT_DB_HOST just decides whether the SQL/pivot skill instructions get
    # added to the prompt - the tools are always on the MCP server either
    # way, they'd just return a clear "credentials not set" error if called
    # without it. no point steering the model toward them if that's all
    # they'll do.
    if AACT_DB_HOST:
        parts += _load_skills("sql_skill.md", "cross_source_pivot_skill.md", known_tools=known_tools)
    return "\n\n".join(parts)


def _build_llm(max_retries: int = 8):
    """the actual chat model - groq by default, gemini if GOOGLE_API_KEY is
    set, ollama cloud if OLLAMA_API_KEY is set (ollama checked first - see
    config.py for the order this whole thing settled into and why). picked
    this apart from build_agent() so eval/ragas_eval.py's judge llm goes
    through the exact same provider switch instead of hardcoding groq for
    itself.

    ollama's ChatOllama (langchain_ollama) has no api_key field at all
    (checked chat_models.py directly) - auth is just a bearer header on
    the http client, which client_kwargs threads into BOTH the sync and
    async clients (confirmed in source: client_kwargs gets merged into
    async_client_kwargs before constructing self._async_client, so this
    is fine for agent.py's ainvoke()-only usage, not just a sync-path
    afterthought like ChatNVIDIA's max_retries was). also has no
    max_retries field either - not passed on this path for the same
    reason it wasn't passed to ChatNVIDIA.

    max_retries=8 still applies to groq and gemini. ChatGoogleGenerativeAI
    actually has a real max_retries field (checked langchain_google_genai/
    _common.py directly - default is 6, genuinely wired through to the
    google sdk's retry options, not silently dropped like ChatNVIDIA's
    was)."""
    if os.getenv("OLLAMA_API_KEY"):
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
        )
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=max_retries)
    return init_chat_model(AGENT_MODEL, max_retries=max_retries)


async def build_agent():
    tools = await get_mcp_tools()

    # semantic long-term memory - same embedding model as retrieval, just
    # loaded in this process since this isn't behind the MCP boundary.
    # "fields": ["fact"] tells it which part of each stored item to embed.
    store = InMemoryStore(index={"dims": EMBED_DIMS, "embed": embed_fn, "fields": ["fact"]})

    model = _build_llm()

    # tool names, so _load_skills can flag a skill file whose allowed-tools
    # lists something that doesn't actually exist on the mcp server (a typo,
    # or a tool that got renamed and the skill file never got updated) -
    # langchain's tool objects always have .name, this doesn't depend on
    # which transport (stdio/http) fetched them.
    tool_names = frozenset(t.name for t in tools)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=_build_system_prompt(known_tools=tool_names),
        # this is what makes it a real chatbot instead of one-shot Q&A -
        # LangGraph keeps each thread_id's message history, so a followup
        # question ("what about its side effects?") actually has the prior
        # turn to refer back to. in-memory only - history's gone when the
        # process exits, which is fine for a REPL.
        checkpointer=InMemorySaver(),
        store=store,
        middleware=[
            # input guardrail runs first - bad requests never reach working
            # memory, long-term memory, or the model at all.
            GuardrailMiddleware(),
            WorkingMemoryMiddleware(),
            # promotes working memory's confirmed_facts into the long-term
            # store after every turn, and recalls relevant past facts
            # before every model call - see long_term_memory.py.
            LongTermMemoryMiddleware(),
            # hard iteration cap from the spec (6 is the reference value) -
            # "end" so the agent gives back whatever it's got instead of
            # blowing up when it hits the cap. this is per-run (one
            # question), not per-thread - it shouldn't get cheaper to ask
            # things just because the conversation's gone on a while.
            ModelCallLimitMiddleware(run_limit=MAX_ITERATIONS, exit_behavior="end"),
        ],
    )


_langfuse_handler = None


def _get_langfuse_handler():
    """returns a langfuse callback handler if LANGFUSE_SECRET_KEY/PUBLIC_KEY
    are set in .env, None otherwise - tracing's opt-in, same pattern
    AACT_DB_HOST already uses to decide whether the sql skill loads. built
    once and reused (same idea as _extractor in mcp_server.py) - spinning
    up the langfuse client isn't free, no reason to do it every call.

    explicitly passing public_key/secret_key/base_url here instead of
    letting the sdk auto-detect them from env - the sdk looks for
    LANGFUSE_HOST by default, but this repo's .env (see README) already
    calls it LANGFUSE_BASE_URL, so this just reads that name straight
    instead of asking for a rename."""
    global _langfuse_handler
    if _langfuse_handler is not None:
        return _langfuse_handler
    secret_key, public_key = os.getenv("LANGFUSE_SECRET_KEY"), os.getenv("LANGFUSE_PUBLIC_KEY")
    if not (secret_key and public_key):
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )
    _langfuse_handler = CallbackHandler()
    logger.info("langfuse tracing: on (%s)", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
    return _langfuse_handler


def _flush_langfuse():
    """short-lived processes (the eval script especially, but also this
    repl on exit) need to flush explicitly - langfuse batches traces and
    ships them in the background, and a process that just exits can beat
    that background flush to the punch, silently dropping the last batch.
    a long-running api.py container doesn't strictly need this on every
    request, just on shutdown - see the lifespan() teardown in api.py."""
    if _langfuse_handler is not None:
        from langfuse import get_client

        get_client().flush()


def _content_to_text(content) -> str:
    """AIMessage.content is a plain str for groq/ollama, but gemini (and some
    other providers) can hand back a list of content blocks instead -
    [{'type': 'text', 'text': '...'}, ...] - even on a normal, non-tool-call
    turn. QueryResponse.answer in api.py is typed as a plain str, so passing
    a list straight through blew up as a pydantic ValidationError (500) the
    moment gemini returned one of these. this joins every text block's text
    together (block order is already the intended reading order), falling
    back to str() for anything unrecognized (a block shape this hasn't seen,
    or content that's already something else entirely) rather than raising -
    an ugly answer beats a crashed request."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            # other block types (tool_use, thinking, etc.) carry no
            # user-facing answer text - skip them rather than stringifying
            # the whole dict into the reply.
        return "".join(parts)
    return str(content)


async def ask(agent, question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    handler = _get_langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return _content_to_text(result["messages"][-1].content)


async def repl():
    print(f"building agent ({AGENT_MODEL}, max {MAX_ITERATIONS} iterations)...")
    agent = await build_agent()
    thread_id = str(uuid.uuid4())  # one thread for the whole REPL session
    print("ready. ask a question (ctrl-c to quit)\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        answer = await ask(agent, question, thread_id)
        print(f"\n{answer}\n")

    _flush_langfuse()


if __name__ == "__main__":
    asyncio.run(repl())
