"""
long-term memory, as agent middleware - separate from working_memory.py on
purpose. working_memory.py's confirmed_facts resets every new thread_id
(each REPL/API session starts fresh); this middleware persists a copy of
those same facts into a LangGraph Store, and pulls back whatever's
semantically relevant to the current question before each model call - so
asking something related in a BRAND NEW thread can still recall facts from
an earlier one.

still in-memory (InMemoryStore, gone when the process exits) - same
limitation as the InMemorySaver checkpointer in agent.py. it's a longer-
lived kind of memory (survives across thread_ids), not a permanently
persisted one (doesn't survive a restart).

facts are keyed by a hash of their own text, not a random id - so
re-confirming the same fact in a later turn overwrites the same store
entry instead of piling up duplicates (working_memory's confirmed_facts is
rebuilt from the whole thread's history each turn, so the same fact can
show up again across several turns of one conversation).

also trims the per-thread message history sent to the model (see
MAX_HISTORY_TOKENS below) - not really a "long-term memory" concern, but
this middleware already runs a wrap_model_call on every single call and
already touches request.messages/system_message, so bolting the trim on
here beat adding a whole separate middleware just for it.
"""

import hashlib
import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME, MAX_HISTORY_TOKENS

logger = logging.getLogger("agenticrag.agent")

MEMORY_NAMESPACE = ("agenticrag", "facts")
RECALL_LIMIT = 5
# a thread's full message history (every past turn's tool calls + tool
# results + final answers) gets re-sent to the model on every new turn in
# that thread - with no cap on that, 2-3 real turns of AACT/retrieval tool
# use is enough to blow straight through groq's 8000 tokens-per-minute
# budget on its own, before the model's even generated anything back
# (confirmed by testing: a fresh thread's first question was fine, but by
# the third turn every request was getting rejected with a 413, size
# climbing each turn). trim_messages is the standard langchain/langgraph
# answer to exactly this - keeps the most recent messages under a token
# budget, and start_on="human" makes sure it can't cut the list off in the
# middle of a tool-call/tool-result pair, which the api would reject
# outright (same shape of bug as the empty-content one in mcp_server.py).
# MAX_HISTORY_TOKENS itself now lives in config.py (env-overridable) - see
# the comment there for why the old hardcoded 3000 got bumped.

# cosine similarity cutoff - without this, a near-empty store just returns
# whatever it has regardless of relevance (confirmed by testing: an
# unrelated question still "recalled" the only fact in the store). rough
# number, not tuned against a real eval set - raise it if irrelevant facts
# keep showing up, lower it if genuinely relevant ones get dropped.
# dropped from 0.3 to 0.2 alongside widening FACT_SNIPPET_CHARS in
# working_memory.py - a stored fact is a tool-call signature + a chunk of
# raw tool output (json-ish, not a clean sentence), so its embedding never
# reads as similar to a natural-language question the way two plain
# sentences would - 0.3 was filtering out real matches, not just noise
# (confirmed by testing: recall never fired once, even for an
# on-topic question straight after the fact was persisted).
RECALL_MIN_SCORE = 0.2
EMBED_DIMS = 384  # all-MiniLM-L6-v2's output size - keep in sync with EMBED_MODEL_NAME

_embedder = None


def embed_fn(texts: list[str]) -> list[list[float]]:
    """raw embedding function for the store's semantic index. same model
    ingestion/retrieval use, but loaded separately here - this runs in the
    agent process, not the mcp server subprocess, so there's no object to
    share between them anyway."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder.encode(texts).tolist()


def _fact_key(fact: str) -> str:
    return hashlib.sha256(fact.encode("utf-8")).hexdigest()[:16]


def _safe_trim_head(head: list) -> list:
    """trim_messages(strategy='last', start_on='human') has an all-or-
    nothing failure mode: if the messages it's given don't fit the budget
    AND can't be cut down to something that starts on a human message
    within that budget, it returns a flat-out empty list - not a shorter
    one. that's silent and easy to miss, because most of the time
    "shorter" is exactly what happens.

    caught this for real: a same-session followup question ("what phase
    is that trial in?"), asked right after the trial was identified in the
    previous turn, came back with zero memory of that trial - not because
    of a restart, not a wrong thread_id, just this. that first turn's own
    AACT/retrieval tool chatter alone (a few tool calls, each returning a
    decent chunk of JSON) was bigger than MAX_HISTORY_TOKENS on its own,
    and since it only has ONE human message (right at its start), trimming
    to "the end, but starting on human" had nothing valid to return -
    reproduced directly: a single oversized turn goes from a real message
    list straight to [].

    losing the tool-call trail for an old turn is fine, that's the trim's
    whole job. losing the fact that the conversation happened AT ALL is
    the actual bug. so: if the real trim comes back empty on a non-empty
    head, fall back to keeping just that head's final answer - enough for
    a followup to anchor "that trial"/"it"/etc. against, even without the
    reasoning that got there."""
    trimmed = trim_messages(
        head,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
    )
    if trimmed or not head:
        return trimmed

    last_answer = next((m for m in reversed(head) if isinstance(m, AIMessage) and m.content), None)
    return [last_answer] if last_answer else []


class LongTermMemoryMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request, handler):
        store = request.runtime.store
        question = next(
            (m.content for m in reversed(request.messages) if getattr(m, "type", None) == "human"),
            None,
        )
        if store is not None and question:
            hits = await store.asearch(MEMORY_NAMESPACE, query=question, limit=RECALL_LIMIT)
            hits = [h for h in hits if h.score is not None and h.score >= RECALL_MIN_SCORE]
            if hits:
                recalled = "\n".join(f"- {h.value['fact']}" for h in hits)
                base = request.system_message.content if request.system_message else ""
                note = f"{base}\n\nRelevant facts recalled from earlier conversations:\n{recalled}"
                request = request.override(system_message=SystemMessage(content=note))
                logger.info("long-term memory: recalled %d fact(s) for this question", len(hits))

        # only trim PRIOR, completed turns - never anything from the current
        # in-progress question onward. first pass at this trimmed
        # unconditionally on every call, which included mid-loop calls
        # WITHIN a single question (schema_tool -> values_tool ->
        # sql_gen_tool -> sql_exec_tool are several model calls for one
        # user turn) - if that turn's own tool chatter already ate the
        # whole token budget, trim_messages correctly refused to violate
        # start_on="human" by keeping a message list that didn't start on
        # one, and returned nothing at all instead. that erased the user's
        # own question mid-turn (confirmed by testing: the model responded
        # "please provide your question" to a real question it never saw).
        # splitting on the LAST human message and only trimming what comes
        # before it fixes that - the current turn's messages (from that
        # human message on) are never touched, whatever their token count.
        messages = request.messages
        last_human_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if getattr(messages[i], "type", None) == "human"),
            None,
        )
        if last_human_idx:  # 0 or None both mean "nothing safe to trim"
            head, tail = messages[:last_human_idx], messages[last_human_idx:]
            trimmed_head = _safe_trim_head(head)
            if len(trimmed_head) != len(head):
                logger.info(
                    "long-term memory: trimmed prior-turn history %d -> %d messages (token budget)",
                    len(head),
                    len(trimmed_head),
                )
                request = request.override(messages=trimmed_head + tail)

        return await handler(request)

    async def aafter_agent(self, state, runtime):
        store = runtime.store
        if store is None:
            return None
        facts = state.get("confirmed_facts", [])
        for fact in facts:
            await store.aput(MEMORY_NAMESPACE, _fact_key(fact), {"fact": fact})
        if facts:
            logger.info("long-term memory: persisted %d fact(s)", len(facts))
        return None
