"""
day5: structured working memory for the ReAct loop, as agent middleware
(create_agent's hook system - see agent.py for where this gets wired in).

spec wants 4 fields tracked across iterations:
  - confirmed_facts:        short notes on what tool calls actually returned
  - open_questions:         reserved for the structured/AACT + skills days -
                             once those tools exist they can push sub-questions
                             in here. stays empty in this pass.
  - recent_tool_signatures: last few (tool, args) calls, so repeats can be caught
  - last_failed_attempt:    whatever the most recent tool error was

rather than hand-updating these piece by piece across different hooks (easy
to get out of sync with what the messages actually say happened),
before_model just rebuilds them from the message history every turn. message
lists here are small, so that's cheap, and it can't drift from reality.

anti-redundancy is the one thing that has to happen at tool-call time, not
after - awrap_tool_call checks the proposed call against
recent_tool_signatures and skips actually re-running it if it's an exact
repeat. async (not wrap_tool_call) because agent.py always runs the agent
via ainvoke() - create_agent needs the async version of this specific hook
when running async, it doesn't fall back to a sync one like before_model
does.
"""

import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware, AgentState
from typing_extensions import NotRequired

logger = logging.getLogger("agenticrag.agent")

MAX_CONFIRMED_FACTS = 10
MAX_RECENT_SIGNATURES = 5
# how much of each tool result's str() gets kept in a fact - this is NOT
# sent back to the model (confirmed_facts never lands in a message, just
# state/logging/the long-term store, see long_term_memory.py), so it costs
# nothing token-wise. was 150, way too aggressive in practice - a
# retrieve_chunks result is a list of dicts, so the first ~150 chars is
# mostly {'id': '...', 'text': ' boilerplate before any real content
# shows up, which meant the long-term store almost never had anything
# distinctive enough to match a later question against (confirmed by
# testing: "persisted" fired every turn, "recalled" never did)
FACT_SNIPPET_CHARS = 500


class WorkingMemoryState(AgentState):
    confirmed_facts: NotRequired[list[str]]
    open_questions: NotRequired[list[str]]
    recent_tool_signatures: NotRequired[list[str]]
    last_failed_attempt: NotRequired[str | None]
    iteration: NotRequired[int]


def _signature(name, args) -> str:
    # sorted so {"a":1,"b":2} and {"b":2,"a":1} count as the same call
    return f"{name}({dict(sorted(args.items()))})"


def _state_get(state, key, default=None):
    # request.state can be a dict, a list, or a BaseModel depending on
    # context per the ToolCallRequest docs - dict is what create_agent
    # actually uses, but .get() defensively in case that ever changes
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


class WorkingMemoryMiddleware(AgentMiddleware):
    state_schema = WorkingMemoryState

    def before_model(self, state, runtime):
        messages = state.get("messages", [])

        call_signatures = {}  # tool_call_id -> signature
        facts, signatures, last_failed = [], [], None

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_signatures[tc["id"]] = _signature(tc["name"], tc["args"])
            elif isinstance(msg, ToolMessage):
                sig = call_signatures.get(msg.tool_call_id, msg.name or "tool")
                snippet = str(msg.content)[:FACT_SNIPPET_CHARS]
                if getattr(msg, "status", None) == "error":
                    last_failed = f"{sig} -> {snippet}"
                    # still register the signature (just not as a "fact") -
                    # a call that failed once with these exact args is going
                    # to fail again with the same error, so there's no
                    # reason to let the model burn another whole iteration
                    # re-running it verbatim. found this gap while digging
                    # into why a question that eval answered in a single
                    # linkage_lookup call (see eval/ragas_results.json) was
                    # blowing through the full iteration cap instead - a
                    # retried-failing-call loop that never counts as a
                    # "repeat" (because only successful calls were ever
                    # added to signatures before) is exactly the kind of
                    # thing that'd do that. a DIFFERENT set of args on the
                    # retry is still a different signature and gets through
                    # fine, this only blocks an EXACT repeat.
                    if sig not in signatures:
                        signatures.append(sig)
                else:
                    facts.append(f"{sig} -> {snippet}")
                    if sig not in signatures:
                        signatures.append(sig)

        iteration = state.get("iteration", 0) + 1
        # logging the actual signatures now, not just a count - a bare
        # "recent_calls=3" told us nothing about WHAT the agent was doing
        # for 15 iterations, which is exactly the visibility we needed
        # missing when this started looping.
        logger.info(
            "[iter %d] facts=%d open_qs=%d recent_calls=%s last_failed=%s",
            iteration,
            len(facts),
            len(state.get("open_questions", [])),
            signatures,
            last_failed,
        )

        return {
            "iteration": iteration,
            "confirmed_facts": facts[-MAX_CONFIRMED_FACTS:],
            "recent_tool_signatures": signatures[-MAX_RECENT_SIGNATURES:],
            "last_failed_attempt": last_failed,
        }

    async def awrap_tool_call(self, request, handler):
        sig = _signature(request.tool_call["name"], request.tool_call["args"])
        seen = _state_get(request.state, "recent_tool_signatures", [])

        if sig in seen:
            logger.info("anti-redundancy: skipping repeat call %s", sig)
            return ToolMessage(
                content=(
                    f"(skipped - already called {sig} earlier and got a result for it; "
                    "see confirmed_facts instead of calling this again)"
                ),
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )

        return await handler(request)
