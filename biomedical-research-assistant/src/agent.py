import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from mcp.types import CallToolResult, TextContent

from src.guardrails import InputGuardrail, OutputGuardrail, OUT_OF_CORPUS_MESSAGE

SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")

SYSTEM_PROMPT = """You are a Biomedical Research Assistant. You answer questions using
only the retrieved chunks from the PubMed paper corpus, plus the conversation history
already visible to you for follow-up questions.

IMPORTANT: Call retrieve_chunks AT MOST ONCE per question, then immediately give your
final answer. Do not call retrieve_chunks a second time under any circumstances.

If this question is a follow-up that can be answered directly from something already
established earlier in this conversation (e.g. "what page was that on?", "which document
was that from?"), answer it from that context WITHOUT calling retrieve_chunks again —
cite the exact same document_id/page_number you already gave for that fact.

If retrieve_chunks returns an empty chunks list (no relevant results found) and this is
NOT an in-context follow-up, respond with EXACTLY: "I could not find this in the available
literature."

Otherwise, you MUST respond in EXACTLY this format, with no extra text before or after:

ANSWER: [your answer here]
CITATIONS: [document_id p.page_number, document_id p.page_number]

If the exact fact asked for isn't stated verbatim, but a closely related fact IS present
in the retrieved chunks or conversation context (e.g. the question asks for "sample size"
and the text states the number of included studies/trials rather than a patient count),
report that closely related fact plainly and note what specifically isn't stated — do not
decline just because the wording doesn't match exactly.

Only if NOTHING relevant to the question is present at all, respond with EXACTLY:
ANSWER: NO_CITABLE_ANSWER
CITATIONS: none

Example of a normal answer:
ANSWER: The study found a 40% response rate at week 12.
CITATIONS: lupus-2021-000581 p.4

Rules:
- Only answer based on what the retrieved chunks or prior conversation context actually say.
- Every claim must be traceable to a citation.
- Never provide clinical advice or treatment recommendations.
"""

langfuse_handler = CallbackHandler()
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()

# Shared across every ask() call and every thread_id — this is what makes
# memory actually persist. It's keyed by thread_id internally, so a fresh
# graph object can be built per call (needed anyway, since tools/interceptor
# state must reset per question) while still sharing the same underlying
# conversation state store.
_checkpointer = MemorySaver()


def _parse_structured(raw_output: str) -> tuple[str, list]:
    """Pulls ANSWER and CITATIONS out of the agent's formatted output, tolerant
    of the LLM not following the format perfectly."""
    answer = raw_output.strip()
    citations = []

    if "CITATIONS:" in raw_output:
        parts = raw_output.split("CITATIONS:")
        citations_part = parts[1].strip()
        answer_part = parts[0]

        if "ANSWER:" in answer_part:
            answer_part = answer_part.split("ANSWER:", 1)[1]

        answer = answer_part.strip()

        if citations_part and citations_part.lower() not in ("none", "n/a", "", "[]"):
            citations = [
                c.strip() for c in citations_part.replace("[", "").replace("]", "").split(",")
                if c.strip()
            ]
    elif "ANSWER:" in raw_output:
        answer = raw_output.split("ANSWER:", 1)[1].strip()

    return answer, citations


def _extract_chunks_from_observation(observation) -> list:
    """Handles both string and list-shaped observations from MultiServerMCPClient."""
    try:
        if isinstance(observation, list):
            if len(observation) > 0 and isinstance(observation[0], dict) and "text" in observation[0]:
                inner_json_str = observation[0]["text"]
                data = json.loads(inner_json_str)
                return data.get("chunks", [])
            return []

        if isinstance(observation, str):
            outer = json.loads(observation)
            if isinstance(outer, list) and len(outer) > 0 and "text" in outer[0]:
                inner_json_str = outer[0]["text"]
                data = json.loads(inner_json_str)
            elif isinstance(outer, dict):
                data = outer
            else:
                return []
            return data.get("chunks", [])

        return []
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return []


_RETRIEVAL_STOP_MESSAGE = (
    "You already called retrieve_chunks once for this question. Do not call "
    "it again — answer now using the chunks you already retrieved. If they "
    "are not sufficient, respond with exactly ANSWER: NO_CITABLE_ANSWER / "
    "CITATIONS: none."
)


def _make_single_retrieval_interceptor():
    """The system prompt tells the model to call retrieve_chunks exactly
    once, but a tool-calling LLM will sometimes retry with a reworded query
    when it's unsatisfied with the first result — and with max_iterations
    capped low, that second tool call can consume the run's last iteration,
    leaving no turn free to produce a final answer at all (observed as
    "Agent stopped due to max iterations."). This enforces the "once" rule
    in code: the real retrieval only ever runs on the first call per
    question; any further call is short-circuited with an instruction to
    stop searching and answer now, instead of hitting the vector store
    again and burning another iteration on redundant retrieval.
    """
    call_counts: dict[str, int] = {}

    async def interceptor(
        request: MCPToolCallRequest, handler
    ) -> MCPToolCallResult:
        if request.name != "retrieve_chunks":
            return await handler(request)

        call_counts[request.name] = call_counts.get(request.name, 0) + 1
        if call_counts[request.name] == 1:
            return await handler(request)

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"chunks": [], "note": _RETRIEVAL_STOP_MESSAGE}),
            )],
            isError=False,
        )

    return interceptor


def _tool_messages_for_this_turn(messages: list) -> list[ToolMessage]:
    """create_agent + the checkpointer returns the FULL conversation
    history for the thread, not just this turn's messages. Walking
    backwards from the end and stopping at the most recent HumanMessage
    isolates just the tool calls made while answering the current
    question, regardless of how much prior history is in the thread."""
    turn_tool_messages = []
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            turn_tool_messages.append(msg)
    turn_tool_messages.reverse()
    return turn_tool_messages


def _recent_history_text(thread_id: str, max_turns: int = 3) -> str:
    """Reads prior (question, final answer) pairs for this thread straight
    from the checkpointer, so the input guardrail can judge follow-up
    questions ("what page was that on?") in context instead of blocking
    them as off-topic in isolation. Returns "" for a new/unknown thread."""
    try:
        checkpoint = _checkpointer.get({"configurable": {"thread_id": thread_id}})
    except Exception:
        return ""
    if not checkpoint:
        return ""

    messages = checkpoint.get("channel_values", {}).get("messages", [])
    turns = []
    pending_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            pending_question = msg.content
        elif isinstance(msg, AIMessage) and msg.content and pending_question is not None:
            turns.append((pending_question, msg.content))
            pending_question = None

    return "\n".join(f"Q: {q}\nA: {a}" for q, a in turns[-max_turns:])


async def ask(question: str, thread_id: str = "default") -> str:
    """thread_id scopes conversation memory: calls with the same thread_id
    share history via the checkpointer; a new thread_id starts fresh. Each
    call still only ever performs the real retrieve_chunks lookup once
    (see _make_single_retrieval_interceptor) — memory is about the model
    seeing prior turns, not about relaxing that per-question limit.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    # 4.1 — Input guardrail (with recent conversation context, so a natural
    # follow-up isn't misjudged as off-topic in isolation)
    history_text = _recent_history_text(thread_id)
    valid, reason = input_guardrail.check(question, history=history_text)
    if not valid:
        return reason

    client = MultiServerMCPClient(
        {
            "pubmed": {
                "command": sys.executable,
                "args": [SERVER_PATH],
                "transport": "stdio",
            }
        },
        tool_interceptors=[_make_single_retrieval_interceptor()],
    )

    tools = await client.get_tools()

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=api_key)
    agent = create_agent(
        llm, tools, system_prompt=SYSTEM_PROMPT, checkpointer=_checkpointer
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler],
        },
    )
    raw_output = result["messages"][-1].content

    # 4.4 — Out-of-corpus fallback: check if retrieval returned no chunks.
    # For a follow-up the model correctly answers from conversation memory
    # without retrieving again, this turn has no tool call at all — that's
    # not "out of corpus", so ground the answer against the most recent
    # successful retrieval earlier in the thread instead.
    this_turn_tool_messages = _tool_messages_for_this_turn(result["messages"])

    cited_chunks_text = ""
    for tool_message in this_turn_tool_messages:
        chunks = _extract_chunks_from_observation(tool_message.content)
        if chunks:
            cited_chunks_text = "\n\n".join(
                f"[{c['document_id']} p.{c['page_number']}] {c['text']}" for c in chunks
            )

    if not cited_chunks_text and not this_turn_tool_messages:
        for tool_message in reversed(result["messages"]):
            if not isinstance(tool_message, ToolMessage):
                continue
            chunks = _extract_chunks_from_observation(tool_message.content)
            if chunks:
                cited_chunks_text = "\n\n".join(
                    f"[{c['document_id']} p.{c['page_number']}] {c['text']}" for c in chunks
                )
                break

    if not cited_chunks_text:
        return OUT_OF_CORPUS_MESSAGE

    # Parse structured answer + citations
    answer, citations = _parse_structured(raw_output)

    status, final_answer = output_guardrail.check(
        question=question,
        answer=answer,
        citations=citations,
        cited_chunks_text=cited_chunks_text,
    )

    # Citations were verified above but never shown to the user — surface
    # them on any answer that actually passed grounding.
    if status in ("GROUNDED", "PARTIAL") and citations:
        final_answer = f"{final_answer}\n\nSources: {', '.join(citations)}"

    return final_answer


if __name__ == "__main__":
    question = "What was the response rate for Iberdomide in the lupus trial?"
    answer = asyncio.run(ask(question))
    print("\n--- ANSWER ---")
    print(answer)