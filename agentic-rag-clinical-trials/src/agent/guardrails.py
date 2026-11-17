import logging
import re

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger("agenticrag.agent")

MAX_QUESTION_CHARS = 4000

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.I),
    re.compile(r"reveal (your |the )?system prompt", re.I),
    re.compile(r"you are now (DAN\b|in developer mode)", re.I),
    re.compile(r"disregard (your |all )?(guidelines|rules|instructions)", re.I),
]

_DANGEROUS_ARG_PATTERNS = [
    re.compile(r"[;&|]\s*(rm|curl|wget|nc|bash|sh)\s", re.I),  # shell chaining
    re.compile(r"<script[\s>]", re.I),
    re.compile(r"\$\([^)]*\)"),  # command substitution
]

_SECRET_PATTERNS = [
    re.compile(r"AACT_DB_PASSWORD\s*=\s*\S+", re.I),
    re.compile(r"GROQ_API_KEY\s*=\s*\S+", re.I),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # groq key by shape, even without the env var name attached
    re.compile(r"GOOGLE_API_KEY\s*=\s*\S+", re.I),
    re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),  # google api key by shape, same idea as gsk_ above
    re.compile(r"OLLAMA_API_KEY\s*=\s*\S+", re.I),  # no confirmed fixed prefix for ollama
    # cloud keys like gsk_/AIzaSy/nvapi- have, so this one's var-name-only
    re.compile(r"postgresql://[^\s]+:[^\s]+@[^\s]+"),  # connection string w/ creds
]

REFUSAL = "I can't help with that request"


def _redact(content, patterns):
    """same list-vs-str content shape gemini can return that agent.py's
    _content_to_text normalizes for the api response - this guardrail runs
    BEFORE that normalization (after_agent sees the raw AIMessage), so
    without this the old isinstance(last.content, str) check just skipped
    redaction entirely on a list-content turn. a leaked secret sitting in
    one of those text blocks would've sailed straight past the output
    guardrail. handles both shapes, returns (possibly-rewritten content,
    whether anything was actually redacted)."""
    if isinstance(content, str):
        redacted = content
        leaked = False
        for pattern in patterns:
            if pattern.search(redacted):
                redacted = pattern.sub("[redacted]", redacted)
                leaked = True
        return redacted, leaked

    if isinstance(content, list):
        leaked = False
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text = block["text"]
                for pattern in patterns:
                    if pattern.search(text):
                        text = pattern.sub("[redacted]", text)
                        leaked = True
                block = {**block, "text": text}
            new_blocks.append(block)
        return new_blocks, leaked

    return content, False


class GuardrailMiddleware(AgentMiddleware):
    # ---- input guardrail ----
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime):
        messages = state.get("messages", [])
        question = next(
            (m.content for m in reversed(messages) if getattr(m, "type", None) == "human"),
            "",
        )
        if not isinstance(question, str) or not question.strip():
            return self._blocked("empty question")
        if len(question) > MAX_QUESTION_CHARS:
            return self._blocked(f"question too long ({len(question)} chars, max {MAX_QUESTION_CHARS})")
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(question):
                return self._blocked("that looks like a prompt-injection attempt")
        return None

    def _blocked(self, reason):
        logger.warning("input guardrail: blocking request (%s)", reason)
        return {"jump_to": "end", "messages": [AIMessage(content=f"{REFUSAL} ({reason}).")]}

    # ---- tool-boundary guardrail ----
    async def awrap_tool_call(self, request, handler):
        args_str = str(request.tool_call.get("args", ""))
        for pattern in _DANGEROUS_ARG_PATTERNS:
            if pattern.search(args_str):
                logger.warning(
                    "tool-boundary guardrail: blocked suspicious args for %s",
                    request.tool_call["name"],
                )
                return ToolMessage(
                    content="(blocked - these arguments look like an injection attempt, not calling the tool)",
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                )
        return await handler(request)

    # ---- output guardrail ----
    def after_agent(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        redacted, leaked = _redact(last.content, _SECRET_PATTERNS)

        if leaked:
            logger.warning("output guardrail: redacted a possible leaked secret from the final answer")
            return {"messages": [AIMessage(content=redacted, id=last.id)]}
        return None
