import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────────────────
# 4.1 — Input guardrail
# ─────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"\bignore (all|any|the) (previous|prior|above)\b.*\binstructions?\b",
    r"\bdisregard (all|any|the) (previous|prior|above)\b",
    r"\breveal (your|the) (system prompt|instructions)\b",
    r"\blist (every|all) (chunk|document)s?\b",
    r"\byou are now\b",
    r"\bnew instructions?:\b",
]

JAILBREAK_PATTERNS = [
    r"\bpretend (you are|to be)\b",
    r"\bact as\b.*\b(doctor|physician|no restrictions?)\b",
    r"\bwith no restrictions?\b",
    r"\bwithout (any )?(restrictions?|limitations?|filters?)\b",
    r"\bDAN\b",
    r"\byou have no (rules|restrictions|limits)\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


INTENT_CLASSIFIER_TEMPLATE = """You are a safety classifier for a biomedical literature assistant.
This assistant answers questions about biomedical and clinical research topics in general
(the underlying corpus covers topics like lupus, pediatrics, pneumonia, neurophysiology, and pain research,
but you should classify based on whether the QUESTION ITSELF is a biomedical/clinical topic,
not whether you know if the specific corpus covers it — a separate retrieval step handles that).

Recent conversation, oldest first (may be "(none)" if this is the first question this session):
{history}

If the current question is a natural follow-up to that conversation — asking for more detail, a
page number, a citation, or clarification about something just discussed — classify it based on
the topic actually being followed up on, not the follow-up phrasing taken in isolation. For
example, if the prior turn discussed a lupus drug trial, "what page was that on?" is SAFE, not
OFF_TOPIC, because it's asking for a citation about that same biomedical topic.

Classify the user's question into exactly one category:

- SAFE: any question about biomedical, clinical, medical, or health research topics in general
  (e.g. "what causes diabetes", "side effects of chemotherapy", "how does the immune system work"),
  including in-context follow-ups about such a topic as described above
- CLINICAL_ADVICE: a request for personal medical advice, diagnosis, treatment recommendations,
  or dosing guidance directed at the user or a patient (e.g. "should I take this drug",
  "what dose should I give my patient", "do I have lupus")
- OFF_TOPIC: has NOTHING to do with biomedical/medical/health/clinical topics at all, even
  considering the conversation above (e.g. weather, geography, sports, cooking, general trivia,
  coding help)

User question: {question}

Respond with only one word: SAFE, CLINICAL_ADVICE, or OFF_TOPIC."""


class InputGuardrail:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        llm = ChatGroq(model=model, temperature=0, api_key=api_key)
        prompt = PromptTemplate(
            input_variables=["question", "history"], template=INTENT_CLASSIFIER_TEMPLATE
        )
        self.chain = prompt | llm | StrOutputParser()

    def check(self, question: str, history: str = "") -> tuple[bool, str]:
        if _matches_any(question, INJECTION_PATTERNS):
            return False, "Blocked: detected a potential prompt injection attempt"

        if _matches_any(question, JAILBREAK_PATTERNS):
            return False, "Blocked: detected a potential jailbreak attempt"

        try:
            classification = self.chain.invoke(
                {"question": question, "history": history.strip() or "(none)"}
            ).strip().upper()
        except Exception as e:
            return False, f"Blocked: classifier error, failing safe ({e})"

        if "CLINICAL_ADVICE" in classification:
            return False, ("Blocked: this assistant summarizes published research findings only "
                            "and cannot provide personal medical advice, diagnosis, or treatment "
                            "recommendations. Please consult a qualified healthcare provider.")
        if "OFF_TOPIC" in classification:
            return False, "Blocked: classified as unrelated to the biomedical research corpus"

        return True, "OK"


# ─────────────────────────────────────────────────────────
# 4.2 — Citation enforcement
# ─────────────────────────────────────────────────────────
#
# Design decision: rather than pattern-matching every possible way the LLM
# might phrase "I don't have a citable answer" (fragile — wording varies
# every run), the agent's prompt requires it to emit an explicit marker,
# NO_CITABLE_ANSWER, when it has nothing citable. This is checked here
# before falling back to the citation-count check.

NO_CITABLE_MARKER = "NO_CITABLE_ANSWER"


def enforce_citations(answer: str, citations: list) -> tuple[bool, str]:
    """Reject a non-empty answer that has no supporting citations, unless it
    explicitly signals it has no citable answer via NO_CITABLE_MARKER."""
    has_answer = bool(answer and answer.strip())
    has_citations = bool(citations) and len(citations) > 0

    if not has_answer:
        return True, "OK — empty answer, nothing to enforce"

    if NO_CITABLE_MARKER in answer:
        return True, "OK — explicit no-citable-answer signal"

    if has_citations:
        return True, "OK"

    return False, "REJECTED: non-empty answer with zero citations — treated as unsupported/hallucinated"


# ─────────────────────────────────────────────────────────
# 4.3 — Grounding check (second pass)
# ─────────────────────────────────────────────────────────

GROUNDING_CHECK_TEMPLATE = """You are verifying whether an answer is actually supported by its cited source text.

Original question: {question}

Answer given: {answer}

Cited source text:
{cited_chunks}

Compare the answer's claims against the cited source text carefully.
- GROUNDED: every claim in the answer is clearly supported by the cited text
- UNGROUNDED: the answer's claims are not supported by the cited text, or contradict it
- PARTIAL: some claims are supported but others are not

Respond with only one word: GROUNDED, UNGROUNDED, or PARTIAL."""


class GroundingChecker:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        llm = ChatGroq(model=model, temperature=0, api_key=api_key)
        prompt = PromptTemplate(
            input_variables=["question", "answer", "cited_chunks"],
            template=GROUNDING_CHECK_TEMPLATE,
        )
        self.chain = prompt | llm | StrOutputParser()

    def check(self, question: str, answer: str, cited_chunks: str) -> str:
        if not cited_chunks or not cited_chunks.strip():
            return "UNGROUNDED"

        try:
            result = self.chain.invoke({
                "question": question,
                "answer": answer,
                "cited_chunks": cited_chunks,
            }).strip().upper()
        except Exception:
            return "UNGROUNDED"

        if "PARTIAL" in result:
            return "PARTIAL"
        if "UNGROUNDED" in result:
            return "UNGROUNDED"
        if "GROUNDED" in result:
            return "GROUNDED"
        return "UNGROUNDED"


# ─────────────────────────────────────────────────────────
# Output guardrail
# ─────────────────────────────────────────────────────────

class OutputGuardrail:
    """Runs after the agent generates an answer. Enforces citations and
    verifies the answer is actually grounded in the cited source text."""

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.grounding_checker = GroundingChecker(model=model)

    def check(self, question: str, answer: str, citations: list, cited_chunks_text: str) -> tuple[str, str]:
        # Explicit no-citable-answer signal — bypass everything else
        if NO_CITABLE_MARKER in answer:
            return "OK", "The retrieved literature does not report a specific answer to this question."

        # Step 1: citation enforcement
        ok, reason = enforce_citations(answer, citations)
        if not ok:
            return "REJECTED", ("I could not verify this answer with proper citations. "
                                 "Please try rephrasing your question.")

        # Step 2: grounding check
        verdict = self.grounding_checker.check(question, answer, cited_chunks_text)

        if verdict == "GROUNDED":
            return "GROUNDED", answer

        if verdict == "PARTIAL":
            caveat = "\n\n[Note: only part of this answer could be verified against the source material.]"
            return "PARTIAL", answer + caveat

        return "UNGROUNDED", ("I could not verify this answer against the source material. "
                               "Please try rephrasing your question.")


# ─────────────────────────────────────────────────────────
# 4.4 — Out-of-corpus fallback
# ─────────────────────────────────────────────────────────

OUT_OF_CORPUS_MESSAGE = "I could not find this in the available literature."
NO_RESULTS_SENTINEL = "NO_RESULTS"