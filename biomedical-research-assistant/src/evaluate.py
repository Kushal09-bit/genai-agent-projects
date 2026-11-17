"""RAGAS evaluation for the biomedical RAG agent (src/agent.py).

Evaluation questions are generated from randomly sampled, actually-indexed
chunks (not hand-written), so there's no hand-authored "ground truth" that
could drift from what's really in the corpus. Scoring uses reference-free
RAGAS metrics: faithfulness (is the answer supported by the retrieved
context), response relevancy (does the answer address the question), and
context precision (is the retrieved context useful).

Run with: python -m src.evaluate
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import types
import uuid

from dotenv import load_dotenv

load_dotenv()

# ragas 0.4.x unconditionally imports langchain_community.chat_models.vertexai,
# a submodule the pinned langchain-community==0.4.2 no longer ships (split out
# into a standalone integration package upstream). This is a real version
# conflict between ragas and this project's pinned langchain-community — shim
# the missing module out rather than pulling in the actual (unused) Vertex AI
# integration and its GCP dependencies.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        _shim = types.ModuleType("langchain_community.chat_models.vertexai")

        class _MissingChatVertexAI:
            def __init__(self, *a, **kw):
                raise ImportError("Vertex AI integration not installed (unused ragas import).")

        _shim.ChatVertexAI = _MissingChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _shim

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from ragas.run_config import RunConfig

from src.agent import ask
from src.guardrails import OUT_OF_CORPUS_MESSAGE
from src.mcp_server import _collection, retrieve_ranked_chunks

# Best-effort detection of guardrail-rejection boilerplate, so low RAGAS
# scores can be told apart from "the agent actually answered badly" vs "the
# output guardrail replaced the answer with a canned refusal" — the latter
# will always score near-zero on faithfulness/relevancy since it's generic
# text, not a real (good or bad) claim about the retrieved context.
_CANNED_ANSWER_MARKERS = (
    "Blocked:",
    OUT_OF_CORPUS_MESSAGE,
    "The retrieved literature does not report a specific answer",
    "I could not verify this answer",
)


def _is_canned_rejection(answer: str) -> bool:
    return any(answer.startswith(m) or m in answer for m in _CANNED_ANSWER_MARKERS)

MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_SAMPLES = int(os.environ.get("EVAL_N_SAMPLES", "8"))
RESULTS_PATH = "eval_results.json"

QUESTION_GEN_PROMPT = (
    "You write evaluation questions for a biomedical RAG system. Read the "
    "excerpt below and write ONE short factual question that is answerable "
    "directly from it. Return only the question, nothing else.\n\nExcerpt:\n{excerpt}"
)


def _get_llm() -> ChatGroq:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY (holding the Groq key) is not set in .env.")
    return ChatGroq(model=MODEL, temperature=0, api_key=api_key)


def sample_chunks(n: int) -> list[str]:
    total = _collection.count()
    if total == 0:
        raise RuntimeError("Chroma collection is empty — run `python src/ingest.py` first.")
    n = min(n, total)
    offsets = random.sample(range(total), n)
    chunks = []
    for offset in offsets:
        got = _collection.get(limit=1, offset=offset, include=["documents"])
        docs = got.get("documents") or []
        if docs and docs[0].strip():
            chunks.append(docs[0])
    return chunks


def retrieve_contexts(query: str, n_results: int = 8) -> list[str]:
    """Uses the same retrieve_ranked_chunks as the real agent's MCP tool
    (candidate retrieval + cross-encoder rerank), so eval scores reflect
    what the agent actually sees rather than a separate, unreranked path."""
    return [c["text"] for c in retrieve_ranked_chunks(query, n_results)]


def generate_questions(chunks: list[str], llm: ChatGroq) -> list[str]:
    questions = []
    for chunk in chunks:
        response = llm.invoke(QUESTION_GEN_PROMPT.format(excerpt=chunk[:1500]))
        question = response.content.strip().strip('"')
        if question:
            questions.append(question)
    return questions


async def run_eval() -> dict:
    llm = _get_llm()

    print(f"Sampling {N_SAMPLES} indexed chunks and generating questions...")
    chunks = sample_chunks(N_SAMPLES)
    questions = generate_questions(chunks, llm)
    print(f"Generated {len(questions)} questions.")

    samples = []
    sample_meta = []
    skipped = []
    for question in questions:
        # Fresh thread_id per question — these are independent eval samples,
        # not a continued conversation, so they must not share agent memory.
        answer = await ask(question, thread_id=str(uuid.uuid4()))
        contexts = retrieve_contexts(question)
        if not contexts:
            skipped.append({"question": question, "reason": "no retrieved context"})
            continue
        samples.append(
            SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
        )
        sample_meta.append({
            "question": question,
            "answer": answer,
            "guardrail_rejection": _is_canned_rejection(answer),
        })

    print(f"{len(samples)} scored, {len(skipped)} skipped (no retrieved context).")
    if not samples:
        print("Nothing to score.")
        return {"n_scored": 0, "n_skipped": len(skipped), "skipped": skipped}

    dataset = EvaluationDataset(samples)
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    ragas_llm = LangchainLLMWrapper(llm)

    eval_result = evaluate(
        dataset=dataset,
        # strictness=1: ResponseRelevancy defaults to strictness=3, which asks
        # the LLM for 3 completions in one call (n=3). Groq's API caps n at 1
        # and 400s on anything higher, so every default-strictness job fails.
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=ragas_llm,
        embeddings=embeddings,
        # ragas's default max_workers=16 fires that many judge calls at Groq
        # concurrently, which trips Groq's rate limits; the resulting retry
        # backoff then exceeds the default 180s per-job timeout, so jobs die
        # with TimeoutError before ever getting a response. Lower concurrency
        # and a longer timeout give retries room to actually succeed.
        run_config=RunConfig(max_workers=2, timeout=300),
    )

    print(eval_result.to_pandas())

    per_sample = [
        {**meta, **scores} for meta, scores in zip(sample_meta, eval_result.scores)
    ]
    n_guardrail_rejections = sum(1 for m in sample_meta if m["guardrail_rejection"])

    output = {
        "aggregate_scores": eval_result._repr_dict,
        "per_sample_scores": per_sample,
        "n_scored": len(samples),
        "n_skipped": len(skipped),
        "n_guardrail_rejections": n_guardrail_rejections,
        "skipped": skipped,
    }
    if n_guardrail_rejections:
        print(
            f"Note: {n_guardrail_rejections}/{len(samples)} scored samples were "
            "guardrail rejections (canned refusal text), not real answers — "
            "their faithfulness/relevancy scores reflect that boilerplate, "
            "not the agent's actual reasoning quality."
        )
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved detailed results to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    asyncio.run(run_eval())
