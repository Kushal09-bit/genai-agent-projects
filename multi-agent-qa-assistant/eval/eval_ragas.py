import asyncio
import json
import os
import sys
import types

# This file lives in eval/, one level below the repo root where graph.py,
# tracing.py, and mcp_layer/ actually live. Put the repo root on sys.path so
# `python eval/eval_ragas.py` works from anywhere, not just when the repo
# root happens to already be on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ragas 0.4.x unconditionally imports langchain_community.chat_models.vertexai,
# a submodule current langchain-community no longer ships (split out into a
# standalone integration package upstream). Stub it out rather than pinning
# to an old, incompatible langchain-community/langchain combo.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - unused placeholder
        pass

    _shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from langfuse import get_client

from graph import build_graph
from mcp_layer.mcp_client import call_tool
from tracing import langfuse_handler

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "multiagent_eval_results.json")

# Ground truths were captured from real graph runs against the live Chinook
# Postgres DB and the ingested PubMed corpus (see the smoke tests these
# questions/answers were validated against before being pinned here).
TEST_SET = [
    {
        "question": "Which genre has the most tracks?",
        "ground_truth": "Rock has the most tracks, with 1,297 tracks.",
    },
    {
        "question": "How many customers do we have from Germany?",
        "ground_truth": "There are 4 customers from Germany.",
    },
    {
        "question": "What was the response rate for Iberdomide in the lupus trial?",
        "ground_truth": (
            "Between 22% and 50% of patients receiving iberdomide achieved a "
            "≥24-point reduction in hybrid SELENA-SLEDAI scores, compared "
            "with 13% of patients receiving placebo."
        ),
    },
    {
        "question": "What did the pneumonia study find about treatment outcomes?",
        "ground_truth": (
            "Treatment success was the most commonly reported outcome, appearing "
            "in about 60% of the 280 randomized controlled trials reviewed, but "
            "there is no standard validated instrument for measuring it and "
            "outcome reporting varies widely across trials."
        ),
    },
]


async def _reconstruct_context(route: str, result: dict, question: str) -> list:
    """Ragas needs the actual retrieved context an answer was grounded in.
    The graph state doesn't carry that around, so re-derive it via the same
    read-only MCP tools sql_node/rag_node already called: the same SQL query
    (sql_query is already in state) or the same retrieval query — both
    deterministic against unchanged data, so this reflects what the answer
    was really grounded in without needing to touch node internals."""
    contexts = []
    if route in ("sql", "both") and result.get("sql_query"):
        sql_result = await call_tool("execute_sql", sql=result["sql_query"])
        contexts.append(sql_result)
    if route in ("rag", "both"):
        raw = await call_tool("retrieve_chunks", query=question, n_results=5)
        try:
            chunks = json.loads(raw).get("chunks", [])
            contexts.extend(c["text"] for c in chunks)
        except (json.JSONDecodeError, TypeError):
            pass
    return contexts if contexts else ["No context retrieved."]


def run_agent_on_test_set(app, test_set: list) -> dict:
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, case in enumerate(test_set):
        question = case["question"]
        print(f"[{i+1}/{len(test_set)}] {question}")

        config = {
            "configurable": {"thread_id": f"ragas-eval-{i}"},
            "callbacks": [langfuse_handler],
        }
        try:
            result = app.invoke({"question": question}, config)
            answer = result["final_answer"]
            contexts = asyncio.run(_reconstruct_context(result["route"], result, question))
        except Exception as e:
            answer = f"ERROR: {e}"
            contexts = ["No context retrieved due to error."]

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(case["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def main():
    app = build_graph()

    print(f"Running the multi-agent graph on {len(TEST_SET)} questions...")
    raw_results = run_agent_on_test_set(app, TEST_SET)

    dataset = Dataset.from_dict(raw_results)

    judge_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    judge_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    print("Grading with Ragas...")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    print("\n=== SCORES ===")
    print(scores)

    scores.to_pandas().to_json(RESULTS_PATH, orient="records", indent=2)
    print(f"Full report saved to {RESULTS_PATH}")

    get_client().flush()


if __name__ == "__main__":
    main()
