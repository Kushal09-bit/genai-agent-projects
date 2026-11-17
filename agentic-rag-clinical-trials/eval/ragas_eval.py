
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import ToolMessage
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from sentence_transformers import SentenceTransformer

from src import aact_db
from src.agent.agent import _build_llm, _flush_langfuse, _get_langfuse_handler, build_agent
from src.config import AACT_DB_HOST, AGENT_MODEL, EMBED_MODEL_NAME, GEMINI_MODEL, OLLAMA_MODEL

RESULTS_PATH = Path(__file__).parent / "ragas_results.json"

# groq's openai/gpt-oss-120b sits on an 8000 tokens-per-minute budget on
# this project's key. a single eval question can involve several tool-
# calling round trips, each one resending the growing message history, so
# back-to-back questions can blow that budget even though no individual
# request is huge. spacing questions out gives the per-minute bucket time
# to refill instead of relying on groq's own retry (which just re-sends
# the same request and hits the same wall). tune this down if the run
# still 413s, or down if eval is running slower than it needs to.
INTER_QUESTION_DELAY_SECONDS = 15


class _SentenceTransformerEmbeddings(BaseRagasEmbeddings):
    """ragas ships its own HuggingfaceEmbeddings class, but as of the
    version this pins (checked both 0.3.9 and the newest 0.4.3) it's
    actually broken - `HuggingfaceEmbeddings(model_name=...)` raises
    "Can't instantiate abstract class ... abstract methods aembed_documents,
    aembed_query" because the class never actually implements the async
    half of ragas's own embeddings interface. rolling this instead - same
    sentence-transformers model retrieval/long-term-memory already load
    (EMBED_MODEL_NAME), just wrapped to satisfy ragas's 4-method interface
    directly. sync calls do the real work; async ones just run that in a
    thread so ragas's own async evaluation loop doesn't block on them."""

    def __init__(self, model_name: str):
        super().__init__()
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


# ---------- live aact ground truth ----------
# small helper queries, kept separate from the eval set below so each
# question's ground_truth can just point at one of these instead of
# repeating sql inline. all go through aact_db.run_query/get_study/
# get_conditions - the same functions schema_tool/sql_exec_tool/
# linkage_lookup wrap in mcp_server.py, just called directly here instead
# of through mcp (see module docstring above for why that's fine in this
# one file).

def _lassa_trial_count():
    rows = aact_db.run_query("""
        SELECT COUNT(DISTINCT s.nct_id) AS n
        FROM ctgov.studies s JOIN ctgov.conditions c ON c.nct_id = s.nct_id
        WHERE c.name ILIKE '%lassa%'
    """)
    return f"{rows[0]['n']} trials in AACT have a condition matching 'Lassa fever'."


def _lassa_recruiting_count():
    rows = aact_db.run_query("""
        SELECT COUNT(DISTINCT s.nct_id) AS n
        FROM ctgov.studies s JOIN ctgov.conditions c ON c.nct_id = s.nct_id
        WHERE c.name ILIKE '%lassa%' AND s.overall_status = 'RECRUITING'
    """)
    return f"{rows[0]['n']} of those Lassa fever trials currently have overall_status RECRUITING."


def _lassa_sample_ncts():
    rows = aact_db.run_query("""
        SELECT DISTINCT s.nct_id
        FROM ctgov.studies s JOIN ctgov.conditions c ON c.nct_id = s.nct_id
        WHERE c.name ILIKE '%lassa%'
        ORDER BY s.nct_id LIMIT 5
    """)
    return ", ".join(r["nct_id"] for r in rows) if rows else "no matching trials found in AACT"


def _kisumu_field(field):
    study = aact_db.get_study("NCT03762382")
    return str(study[field]) if study and study.get(field) is not None else "not found in AACT"


def _kisumu_conditions():
    conditions = aact_db.get_conditions("NCT03762382")
    return ", ".join(conditions) if conditions else "no conditions listed in AACT"


# ---------- labelled eval set ----------
# 16 questions, mixing structured (aact) / unstructured (papers) /
# cross-source pivot - spec wants a "15+ question labelled multi-hop eval
# set" covering both sources. paper ground truths were read straight out
# of docs/*.pdf; aact/cross-source ones resolve live at eval time (see
# helpers above) instead of being guessed and baked in.
#
# NCT03762382 shows up a lot below on purpose - it's a real trial that's
# both registered in AACT AND the subject of one of the ingested papers
# (the Kisumu Combined Ring Study), so it's the one genuine, verifiable
# cross-source anchor this eval set has. everything else is single-source.

EVAL_SET = [
    # ---- paper-only (unstructured retrieval) ----
    {
        "source": "paper",
        "question": "According to the Lassa fever therapeutics paper, how many confirmed Lassa fever cases were reported in Nigeria in 2023, and what was the case-fatality ratio?",
        "ground_truth": "1,067 confirmed cases were reported in Nigeria in 2023, resulting in 189 deaths, for a case-fatality ratio (CFR) of 18%.",
    },
    {
        "source": "paper",
        "question": "What year was the West Africa Lassa Fever Consortium (WALC) established?",
        "ground_truth": "WALC was established in 2021.",
    },
    {
        "source": "paper",
        "question": "In the cross-over trial on hypnotic analgesia and tDCS for fibromyalgia, how many participants were enrolled and what was their age range?",
        "ground_truth": "18 women with fibromyalgia were enrolled, aged 18 to 65 years old.",
    },
    {
        "source": "paper",
        "question": "What were the two primary outcomes measured in the fibromyalgia hypnotic-analgesia/tDCS study?",
        "ground_truth": "The cold pressor test (CPT) and motor evoked potential (MEP) amplitude were the two primary outcomes.",
    },
    {
        "source": "paper",
        "question": "What is the NCT ID of the Kisumu Combined Ring Study on intravaginal rings?",
        "ground_truth": "NCT03762382.",
    },
    {
        "source": "paper",
        "question": "How many women participated in the qualitative interviews of the Kisumu Combined Ring Study, and what was their age range?",
        "ground_truth": "25 cisgender women aged 18 to 34 years participated in the qualitative interviews.",
    },
    {
        "source": "paper",
        "question": "In the OCD rTMS study, how many patients participated and which brain region did the stimulation target?",
        "ground_truth": "18 OCD patients participated, and low-frequency rTMS targeted the supplementary motor area (SMA).",
    },
    # ---- aact-only (structured) ----
    {
        "source": "aact",
        "question": "How many distinct trials in AACT have a condition matching 'Lassa fever'?",
        "ground_truth": _lassa_trial_count,
    },
    {
        "source": "aact",
        "question": "Of the AACT trials with a condition matching 'Lassa fever', how many currently have overall_status RECRUITING?",
        "ground_truth": _lassa_recruiting_count,
    },
    {
        "source": "aact",
        "question": "What is the overall_status AACT lists for trial NCT03762382?",
        "ground_truth": lambda: _kisumu_field("overall_status"),
    },
    {
        "source": "aact",
        "question": "What phase is trial NCT03762382 in, according to AACT?",
        "ground_truth": lambda: _kisumu_field("phase"),
    },
    {
        "source": "aact",
        "question": "What official title does AACT list for trial NCT03762382?",
        "ground_truth": lambda: _kisumu_field("official_title"),
    },
    # ---- cross-source pivot ----
    {
        "source": "cross-source",
        "question": "The Kisumu Combined Ring Study paper cites NCT03762382 - what conditions does AACT list for that trial?",
        "ground_truth": _kisumu_conditions,
    },
    {
        "source": "cross-source",
        "question": "What brief_title does AACT list for the trial described in the Kisumu Combined Ring Study paper (NCT03762382)?",
        "ground_truth": lambda: _kisumu_field("brief_title"),
    },
    {
        "source": "cross-source",
        "question": "Is trial NCT03762382, the one referenced in the Kisumu Combined Ring Study paper, still recruiting according to AACT?",
        "ground_truth": lambda: _kisumu_field("overall_status"),
    },
    {
        "source": "cross-source",
        "question": "List up to 5 NCT IDs of AACT trials studying Lassa fever, the disease covered in the Lassa fever therapeutics paper.",
        "ground_truth": _lassa_sample_ncts,
    },
]


# ---------- run each question through the real agent ----------

async def _run_question(agent, entry: dict) -> dict | None:
    ground_truth = entry["ground_truth"]
    if callable(ground_truth):
        try:
            ground_truth = ground_truth()
        except Exception as e:
            print(f"  skipping (couldn't get live ground truth from AACT): {e}")
            return None

    thread_id = str(uuid.uuid4())  # fresh thread per question - no memory leaking between eval questions
    config = {"configurable": {"thread_id": thread_id}}
    handler = _get_langfuse_handler()
    if handler:
        config["callbacks"] = [handler]

    # one bad question (groq TPM 413s the most common way - a single
    # multi-hop question's tool-result context can outweigh the per-minute
    # token budget on its own, so retrying the same request just fails
    # again) used to take the whole 16-question run down with it. catch
    # broadly here on purpose - agent.ainvoke() can surface this as several
    # different exception types depending on which layer (groq sdk,
    # langchain-groq, langgraph) ends up raising it, and any of them should
    # mean "skip this one question", not "kill the script".
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": entry["question"]}]}, config=config)
    except Exception as e:
        print(f"  skipping (agent run failed - {type(e).__name__}: {e})")
        return None
    messages = result["messages"]
    answer = messages[-1].content

    # contexts for ragas = whatever the tools actually returned this run -
    # same source working_memory.py's confirmed_facts comes from, just not
    # truncated to FACT_SNIPPET_CHARS since ragas needs the real content to
    # judge faithfulness/context_precision against, not a 500-char snippet
    contexts = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
    if not contexts:
        contexts = ["(no tool was called for this question)"]  # ragas needs a non-empty list either way

    return {
        "source": entry["source"],
        "question": entry["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


async def _collect_rows() -> list[dict]:
    print("building agent...")
    agent = await build_agent()

    rows = []
    for i, entry in enumerate(EVAL_SET, 1):
        if entry["source"] != "paper" and not AACT_DB_HOST:
            print(f"[{i}/{len(EVAL_SET)}] skipping (AACT_DB_HOST not set): {entry['question'][:60]}...")
            continue
        print(f"[{i}/{len(EVAL_SET)}] {entry['question'][:70]}")
        row = await _run_question(agent, entry)
        if row:
            rows.append(row)

        if i < len(EVAL_SET):
            await asyncio.sleep(INTER_QUESTION_DELAY_SECONDS)

    skipped = len(EVAL_SET) - len(rows)
    print(f"\ncollected {len(rows)}/{len(EVAL_SET)} questions ({skipped} skipped - see messages above for why)")

    _flush_langfuse()  # short-lived script - flush now or traces from this run never make it out
    return rows


# ---------- score with ragas ----------

def _score_with_ragas(rows: list[dict]) -> dict:
    # ragas needs its own judge llm + embeddings to compute these metrics -
    # reusing the exact same model the agent itself ran on (_build_llm()
    # picks groq/gemini/ollama the same way agent.py's build_agent() does,
    # see config.py) and the same sentence-transformers model retrieval/
    # long-term-memory already use, so this doesn't pull in a second
    # embedding model just for eval.
    judge_llm = LangchainLLMWrapper(_build_llm())
    embeddings = _SentenceTransformerEmbeddings(EMBED_MODEL_NAME)

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": r["ground_truth"],
            }
            for r in rows
        ]
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
    )

    # .to_pandas() instead of pulling aggregate scores off the result
    # object directly - ragas has reshuffled that object's api across
    # versions a few times, to_pandas() has stayed stable, and it gives
    # per-row scores for free too (attached back onto rows below), not
    # just the averages.
    df = result.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for row, (_, scored) in zip(rows, df.iterrows()):
        row["ragas_scores"] = {m: float(scored[m]) for m in metric_names if m in scored}

    return {m: float(df[m].mean()) for m in metric_names if m in df.columns}


def _write_results(rows: list[dict], averages: dict) -> None:
    # report whichever provider actually ran, not just AGENT_MODEL's groq
    # default - same env-presence checks _build_llm() uses, same order.
    if os.getenv("OLLAMA_API_KEY"):
        agent_model = OLLAMA_MODEL
    elif os.getenv("GOOGLE_API_KEY"):
        agent_model = GEMINI_MODEL
    else:
        agent_model = AGENT_MODEL
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "agent_model": agent_model,
        "num_questions": len(rows),
        "averages": averages,
        "results": rows,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))


def main():
    rows = asyncio.run(_collect_rows())
    if not rows:
        print("nothing to score - no questions ran (check chroma_db/ and AACT_DB_HOST)")
        return

    print(f"scoring {len(rows)} answers with ragas...")
    averages = _score_with_ragas(rows)
    _write_results(rows, averages)

    print(f"\ndone - {len(rows)} questions scored, results written to {RESULTS_PATH}")
    for metric, score in averages.items():
        print(f"  {metric}: {score:.3f}")


if __name__ == "__main__":
    main()
