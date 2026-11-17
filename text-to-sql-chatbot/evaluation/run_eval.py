import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.db.metadata_extractor import MetadataExtractor
from src.pipeline.main import build_agent

DB_URL = "postgresql://postgres:postgres@localhost:5432/northwind"
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_queries.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.json")


def load_test_set():
    with open(TEST_SET_PATH, "r") as f:
        return json.load(f)


def build_pipeline():
    extractor = MetadataExtractor(DB_URL)
    metadata = extractor.extract_all()
    return build_agent(metadata, extractor.engine)


def extract_context(result: dict) -> list:
    contexts = []
    for action, observation in result.get("intermediate_steps", []):
        if isinstance(observation, str):
            contexts.append(observation)
    return contexts if contexts else ["No context retrieved."]


def run_agent_on_test_set(agent_executor, test_set: list) -> dict:
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, case in enumerate(test_set):
        question = case["question"]
        print(f"[{i+1}/{len(test_set)}] {question}")

        try:
            result = agent_executor.invoke({"input": question, "chat_history": []})
            answer = result.get("output", "")
            contexts = extract_context(result)
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
    test_set = load_test_set()
    agent_executor = build_pipeline()

    print(f"Running agent on {len(test_set)} questions...")
    raw_results = run_agent_on_test_set(agent_executor, test_set)

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

    scores.to_pandas().to_json(REPORT_PATH, orient="records", indent=2)
    print(f"Full report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()