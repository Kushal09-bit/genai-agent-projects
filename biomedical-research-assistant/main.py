"""CLI entrypoint: ask questions against the biomedical RAG agent interactively."""
import asyncio
import uuid

from dotenv import load_dotenv

from src.agent import ask

load_dotenv()


async def main() -> None:
    # One thread_id for the whole CLI run, so the agent remembers earlier
    # questions in this session. Restarting the CLI starts a fresh thread.
    thread_id = str(uuid.uuid4())

    print("Biomedical RAG assistant. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = await ask(question, thread_id=thread_id)
        except Exception as e:
            print(f"[error] {e}")
            continue

        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
