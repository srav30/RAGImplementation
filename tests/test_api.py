from fastapi.testclient import TestClient

from rag_implementation.api import create_app
from rag_implementation.models import Chunk, RagAnswer, RetrievalResult


class StubPipeline:
    def ask(self, question: str, top_k: int = 3) -> RagAnswer:
        return RagAnswer(
            answer=f"Answer for {question}",
            sources=[
                RetrievalResult(
                    chunk=Chunk(
                        id="aethelgard.txt#sentence-1",
                        document_id="aethelgard.txt",
                        text="The retrieved sentence.",
                    ),
                    score=0.9,
                )
            ][:top_k],
        )


class StubDirectAnswerGenerator:
    def generate_without_context(self, question: str) -> str:
        return f"Direct answer for {question}"


def test_ask_something_rag_returns_answer_and_context() -> None:
    app = create_app(
        pipeline=StubPipeline(),
        direct_answer_generator=StubDirectAnswerGenerator(),
        ingest_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/askSomething/rag",
            json={"question": "What happened?", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for What happened?",
        "context_used": ["The retrieved sentence."],
        "chunk_ids": ["aethelgard.txt#sentence-1"],
        "num_of_chunks_used": 1,
    }


def test_ask_something_no_rag_returns_no_context() -> None:
    app = create_app(
        pipeline=StubPipeline(),
        direct_answer_generator=StubDirectAnswerGenerator(),
        ingest_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/askSomething/noRag",
            json={"question": "What happened?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Direct answer for What happened?",
        "context_used": [],
        "chunk_ids": [],
        "num_of_chunks_used": 0,
    }
