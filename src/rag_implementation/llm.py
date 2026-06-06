"""OpenAI-backed answer generation for retrieved RAG context."""

from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from rag_implementation.models import RetrievalResult

DEFAULT_CHAT_MODEL = "gpt-5.5"


class OpenAIAnswerGenerator:
    """Generate grounded answers from retrieved chunks using OpenAI."""

    def __init__(self, model: str = DEFAULT_CHAT_MODEL) -> None:
        load_dotenv()
        self.model = model
        self.client = OpenAI()

    def generate(self, question: str, results: Sequence[RetrievalResult]) -> str:
        """Return an answer grounded only in retrieved context."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": _build_prompt(question, results),
                },
            ],
        )
        return response.choices[0].message.content or ""

    def generate_without_context(self, question: str) -> str:
        """Return a direct model answer without retrieved context."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        return response.choices[0].message.content or ""


def _build_prompt(question: str, results: Sequence[RetrievalResult]) -> str:
    context = "\n\n".join(
        (
            f"Source: {result.chunk.document_id}\n"
            f"Score: {result.score:.3f}\n"
            f"{result.chunk.text}"
        )
        for result in results
    )
    return f"Question: {question}\n\nContext:\n{context}"
