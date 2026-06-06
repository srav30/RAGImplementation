"""RAG pipeline orchestration."""

from typing import Protocol

from rag_implementation.loader import TextFileLoader
from rag_implementation.models import Chunk, Document, RagAnswer, RetrievalResult
from rag_implementation.splitter import TextSplitter
from rag_implementation.vector_store import InMemoryVectorStore


class Splitter(Protocol):
    """Splits loaded documents into chunks."""

    def split(self, documents: list[Document]) -> list[Chunk]: ...


class VectorStore(Protocol):
    """Indexes chunks and retrieves relevant matches."""

    def add(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]: ...


class AnswerGenerator(Protocol):
    """Produces a final answer from a question and retrieved context."""

    def generate(self, question: str, results: list[RetrievalResult]) -> str: ...


class RagPipeline:
    """Load documents, retrieve relevant chunks, and synthesize an answer."""

    def __init__(
        self,
        loader: TextFileLoader | None = None,
        splitter: Splitter | None = None,
        vector_store: VectorStore | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.loader = loader or TextFileLoader()
        self.splitter = splitter or TextSplitter()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.answer_generator = answer_generator

    def ingest_directory(self, directory: str) -> int:
        """Load files from a directory and index their chunks."""
        documents = self.loader.load_directory(directory)
        chunks = self.splitter.split(documents)
        self.vector_store.add(chunks)
        return len(chunks)

    def ask(self, question: str, top_k: int = 3) -> RagAnswer:
        """Retrieve context and generate an answer."""
        results = self.vector_store.search(question, top_k=top_k)
        if not results:
            return RagAnswer(
                answer="I could not find enough relevant context to answer that.",
                sources=[],
            )

        if self.answer_generator is None:
            answer = _generate_answer(question, results)
        else:
            answer = self.answer_generator.generate(question, results)

        return RagAnswer(answer=answer, sources=results)


def _generate_answer(question: str, results: list[RetrievalResult]) -> str:
    """Generate a simple grounded answer from retrieved context.

    This is intentionally extractive for now. It keeps the first version local and
    deterministic, while leaving a clear place to plug in an LLM later.
    """
    context = "\n\n".join(
        f"Source: {result.chunk.document_id}\n{result.chunk.text}" for result in results
    )
    return (
        f"Question: {question}\n\n"
        "Answer based on retrieved context:\n"
        f"{context}"
    )
