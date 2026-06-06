from rag_implementation.models import Document, RetrievalResult
from rag_implementation.pipeline import RagPipeline
from rag_implementation.splitter import SentenceSplitter, TextSplitter
from rag_implementation.vector_store import (
    InMemoryVectorStore,
    _retrieval_results_from_chroma,
)


class StubAnswerGenerator:
    def generate(self, question: str, results: list[RetrievalResult]) -> str:
        return f"LLM answer for: {question} ({len(results)} sources)"


def test_pipeline_retrieves_relevant_context() -> None:
    documents = [
        Document(
            id="rag.md",
            text="RAG retrieves relevant context before generating an answer.",
        ),
        Document(
            id="other.md",
            text="Databases store structured records for applications.",
        ),
    ]
    chunks = TextSplitter(chunk_size=20, chunk_overlap=0).split(documents)
    store = InMemoryVectorStore()
    store.add(chunks)

    results = store.search("How does RAG use context?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.document_id == "rag.md"


def test_pipeline_returns_answer_with_sources(tmp_path) -> None:
    sample_file = tmp_path / "sample.md"
    sample_file.write_text(
        "RAG loads documents, retrieves chunks, and generates grounded answers.",
        encoding="utf-8",
    )
    pipeline = RagPipeline(splitter=TextSplitter(chunk_size=20, chunk_overlap=0))

    chunk_count = pipeline.ingest_directory(str(tmp_path))
    answer = pipeline.ask("What does RAG retrieve?")

    assert chunk_count == 1
    assert "retrieves chunks" in answer.answer
    assert answer.sources


def test_pipeline_can_use_answer_generator(tmp_path) -> None:
    sample_file = tmp_path / "sample.md"
    sample_file.write_text(
        "RAG loads documents, retrieves chunks, and generates grounded answers.",
        encoding="utf-8",
    )
    pipeline = RagPipeline(
        splitter=TextSplitter(chunk_size=20, chunk_overlap=0),
        answer_generator=StubAnswerGenerator(),
    )

    pipeline.ingest_directory(str(tmp_path))
    answer = pipeline.ask("What does RAG retrieve?")

    assert answer.answer == "LLM answer for: What does RAG retrieve? (1 sources)"
    assert answer.sources


def test_sentence_splitter_creates_one_chunk_per_sentence() -> None:
    documents = [
        Document(
            id="aethelgard.txt",
            text=(
                "1. Executive Summary & Asset Allocation\n"
                "AMCF gained a 4.2-millisecond advantage. "
                "The fund shifted 14.2% of its liquid capital.\n\n"
                "2. Synthetic Derivative Structures\n"
                "The position paid $8.4 million only under strict conditions."
            ),
        )
    ]

    chunks = SentenceSplitter().split(documents)

    assert [chunk.id for chunk in chunks] == [
        "aethelgard.txt#sentence-1",
        "aethelgard.txt#sentence-2",
        "aethelgard.txt#sentence-3",
        "aethelgard.txt#sentence-4",
        "aethelgard.txt#sentence-5",
    ]
    assert [chunk.text for chunk in chunks] == [
        "1. Executive Summary & Asset Allocation",
        "AMCF gained a 4.2-millisecond advantage.",
        "The fund shifted 14.2% of its liquid capital.",
        "2. Synthetic Derivative Structures",
        "The position paid $8.4 million only under strict conditions.",
    ]
    assert chunks[1].metadata["sentence_index"] == "2"


def test_chroma_results_are_converted_to_retrieval_results() -> None:
    results = _retrieval_results_from_chroma(
        {
            "ids": [["chunk-1"]],
            "documents": [["Stored sentence."]],
            "distances": [[0.25]],
            "metadatas": [
                [
                    {
                        "chunk_id": "document.txt#sentence-1",
                        "document_id": "document.txt",
                        "sentence_index": "1",
                    }
                ]
            ],
        }
    )

    assert len(results) == 1
    assert results[0].chunk.id == "document.txt#sentence-1"
    assert results[0].chunk.document_id == "document.txt"
    assert results[0].chunk.text == "Stored sentence."
    assert results[0].chunk.metadata == {"sentence_index": "1"}
    assert results[0].score == 0.75
