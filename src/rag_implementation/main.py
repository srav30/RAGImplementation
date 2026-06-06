"""Command-line entry point for the RAG pipeline."""

import argparse
from pathlib import Path

from rag_implementation.llm import DEFAULT_CHAT_MODEL, OpenAIAnswerGenerator
from rag_implementation.pipeline import RagPipeline
from rag_implementation.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local RAG pipeline.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What does this project implement?",
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--data",
        default="data/aethelgard",
        help="Directory containing .txt or .md documents.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to use.")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAT_MODEL,
        help="OpenAI model to use for answer generation.",
    )
    parser.add_argument(
        "--vector-store-path",
        default="data/chroma",
        help="Directory where Chroma stores embeddings.",
    )
    parser.add_argument(
        "--collection",
        default="rag_chunks",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model to use for retrieval.",
    )
    args = parser.parse_args()

    vector_store = ChromaVectorStore(
        persist_directory=args.vector_store_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    pipeline = RagPipeline(
        vector_store=vector_store,
        answer_generator=OpenAIAnswerGenerator(model=args.model),
    )
    chunk_count = pipeline.ingest_directory(args.data)
    answer = pipeline.ask(args.question, top_k=args.top_k)

    print(f"Indexed {chunk_count} chunks from {Path(args.data)}")
    print()
    print(answer.answer)

    if answer.sources:
        print()
        print("Sources:")
        for source in answer.sources:
            print(f"- {source.chunk.document_id} (score={source.score:.3f})")


if __name__ == "__main__":
    main()
