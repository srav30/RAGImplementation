"""Command-line entry point for ingesting documents into ChromaDB."""

import argparse
from pathlib import Path

from rag_implementation.pipeline import RagPipeline
from rag_implementation.splitter import SentenceSplitter
from rag_implementation.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--data",
        default="data/aethelgard",
        help="Directory containing .txt or .md documents.",
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
        help="OpenAI embedding model to use.",
    )
    args = parser.parse_args()

    vector_store = ChromaVectorStore(
        persist_directory=args.vector_store_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    pipeline = RagPipeline(
        splitter=SentenceSplitter(),
        vector_store=vector_store,
    )
    chunk_count = pipeline.ingest_directory(args.data)

    print(f"Indexed {chunk_count} chunks from {Path(args.data)}")
    print(f"Vector store: {Path(args.vector_store_path)}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
