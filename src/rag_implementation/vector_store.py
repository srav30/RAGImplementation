"""Vector stores for local RAG experiments."""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from rag_implementation.models import Chunk, RetrievalResult

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class InMemoryVectorStore:
    """Store chunks and retrieve by cosine similarity over token counts."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[Counter[str]] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks for retrieval."""
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors.append(_tokenize(chunk.text))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the top matching chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = _tokenize(query)
        scored_results = [
            RetrievalResult(chunk=chunk, score=_cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]

        return [
            result
            for result in sorted(
                scored_results,
                key=lambda item: item.score,
                reverse=True,
            )[:top_k]
            if result.score > 0
        ]


def _tokenize(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(text))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    shared_tokens = left.keys() & right.keys()
    dot_product = sum(left[token] * right[token] for token in shared_tokens)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


class ChromaVectorStore:
    """Persist chunks and OpenAI embeddings in a local Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "rag_chunks",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        load_dotenv()
        self.embedding_model = embedding_model
        self.openai_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Embed chunks and store them in Chroma."""
        if not chunks:
            return

        embeddings = self._embed([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_metadata_for_chunk(chunk) for chunk in chunks],
        )

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the top matching chunks for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self._embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        return _retrieval_results_from_chroma(result)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


def _metadata_for_chunk(chunk: Chunk) -> dict[str, str]:
    return {
        **chunk.metadata,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
    }


def _retrieval_results_from_chroma(result: dict[str, Any]) -> list[RetrievalResult]:
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    retrieval_results = []
    for chunk_id, text, distance, metadata in zip(
        ids,
        documents,
        distances,
        metadatas,
        strict=True,
    ):
        metadata = dict(metadata or {})
        retrieval_results.append(
            RetrievalResult(
                chunk=Chunk(
                    id=str(metadata.pop("chunk_id", chunk_id)),
                    document_id=str(metadata.pop("document_id", "")),
                    text=text or "",
                    metadata={key: str(value) for key, value in metadata.items()},
                ),
                score=1 - float(distance),
            )
        )

    return retrieval_results
