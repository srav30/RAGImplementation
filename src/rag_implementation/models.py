"""Shared models for the RAG pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """Source document loaded into the RAG pipeline."""

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Searchable chunk derived from a source document."""

    id: str
    document_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RagAnswer:
    """Final answer and supporting retrieval context."""

    answer: str
    sources: list[RetrievalResult]
