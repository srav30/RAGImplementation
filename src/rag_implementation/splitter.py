"""Text splitting for RAG ingestion."""

import re

from rag_implementation.models import Chunk, Document

HEADING_PATTERN = re.compile(r"^(?:Part\s+\d+:|\d+\.\s+).+", re.IGNORECASE)
SENTENCE_ENDINGS = ".!?"


class TextSplitter:
    """Split documents into overlapping word chunks."""

    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 30) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Return chunks for all documents."""
        chunks = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        if not words:
            return []

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for chunk_index, start in enumerate(range(0, len(words), step), start=1):
            chunk_words = words[start : start + self.chunk_size]
            if not chunk_words:
                continue

            chunks.append(
                Chunk(
                    id=f"{document.id}#{chunk_index}",
                    document_id=document.id,
                    text=" ".join(chunk_words),
                    metadata=document.metadata,
                )
            )

        return chunks


class SentenceSplitter:
    """Split documents into one sentence per chunk."""

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Return sentence chunks for all documents."""
        chunks = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Chunk]:
        chunks = []
        for sentence_index, sentence in enumerate(
            _split_sentences(document.text),
            start=1,
        ):
            chunks.append(
                Chunk(
                    id=f"{document.id}#sentence-{sentence_index}",
                    document_id=document.id,
                    text=sentence,
                    metadata={
                        **document.metadata,
                        "sentence_index": str(sentence_index),
                    },
                )
            )

        return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if _is_heading(line):
            sentences.append(line)
        else:
            sentences.extend(_split_line_sentences(line))

    return sentences


def _is_heading(line: str) -> bool:
    return not line.endswith(tuple(SENTENCE_ENDINGS)) and bool(
        HEADING_PATTERN.match(line)
    )


def _split_line_sentences(line: str) -> list[str]:
    sentences = []
    start = 0
    for index, character in enumerate(line):
        if character not in SENTENCE_ENDINGS:
            continue
        if character == "." and not _is_period_sentence_boundary(line, start, index):
            continue

        sentence = line[start : index + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = index + 1

    remainder = line[start:].strip()
    if remainder:
        sentences.append(remainder)

    return sentences


def _is_period_sentence_boundary(line: str, start: int, index: int) -> bool:
    previous_character = line[index - 1] if index > 0 else ""
    next_character = line[index + 1] if index + 1 < len(line) else ""
    if previous_character.isdigit() and next_character.isdigit():
        return False

    fragment = line[start:index].strip()
    return not fragment.isdigit()
