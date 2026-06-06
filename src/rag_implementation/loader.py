"""Document loading utilities."""

from pathlib import Path

from rag_implementation.models import Document


class TextFileLoader:
    """Load plain text and Markdown files from a folder."""

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Return documents loaded from supported text files."""
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Document directory does not exist: {root}")

        documents = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue

            documents.append(
                Document(
                    id=str(path.relative_to(root)).replace("\\", "/"),
                    text=path.read_text(encoding="utf-8"),
                    metadata={"path": str(path)},
                )
            )

        return documents
