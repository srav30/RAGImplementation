"""FastAPI app for asking questions against the RAG pipeline."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_implementation.llm import DEFAULT_CHAT_MODEL, OpenAIAnswerGenerator
from rag_implementation.models import RagAnswer
from rag_implementation.pipeline import RagPipeline
from rag_implementation.splitter import SentenceSplitter
from rag_implementation.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
)

DEFAULT_DATA_DIRECTORY = "data/aethelgard"
DEFAULT_VECTOR_STORE_PATH = "data/chroma"
DEFAULT_COLLECTION = "rag_chunks"


class AskPipeline(Protocol):
    """Pipeline behavior needed by the API."""

    def ask(self, question: str, top_k: int = 3) -> RagAnswer: ...


class DirectAnswerGenerator(Protocol):
    """Answers questions without retrieved context."""

    def generate_without_context(self, question: str) -> str: ...


class AskSomethingRequest(BaseModel):
    """Request body for asking a question."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1)


class AskSomethingResponse(BaseModel):
    """Response body with answer and retrieved context details."""

    answer: str
    context_used: list[str]
    chunk_ids: list[str]
    num_of_chunks_used: int


def create_app(
    pipeline: AskPipeline | None = None,
    direct_answer_generator: DirectAnswerGenerator | None = None,
    data_directory: str = DEFAULT_DATA_DIRECTORY,
    ingest_on_startup: bool = True,
) -> FastAPI:
    """Create the FastAPI app with an initialized RAG pipeline."""

    rag_pipeline = pipeline or _build_default_pipeline()
    no_rag_answer_generator = direct_answer_generator or OpenAIAnswerGenerator(
        model=DEFAULT_CHAT_MODEL
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if ingest_on_startup and isinstance(rag_pipeline, RagPipeline):
            rag_pipeline.ingest_directory(data_directory)
        app.state.pipeline = rag_pipeline
        yield

    app = FastAPI(title="RAG Implementation API", lifespan=lifespan)

    @app.post("/askSomething/rag", response_model=AskSomethingResponse)
    def ask_something_with_rag(request: AskSomethingRequest) -> AskSomethingResponse:
        answer = app.state.pipeline.ask(request.question, top_k=request.top_k)
        sources = answer.sources
        return AskSomethingResponse(
            answer=answer.answer,
            context_used=[source.chunk.text for source in sources],
            chunk_ids=[source.chunk.id for source in sources],
            num_of_chunks_used=len(sources),
        )

    @app.post("/askSomething/noRag", response_model=AskSomethingResponse)
    def ask_something_without_rag(request: AskSomethingRequest) -> AskSomethingResponse:
        answer = no_rag_answer_generator.generate_without_context(request.question)
        return AskSomethingResponse(
            answer=answer,
            context_used=[],
            chunk_ids=[],
            num_of_chunks_used=0,
        )

    return app


def _build_default_pipeline() -> RagPipeline:
    vector_store = ChromaVectorStore(
        persist_directory=DEFAULT_VECTOR_STORE_PATH,
        collection_name=DEFAULT_COLLECTION,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    return RagPipeline(
        splitter=SentenceSplitter(),
        vector_store=vector_store,
        answer_generator=OpenAIAnswerGenerator(model=DEFAULT_CHAT_MODEL),
    )


app = create_app()
