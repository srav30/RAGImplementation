# RAG Pipeline Overview

Retrieval-augmented generation, or RAG, answers questions by retrieving relevant
context before generating a response. A typical RAG pipeline loads source
documents, splits them into chunks, indexes the chunks, retrieves the most
relevant context for a user question, and generates a grounded answer.

This project starts with a local in-memory retriever so the pipeline can run
without external services. Later versions can replace the local retriever with a
vector database and replace the extractive answer generator with an LLM call.
