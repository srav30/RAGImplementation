# RAG Implementation

Python project that implements a small retrieval-augmented generation pipeline.

## Functionality

- Loads `.txt` and `.md` documents from a directory.
- Splits source documents into overlapping word chunks.
- Embeds chunks with OpenAI and stores them in local ChromaDB.
- Retrieves the most relevant chunks by embedding similarity.
- Generates a grounded answer from retrieved context using OpenAI.
- Includes a CLI entry point for local experimentation.

## Run

Add your OpenAI API key to `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

```powershell
cd c:\Users\srava\projects\RAGImplementation
$env:PYTHONPATH="src"
python -m rag_implementation "What was the Zorblax-9 Execution Protocol used for?"
```

If your system `python` is not available, use a virtual environment Python:

```powershell
.\.venv\Scripts\python.exe -m rag_implementation "What was the Zorblax-9 Execution Protocol used for?"
```

Use `--model` to select a different OpenAI model:

```powershell
.\.venv\Scripts\python.exe -m rag_implementation "What was the Zorblax-9 Execution Protocol used for?" --model gpt-5.5
```

Chroma stores embeddings locally in `data/chroma` by default. Use
`--vector-store-path`, `--collection`, or `--embedding-model` to experiment:

```powershell
.\.venv\Scripts\python.exe -m rag_implementation "What changed in the fund?" --data data/aethelgard --collection aethelgard
```

## API

Run the FastAPI server:

```powershell
.\.venv\Scripts\uvicorn.exe rag_implementation.api:app --reload
```

Ask a question with RAG:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/askSomething/rag -ContentType "application/json" -Body '{"question":"What was the Zorblax-9 Execution Protocol used for?","top_k":3}'
```

Ask the model directly without RAG:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/askSomething/noRag -ContentType "application/json" -Body '{"question":"What was the Zorblax-9 Execution Protocol used for?"}'
```

The response includes `answer`, `context_used`, `chunk_ids`, and
`num_of_chunks_used`. The no-RAG endpoint returns empty context fields.

## File Structure

```text
RAGImplementation/
├── README.md
├── pyproject.toml
├── data/
│   ├── aethelgard/
│   │   └── Aethelgard_Financial_2025.txt
│   └── sample/
│       └── rag_overview.md
├── src/
│   └── rag_implementation/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── loader.py
│       ├── main.py
│       ├── models.py
│       ├── pipeline.py
│       ├── splitter.py
│       └── vector_store.py
└── tests/
    └── test_pipeline.py
```

- `loader.py` loads text and Markdown documents from disk.
- `splitter.py` creates searchable chunks from source documents.
- `vector_store.py` provides ChromaDB retrieval with OpenAI embeddings, plus a
  simple in-memory retriever for comparison.
- `pipeline.py` orchestrates ingestion, retrieval, and answer generation.
- `llm.py` calls OpenAI with retrieved context.
- `api.py` exposes the `/askSomething/rag` and `/askSomething/noRag` FastAPI
  endpoints.
- `main.py` provides the command-line entry point.
