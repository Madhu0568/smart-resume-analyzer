# RAG Knowledge Assistant

A document question-answering app built with Python, FastAPI, ChromaDB, Sentence Transformers and a local Ollama LLM. Upload PDF/TXT/Markdown files, retrieve relevant passages by vector similarity, and generate answers grounded in those passages with source references.

## Features
- PDF, TXT and Markdown ingestion (scanned PDFs need OCR)
- Word-based chunking with configurable overlap
- Sentence Transformer embeddings (all-MiniLM-L6-v2)
- Persistent ChromaDB vector storage and top-k semantic retrieval
- Local LLM answer generation through Ollama
- Source filename and chunk references
- FastAPI endpoints, Swagger docs and browser UI
- Unit tests for chunking and deterministic IDs

## Requirements
- Python 3.10+
- Ollama installed and running locally
- Model pulled in Ollama (default: llama3.2)
- Internet on first run to download the embedding model

## Windows PowerShell setup
Run in this project folder. Direct venv interpreter calls avoid PowerShell activation-policy issues:
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull llama3.2
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000 or http://127.0.0.1:8000/docs. Ollama must be installed and running. run_windows.bat automates Python setup and startup after Ollama is installed.

## API
- GET /health — health check
- POST /documents — multipart upload with field file; PDF/TXT/MD up to 15 MB
- POST /chat — JSON body: {"question":"What does the document say about Python decorators?"}
Chat responses contain question, answer and sources (filename, chunk number, retrieval distance).

## Configuration
OLLAMA_URL defaults to http://localhost:11434; OLLAMA_MODEL defaults to llama3.2. Chunk size, overlap, embedding model, Chroma path and top-k are in app/config.py.

## Tests
```powershell
.\venv\Scripts\python.exe -m pytest -q
```
Tests do not download models or require ChromaDB/Ollama services.

## Limitations
- Scanned/image-only PDFs need OCR.
- The embedding model downloads on first use.
- Ollama and the selected model must be installed locally for generated answers.
- Verify important answers against the cited source documents.
- No authentication, multi-user isolation, document deletion UI or production deployment hardening yet.
