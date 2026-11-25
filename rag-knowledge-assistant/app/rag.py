from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import requests
from pypdf import PdfReader
from .config import settings

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError("Only PDF, TXT and MD files are supported.")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        raise ValueError("No extractable text was found in this document.")
    return text

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

def stable_id(source: str, index: int, chunk: str) -> str:
    return hashlib.sha256(f"{source}:{index}:{chunk}".encode("utf-8")).hexdigest()

class RAGService:
    def __init__(self, settings_obj=settings):
        self.settings = settings_obj
        self._model = self._client = self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.settings.collection_name, metadata={"hnsw:space": "cosine"})
        return self._collection

    def _get_embedding_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def ingest_file(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        chunks = chunk_text(extract_text(path), self.settings.chunk_size, self.settings.chunk_overlap)
        vectors = self._get_embedding_model().encode(chunks, normalize_embeddings=True).tolist()
        ids = [stable_id(path.name, i, chunk) for i, chunk in enumerate(chunks)]
        self._get_collection().upsert(ids=ids, documents=chunks, embeddings=vectors,
            metadatas=[{"source": path.name, "chunk": i} for i in range(len(chunks))])
        return {"source": path.name, "chunks_indexed": len(chunks)}

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("question cannot be empty.")
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []
        vector = self._get_embedding_model().encode([question], normalize_embeddings=True).tolist()[0]
        result = collection.query(query_embeddings=[vector], n_results=min(top_k or self.settings.top_k, count),
                                  include=["documents", "metadatas", "distances"])
        return [{"text": d, "source": m.get("source", "unknown"), "chunk": m.get("chunk", -1),
                 "distance": float(dist)} for d, m, dist in zip(
                     result["documents"][0], result["metadatas"][0], result["distances"][0])]

    def generate_answer(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "I could not find relevant information in the indexed documents."
        context = "\n\n".join(f"[Source: {x['source']}, chunk {x['chunk']}]\n{x['text']}" for x in contexts)
        prompt = ("Answer using only the context. If it does not contain the answer, say so. "
                  "Do not invent facts. Mention source filenames.\n\n"
                  f"CONTEXT:\n{context}\n\nQUESTION:\n{question}")
        response = requests.post(f"{self.settings.ollama_url.rstrip('/')}/api/chat",
            json={"model": self.settings.ollama_model, "stream": False,
                  "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("The local LLM returned an empty response.")
        return answer

    def ask(self, question: str) -> dict[str, Any]:
        contexts = self.retrieve(question)
        return {"question": question, "answer": self.generate_answer(question, contexts),
                "sources": [{"source": x["source"], "chunk": x["chunk"], "distance": x["distance"]}
                            for x in contexts]}
