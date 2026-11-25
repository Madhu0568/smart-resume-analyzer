from dataclasses import dataclass
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
@dataclass(frozen=True)
class Settings:
    documents_dir: Path = BASE_DIR / "data" / "documents"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    collection_name: str = "knowledge_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
settings = Settings()
