from pathlib import Path
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .config import settings
from .rag import RAGService
app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0",
              description="Ask questions about your own PDF, TXT and Markdown documents.")
service = RAGService()
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
@app.get("/health")
def health():
    return {"status": "ok", "service": "RAG Knowledge Assistant"}
@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Upload a PDF, TXT or MD file.")
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    target = settings.documents_dir / f"{uuid.uuid4().hex}_{Path(file.filename or 'document').name}"
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Maximum upload size is 15 MB.")
        target.write_bytes(content)
        return {"message": "Document indexed successfully.", **service.ingest_file(target)}
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"Could not index document: {exc}") from exc
@app.post("/chat")
def chat(payload: AskRequest):
    try:
        return service.ask(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG request failed: {exc}") from exc
