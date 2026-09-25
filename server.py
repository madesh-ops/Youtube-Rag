import io
import contextlib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.Ingestion import YTSloader, Chunking
from rag.Embedding import EmbeddingManager
from rag.vectorstore import VectorStore
from rag.retrieval import RAGRetriever
from rag.Generation import RAGGenerator

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="YouTube RAG")

### Heavy objects are created once at startup and shared by every request

embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
retriever = RAGRetriever(vector_store=vectorstore, embedding_manager=embedding_manager)
generator = RAGGenerator(retriever)


class IngestRequest(BaseModel):
    video_id: str

class AskRequest(BaseModel):
    question: str


def run_captured(fn, *args):
    """Run fn and return its result plus everything it printed, for the page's console.

    Any failure becomes an HTTPException so the page shows the real reason
    instead of a bare "Internal Server Error".
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return result, buf.getvalue().splitlines()


def _ingest(video_id: str) -> int:
    docs = YTSloader(video_id, "en")
    if not docs:
        raise HTTPException(404, f"No English transcript found for video '{video_id}'.")
    chunks = Chunking(docs)
    embeddings = embedding_manager.generate_embeddings([c.page_content for c in chunks])

    ## Keep exactly one video in the store
    vectorstore.clear()
    vectorstore.add_documents(chunks, embeddings)
    return len(chunks)


@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks, logs = run_captured(_ingest, req.video_id.strip())
    return {"chunks": chunks, "logs": logs}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    result, logs = run_captured(generator.answer, question)
    return {**result, "logs": logs}


## Must be last: a mount at "/" would otherwise swallow the API routes
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
