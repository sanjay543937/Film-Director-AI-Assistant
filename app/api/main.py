from fastapi import FastAPI, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, IngestResponse
from app.agents.graph import ask_film_assistant
from app.rag.ingest import build_vector_database

app = FastAPI(
    title="Film Director AI Assistant API",
    description="A filmmaking assistant powered by RAG, web search, and LangGraph.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = ask_film_assistant(request.question)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest():
    try:
        build_vector_database()
        return {
            "message": "PDF ingestion completed successfully.",
            "status": "success",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))