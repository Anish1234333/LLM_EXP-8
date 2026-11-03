"""FastAPI service exposing the bank RAG assistant."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from support_rag import BackendError, RAGService, get_rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("support_api")

MAX_QUESTION_CHARS = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and FAISS index once at startup
    try:
        get_rag()
        app.state.rag_ready = True
        logger.info("RAG pipeline initialised")
    except Exception:
        app.state.rag_ready = False
        logger.exception("RAG pipeline failed to initialise")
    yield


app = FastAPI(
    title="Bank Customer Support RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS,
                          examples=["How do I block my debit card?"])

    @field_validator("question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class AskResponse(BaseModel):
    answer: str
    status: str            # "answered" or "insufficient_information"
    sources: list[str]
    latency_ms: float


def get_service() -> RAGService:
    try:
        return get_rag()
    except Exception:
        logger.exception("RAG service unavailable")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable.")


@app.get("/health")
def health(request: Request):
    """Reports that the API process is running and whether the RAG pipeline loaded."""
    return {"status": "ok", "rag_ready": getattr(request.app.state, "rag_ready", False)}


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"description": "Invalid question (empty or longer than 1000 characters)"},
        503: {"description": "Answer service temporarily unavailable"},
    },
)
def ask(payload: AskRequest, service: RAGService = Depends(get_service)):
    started = time.perf_counter()
    try:
        result = service.ask(payload.question)
    except BackendError:
        logger.error("ask failed: backend error")
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable. Please try again later.",
        )
    except Exception:
        logger.exception("ask failed: unexpected error")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    latency = round((time.perf_counter() - started) * 1000, 1)
    # Log operational details only, never the question text or secrets
    logger.info("ask status=%s sources=%d latency_ms=%.1f question_chars=%d",
                result["status"], len(result["sources"]), latency, len(payload.question))
    return AskResponse(**result, latency_ms=latency)
