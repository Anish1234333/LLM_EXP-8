"""RAG pipeline for the bank customer guide.

Retrieval (LangChain + FAISS, local embeddings) is kept separate from the LLM
backend (Groq's OpenAI-compatible API), so either can be changed on its own.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()  # reads a local .env file if present; does nothing on Streamlit Cloud
logger = logging.getLogger("support_rag")

BASE_DIR = Path(__file__).resolve().parent
GUIDE_PATH = BASE_DIR / "bank_guide.txt"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TOP_K = 3
# Optional retrieval guard: if the best match distance is above this value, refuse.
# 0 disables the guard. Calibrate with calibrate_threshold.py before enabling.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0") or 0)

FALLBACK_ANSWER = (
    "I couldn't find a reliable answer in the bank customer guide. "
    "Please contact the customer care line or visit a branch. "
    "Never share your PIN, full card number, or one-time password with anyone."
)
INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are a polite retail bank customer-support assistant.
Use only the BANK GUIDE CONTEXT to answer the CUSTOMER QUESTION.
Do not add steps, numbers, fees, timelines or policies that are not written in the context.
Do not invent interest rates, loan decisions, account balances, or card details.
If the context does not contain the answer, output exactly: {INSUFFICIENT_MARKER}
Never ask for or repeat a PIN, full card number, or one-time password.
Answer in at most four sentences."""


class BackendError(Exception):
    """Raised when the LLM provider cannot produce an answer."""


class RAGService:
    def __init__(self):
        if not GUIDE_PATH.exists():
            raise FileNotFoundError("bank_guide.txt not found next to support_rag.py")
        text = GUIDE_PATH.read_text(encoding="utf-8")
        doc = Document(page_content=text, metadata={"source": GUIDE_PATH.name})
        splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
        self.chunks = splitter.split_documents([doc])
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.store = FAISS.from_documents(self.chunks, self.embeddings)
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=GROQ_BASE_URL,
            timeout=60,
        )
        logger.info("RAG ready: %d chunks, embed=%s, llm=%s",
                    len(self.chunks), EMBED_MODEL, LLM_MODEL)

    def retrieve(self, question: str):
        """Return (document, distance) pairs. For FAISS, a lower distance is closer."""
        return self.store.similarity_search_with_score(question, k=TOP_K)

    def generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"BANK GUIDE CONTEXT:\n{context}\n\nCUSTOMER QUESTION:\n{question}"},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL, messages=messages, max_tokens=1024, temperature=0.1
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            # Log only the exception type so no key or URL reaches the logs
            logger.error("LLM call failed: %s", type(exc).__name__)
            raise BackendError("LLM call failed") from exc

    def ask(self, question: str) -> dict:
        hits = self.retrieve(question)
        if not hits or (MAX_DISTANCE > 0 and hits[0][1] > MAX_DISTANCE):
            return {"answer": FALLBACK_ANSWER, "status": "insufficient_information", "sources": []}

        context = "\n\n".join(doc.page_content for doc, _ in hits)
        sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in hits})
        raw = self.generate(question, context)
        if not raw or INSUFFICIENT_MARKER in raw:
            return {"answer": FALLBACK_ANSWER, "status": "insufficient_information", "sources": []}
        return {"answer": raw, "status": "answered", "sources": sources}


@lru_cache(maxsize=1)
#Build the index and embedding model once per process, not once per request.
def get_rag() -> RAGService:
    return RAGService()
