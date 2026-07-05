import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SEC Earnings RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy model loading
# All heavy imports (torch, sentence-transformers, pinecone, langchain) are
# inside these loader functions. This means uvicorn can start, bind to the
# port, and pass Render's health check before any model is loaded.
# Each model loads once on first request, then stays cached.

_embedder = None
_reranker = None
_index    = None
_llm      = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_index():
    global _index
    if _index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(os.getenv("PINECONE_INDEX"))
    return _index


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
        )
    return _llm


# Request / response models

class QueryRequest(BaseModel):
    question:    str
    ticker:      Optional[str] = None
    form_type:   Optional[str] = None
    fiscal_year: Optional[int] = None
    top_k:       int = 10
    top_n:       int = 4


class SourceChunk(BaseModel):
    chunk_id:    str
    ticker:      str
    form_type:   str
    fiscal_year: int
    section:     str
    text:        str
    score:       float


class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceChunk]


# Core RAG pipeline

def retrieve(request: QueryRequest) -> list:
    query_vector = get_embedder().encode(request.question).tolist()

    filters = {}
    if request.ticker:
        filters["ticker"] = request.ticker.upper()
    if request.form_type:
        filters["form_type"] = request.form_type
    if request.fiscal_year:
        filters["fiscal_year"] = request.fiscal_year

    results = get_index().query(
        vector=query_vector,
        top_k=request.top_k,
        include_metadata=True,
        filter=filters if filters else None,
    )

    return results.matches


def rerank(question: str, matches: list, top_n: int) -> list:
    if not matches:
        return []

    pairs = [[question, m.metadata["text"]] for m in matches]
    scores = get_reranker().predict(pairs)

    scored = sorted(
        zip(scores, matches),
        key=lambda x: x[0],
        reverse=True,
    )

    return scored[:top_n]


def build_context(scored_matches: list) -> tuple[str, list[SourceChunk]]:
    context_parts = []
    sources = []

    for rank, (score, match) in enumerate(scored_matches, 1):
        meta = match.metadata
        text = meta.get("text", "")

        context_parts.append(
            f"[Source {rank}: {meta.get('ticker')} {meta.get('form_type')} "
            f"FY{meta.get('fiscal_year')} — {meta.get('section')}]\n{text}"
        )

        sources.append(SourceChunk(
            chunk_id    = match.id,
            ticker      = meta.get("ticker", ""),
            form_type   = meta.get("form_type", ""),
            fiscal_year = meta.get("fiscal_year", 0),
            section     = meta.get("section", ""),
            text        = text,
            score       = float(score),
        ))

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(question: str, context: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = """You are a financial analyst assistant specializing in SEC filings.
Answer questions using ONLY the provided context from SEC filings.
Always cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so clearly.
Be precise with numbers, dates, and financial figures."""

    user_prompt = f"""Context from SEC filings:
{context}

Question: {question}

Answer with citations:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = get_llm().invoke(messages)
    return response.content


# API endpoints

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query: {request.question[:80]}")

    matches = retrieve(request)
    if not matches:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    scored = rerank(request.question, matches, request.top_n)
    context, sources = build_context(scored)
    answer = generate_answer(request.question, context)

    logger.info(f"Answer generated from {len(sources)} sources")
    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok", "index": os.getenv("PINECONE_INDEX")}


@app.get("/tickers")
def list_tickers():
    return {"tickers": ["AAPL", "MSFT", "NVDA"]}