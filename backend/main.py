import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

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
# Models are loaded on first request and cached — avoids OOM on startup
# which would crash the process before it binds to a port.
_embedder     = None
_reranker     = None
_index        = None
_llm          = None

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(os.getenv("PINECONE_INDEX"))
    return _index

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )
    return _llm

# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    ticker:   Optional[str] = None      # e.g. "AAPL" — filters Pinecone search
    form_type: Optional[str] = None     # "10-K" or "10-Q"
    fiscal_year: Optional[int] = None
    top_k:    int = 10                  # candidates before reranking
    top_n:    int = 4                   # chunks kept after reranking


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


# ── Core RAG pipeline ─────────────────────────────────────────────────────────

def retrieve(request: QueryRequest) -> list[dict]:
    """
    Embed the query and search Pinecone with optional metadata filters.
    Returns raw Pinecone matches.
    """
    query_vector = get_embedder.encode(request.question).tolist()

    # Build metadata filter — only apply fields the user specified
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
    """
    Cross-encoder reranking: scores every (question, chunk) pair
    and returns the top_n highest scoring chunks.
    
    Why this matters: vector similarity finds semantically related chunks
    but can miss the most directly relevant ones. The cross-encoder
    reads question + chunk together, giving much more accurate relevance scores.
    """
    if not matches:
        return []

    pairs = [[question, m.metadata["text"]] for m in matches]
    scores = get_reranker().predict(pairs)

    # Attach scores and sort descending
    scored = sorted(
        zip(scores, matches),
        key=lambda x: x[0],
        reverse=True,
    )

    return scored[:top_n]


def build_context(scored_matches: list) -> tuple[str, list[SourceChunk]]:
    """
    Format reranked chunks into a context block for the LLM
    and build the source citations list.
    """
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
    """
    Call Groq with the retrieved context and return a cited answer.
    """
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


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG endpoint.
    1. Embed query → Pinecone search
    2. Cross-encoder rerank
    3. Groq LLM answer with citations
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query: {request.question[:80]}")

    # Step 1: retrieve
    matches = retrieve(request)
    if not matches:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Step 2: rerank
    scored = rerank(request.question, matches, request.top_n)

    # Step 3: build context + generate
    context, sources = build_context(scored)
    answer = generate_answer(request.question, context)

    logger.info(f"Answer generated from {len(sources)} sources")
    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok", "index": os.getenv("PINECONE_INDEX")}


@app.get("/tickers")
def list_tickers():
    """Return the list of tickers available in the index."""
    return {"tickers": ["AAPL", "MSFT", "NVDA"]}