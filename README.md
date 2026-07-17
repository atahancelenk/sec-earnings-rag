# 📊 SEC Earnings Intelligence

A retrieval-augmented generation (RAG) system for querying SEC 10-K and 10-Q filings from Apple, Microsoft, and NVIDIA using natural language. Answers are grounded in and cited from actual filing text — no hallucinated numbers.

> **⚠️ Live demo note:** The hosted version is **not reliably available online**. Both services run on Render's free tier (512 MB RAM), and the embedding + reranking models (`sentence-transformers`, PyTorch) regularly exceed that limit and get OOM-killed. See [Known Limitations](#known-limitations) for details and workarounds. A local screen-capture walkthrough is included instead of relying on the live URL.

---

## What it does

Ask a question like *"What was Apple's total net sales in fiscal year 2024?"* and the system:

1. Embeds the query locally with `sentence-transformers`
2. Retrieves candidate chunks from a Pinecone vector index, optionally filtered by ticker / form type / fiscal year
3. Reranks candidates with a cross-encoder for relevance
4. Passes the top chunks to an LLM (Groq / Llama 3.1 8B Instant) to generate a cited answer
5. Returns the answer alongside the exact source chunks it was grounded in

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  SEC EDGAR API   │────▶│  Ingestion        │────▶│  Pinecone        │
│  (10-K / 10-Q)   │     │  Pipeline         │     │  Serverless      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit       │◀───▶│  FastAPI          │◀───▶│  Retrieve →      │
│  Frontend         │     │  Backend          │     │  Rerank → Groq   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

Built in four phases:

1. **Ingestion pipeline** — download, parse, chunk, and embed filings into Pinecone
2. **FastAPI query backend** — retrieval, reranking, and answer generation
3. **RAGAS evaluation** — a golden dataset used to score retrieval and generation quality
4. **Streamlit frontend** — a Bloomberg-terminal-inspired UI for querying the system

---

## Ingestion pipeline

| Stage | File | What it does |
|---|---|---|
| Download | `ingestion/edgar_downloader.py` | Resolves ticker → CIK via EDGAR's `company_tickers.json`, pulls recent 10-K/10-Q filings from the submissions API, downloads the primary document HTML |
| Parse | `ingestion/pdf_parser.py` | Strips HTML noise, splits the filing into sections (MD&A, Risk Factors, Financial Statements, etc.) by heading detection, extracts tables separately from prose |
| Chunk | `ingestion/chunker.py` | Two separate chunking paths (see below) |
| Embed | `ingestion/embedder.py` | Embeds chunks locally with `all-MiniLM-L6-v2` and upserts to Pinecone in batches |

### Two-path chunking

Prose and tables are chunked differently because they have fundamentally different structure:

- **Prose** (`_chunk_prose`): a sliding window of 512 tokens with 50-token overlap, tokenized with `tiktoken`. Each chunk is prefixed with a context header (`[TICKER | FORM_TYPE | FY | Section | Chunk N]`) so it's self-contained and interpretable without neighboring chunks.
- **Tables** (`_chunk_tables`): each table becomes one chunk to preserve row/column relationships. Tables over 1,200 tokens are split by row groups, with the header row re-attached to every group so no split loses column context.

This is a deliberate, explainable architectural decision: financial tables lose meaning if split mid-row by a generic sliding window, while prose benefits from overlap for context continuity.

### Ingestion results

- **~3,757 vectors** ingested across **18 filings** (10-K and 10-Q) for AAPL, MSFT, and NVDA
- Embedding dimension: 384 (`all-MiniLM-L6-v2`)
- Vector store: Pinecone Serverless

---

## Backend (`backend/main.py`)

FastAPI service exposing:

- `POST /query` — runs the full retrieve → rerank → generate pipeline
- `GET /health` — health check endpoint
- `GET /tickers` — lists indexed tickers

**Retrieval → rerank → generate:**

1. `retrieve()` — embeds the question, queries Pinecone (`top_k`, default 10) with optional metadata filters (ticker, form type, fiscal year)
2. `rerank()` — scores retrieved candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keeps the top `top_n` (default 4)
3. `build_context()` — assembles the reranked chunks into a labeled context block and a list of `SourceChunk`s returned to the client
4. `generate_answer()` — sends the context + question to Groq (Llama 3.1 8B Instant) via LangChain, instructed to answer only from context and cite `[Source N]`

**Lazy model loading:** heavy imports (`torch`, `sentence-transformers`, `pinecone`, `langchain`) are deferred inside loader functions rather than imported at module load. This lets `uvicorn` bind to the port and pass Render's health check immediately, with the embedder/reranker loading on first request. This is a *mitigation*, not a full fix — see [Known Limitations](#known-limitations).

---

## Frontend (`frontend/app.py`)

A Streamlit app styled after a Bloomberg terminal (dark theme, monospace type, amber accents) with:

- Sidebar filters for ticker, form type, fiscal year, and retrieval/reranking depth (`top_k`, `top_n`)
- One-click example queries
- An answer panel with inline citations
- Source cards showing each retrieved chunk, its rerank score, and provenance metadata

---

## Evaluation (RAGAS)

`evaluation/golden_dataset.py` defines a hand-written set of question → ground-truth pairs covering simple lookups, cross-document reasoning, year-over-year comparisons, and deliberate "should fail" cases (e.g., asking about a company not in the index) to test for hallucination.

`evaluation/ragas_eval.py` runs every golden question through the live API, then scores retrieval/generation quality with RAGAS 0.2.x using Groq as the judge LLM and local `HuggingFaceEmbeddings` for embedding-based metrics.

### Results

| Metric | Score |
|---|---|
| Context recall | **1.0** |
| Answer relevancy | **~0.63** |
| Context precision | **0.5** |
| Faithfulness | **NaN** — judge LLM calls hit Groq free-tier rate limits during scoring |

The faithfulness NaN is reported honestly rather than hidden. RAGAS's faithfulness metric requires the most judge-LLM calls per sample (claim decomposition + verification), which made it the first metric to break under the rate limit.

---

## Known Limitations

- **Live deployment does not reliably work.** Both the FastAPI backend and Streamlit frontend are deployed on Render's **free tier (512 MB RAM)**. `sentence-transformers` and its PyTorch dependency alone can approach or exceed that limit once loaded, especially with the cross-encoder reranker also in memory — the process gets OOM-killed under real traffic. Deferred/lazy imports (loading models only on first request, after the health check passes) delay the problem but don't solve it; the first real query can still crash the dyno. **Because of this, the README relies on a local screen-recording walkthrough instead of the hosted URL.**
- **RAGAS faithfulness returned NaN** due to Groq free-tier rate limiting during the judge LLM's multi-call scoring process.
- **No OpenAI API key used anywhere** — the project deliberately runs entirely on free/open alternatives (local `sentence-transformers` embeddings, Groq for generation and judging), which is part of why RAM and rate limits are the binding constraints rather than cost.
- Fiscal year/quarter inference from filing dates (`edgar_downloader.py`) is an approximation — SEC filing dates lag the actual period end by ~45 days, so edge cases near fiscal year boundaries can be misclassified.

### What would fix the RAM issue

- Move embedding/reranking off the web dyno entirely (e.g., a separate worker, or an embeddings API instead of loading PyTorch models in-process)
- Upgrade to a paid Render tier with more memory
- Swap `sentence-transformers` for a lighter/quantized model, or move embedding inference to Pinecone's own hosted embedding feature

---

## Tech stack

| Layer | Tool |
|---|---|
| Backend / API | FastAPI, uvicorn |
| Frontend | Streamlit |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | Pinecone Serverless |
| LLM | Groq (Llama 3.1 8B Instant) |
| Orchestration | LangChain, `langchain-groq` |
| Evaluation | RAGAS 0.2.x |
| Parsing | BeautifulSoup, lxml, tiktoken |
| Deployment | Render (free tier) |

---

## Project structure

```
.
├── backend/
│   └── main.py              # FastAPI query API — retrieve, rerank, generate
├── frontend/
│   └── app.py                # Streamlit UI
├── ingestion/
│   ├── edgar_downloader.py   # SEC EDGAR download client
│   ├── pdf_parser.py         # HTML filing → structured sections + tables
│   ├── chunker.py            # Two-path (prose/table) chunking
│   ├── embedder.py           # Local embedding + Pinecone upsert
│   └── run_ingestion.py      # End-to-end ingestion entrypoint
├── evaluation/
│   ├── golden_dataset.py     # Hand-written Q&A ground truth
│   └── ragas_eval.py         # RAGAS scoring pipeline
├── render.yaml                # Render deployment config (backend + frontend)
├── requirements.txt
└── .python-version            # Pinned to 3.11.9 (3.14 breaks on Render)
```

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (.env)
PINECONE_API_KEY=...
PINECONE_INDEX=...
GROQ_API_KEY=...

# 3. Ingest filings (one-time; populates Pinecone)
cd ingestion
python run_ingestion.py

# 4. Start the backend
uvicorn backend.main:app --reload --port 8000

# 5. Start the frontend (in a separate terminal)
API_URL=http://localhost:8000 streamlit run frontend/app.py
```

> Running locally avoids the RAM ceiling entirely — the free-tier memory constraint is specific to the hosted Render deployment, not the system itself.

---

## Planned improvements

- **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer before embedding the query, to close the semantic gap between short questions and dense filing prose
- **Hybrid retrieval (BM25 + dense)** — combine keyword and semantic search, particularly to improve exact-figure lookups where dense retrieval alone underperforms
- Move model inference off the web process to resolve the memory ceiling for a real hosted demo