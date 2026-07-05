# frontend/app.py

import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "https://sec-rag-api.onrender.com")

# Page config

st.set_page_config(
    page_title="SEC Earnings Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  /* Base */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; max-width: 1100px; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2a42;
  }
  [data-testid="stSidebar"] * { color: #a0aec0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stTextInput label { color: #718096 !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }

  /* Header */
  .app-header {
    border-bottom: 1px solid #1e2a42;
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
  }
  .app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f6ad55;
    letter-spacing: 0.04em;
    margin: 0;
  }
  .app-subtitle {
    font-size: 0.8rem;
    color: #4a5568;
    margin-top: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Query input */
  .stTextArea textarea {
    background-color: #0d1220 !important;
    border: 1px solid #2d3748 !important;
    color: #e8eaf0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: 4px !important;
  }
  .stTextArea textarea:focus {
    border-color: #f6ad55 !important;
    box-shadow: 0 0 0 1px #f6ad55 !important;
  }

  /* Button */
  .stButton button {
    background-color: #f6ad55;
    color: #0a0e1a;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 3px;
    padding: 0.5rem 1.5rem;
    width: 100%;
    transition: background-color 0.15s;
  }
  .stButton button:hover { background-color: #ed8936; }

  /* Answer panel — the signature element */
  .answer-panel {
    background-color: #0d1220;
    border: 1px solid #2d3748;
    border-left: 3px solid #f6ad55;
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #e2e8f0;
  }
  .answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #f6ad55;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }

  /* Source cards — terminal scanline feel */
  .source-card {
    background-color: #0d1220;
    border: 1px solid #1e2a42;
    border-radius: 3px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    position: relative;
    overflow: hidden;
  }
  .source-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #f6ad55, transparent);
    opacity: 0.4;
  }
  .source-meta {
    color: #f6ad55;
    font-weight: 600;
    margin-bottom: 0.4rem;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
  }
  .source-score {
    display: inline-block;
    background-color: #1a2035;
    color: #68d391;
    font-size: 0.65rem;
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    margin-left: 0.5rem;
    letter-spacing: 0.04em;
  }
  .source-text {
    color: #718096;
    line-height: 1.6;
    font-size: 0.75rem;
    border-top: 1px solid #1e2a42;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
  }

  /* Metrics row */
  .metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .metric-box {
    background-color: #0d1220;
    border: 1px solid #1e2a42;
    border-radius: 3px;
    padding: 0.6rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #4a5568;
    flex: 1;
  }
  .metric-box span {
    display: block;
    color: #a0aec0;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 0.2rem;
  }

  /* Section label */
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2a42;
  }

  /* Error */
  .error-box {
    background-color: #1a0a0a;
    border: 1px solid #742a2a;
    border-radius: 3px;
    padding: 1rem 1.25rem;
    color: #fc8181;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
  }

  /* Example queries */
  .example-btn {
    background: none;
    border: 1px solid #1e2a42;
    color: #4a5568;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.3rem 0.6rem;
    border-radius: 2px;
    cursor: pointer;
    margin: 0.2rem;
    transition: all 0.15s;
  }
</style>
""", unsafe_allow_html=True)


# Header

st.markdown("""
<div class="app-header">
  <p class="app-title">▸ SEC EARNINGS INTELLIGENCE</p>
  <p class="app-subtitle">RAG system · AAPL · MSFT · NVDA · 10-K & 10-Q filings</p>
</div>
""", unsafe_allow_html=True)


# Sidebar

with st.sidebar:
    st.markdown("### Filters")
    st.markdown("*Narrow the search scope. Leave blank to search all filings.*")

    ticker = st.selectbox(
        "Company",
        options=["All", "AAPL", "MSFT", "NVDA"],
        index=0,
    )

    form_type = st.selectbox(
        "Filing type",
        options=["All", "10-K", "10-Q"],
        index=0,
    )

    fiscal_year = st.selectbox(
        "Fiscal year",
        options=["All", 2026, 2025, 2024, 2023],
        index=0,
    )

    top_k = st.slider("Candidates retrieved", min_value=5, max_value=20, value=10)
    top_n = st.slider("Kept after reranking", min_value=2, max_value=8, value=4)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Built with **FastAPI**, **Pinecone**, **sentence-transformers**, 
    and **Groq** (Llama 3.1). Evaluated with **RAGAS**.
    
    [GitHub →](#)
    """)


# Example queries

EXAMPLES = [
    "What was Apple's total revenue in fiscal year 2024?",
    "What AI risks did Microsoft identify in their annual report?",
    "Which segment drove NVIDIA's revenue growth?",
    "How did Apple's gross margin change year over year?",
    "What did Microsoft say about cloud growth in their latest 10-Q?",
]

# Query input

if "question" not in st.session_state:
    st.session_state.question = ""

st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)

cols = st.columns(len(EXAMPLES))
for i, (col, example) in enumerate(zip(cols, EXAMPLES)):
    with col:
        if st.button(f"eg.{i+1}", key=f"ex_{i}", help=example):
            st.session_state.question = example

question = st.text_area(
    label="Ask anything about the filings",
    value=st.session_state.question,
    height=90,
    placeholder="e.g. What were Apple's main revenue drivers in FY2024?",
    label_visibility="collapsed",
)

run = st.button("RUN QUERY ▸")


# Query execution

if run and question.strip():
    payload = {
        "question": question,
        "top_k": top_k,
        "top_n": top_n,
    }
    if ticker != "All":
        payload["ticker"] = ticker
    if form_type != "All":
        payload["form_type"] = form_type
    if fiscal_year != "All":
        payload["fiscal_year"] = int(fiscal_year)

    with st.spinner("Retrieving · Reranking · Generating..."):
        try:
            # Ping health endpoint first — wakes the backend if it was sleeping
            try:
                requests.get(f"{API_URL}/health", timeout=90)
            except Exception:
                pass
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.ConnectionError:
            st.markdown("""
            <div class="error-box">
            ✗ Cannot reach the backend API at localhost:8000.<br>
            Make sure <code>uvicorn backend.main:app --reload --port 8000</code> is running.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">✗ Query failed: {e}</div>
            """, unsafe_allow_html=True)
            st.stop()

    # Metrics row
    sources = data.get("sources", [])
    avg_score = sum(s["score"] for s in sources) / len(sources) if sources else 0

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">Sources used<span>{len(sources)}</span></div>
      <div class="metric-box">Avg rerank score<span>{avg_score:.3f}</span></div>
      <div class="metric-box">Ticker filter<span>{ticker}</span></div>
      <div class="metric-box">Filing type<span>{form_type}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Answer panel
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="answer-panel">
      <div class="answer-label">▸ generated answer</div>
      {data['answer'].replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # Source chunks
    st.markdown('<div class="section-label">Retrieved sources</div>', unsafe_allow_html=True)

    for i, source in enumerate(sources, 1):
        score_color = "#68d391" if source["score"] > 5 else "#f6ad55" if source["score"] > 0 else "#fc8181"
        st.markdown(f"""
        <div class="source-card">
          <div class="source-meta">
            [{i}] {source['ticker']} · {source['form_type']} · FY{source['fiscal_year']} · {source['section'].upper()}
            <span class="source-score" style="color:{score_color}">score: {source['score']:.3f}</span>
          </div>
          <div class="source-text">{source['text'][:400]}{'...' if len(source['text']) > 400 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

elif run and not question.strip():
    st.markdown("""
    <div class="error-box">✗ Enter a question before running a query.</div>
    """, unsafe_allow_html=True)