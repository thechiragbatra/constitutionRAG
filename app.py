"""Stage 6: Streamlit frontend — Indian Emerald theme.

A Constitution-of-India-coloured (saffron / white / emerald green / navy)
dynamic frontend that talks to the FastAPI backend over HTTP.

Endpoints used:
  POST   /ask          - question + config overrides -> answer + citations
  GET    /article/{n}  - exact Article text
  GET    /config       - current configuration
  GET    /health       - service status

Run:  streamlit run app.py
Backend:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import requests
import streamlit as st


def _backend_url() -> str:
    try:
        url = st.secrets.get("BACKEND_URL", "http://localhost:8000")
        if not url:
            url = "http://localhost:8000"
        return url
    except Exception:
        return "http://localhost:8000"


BACKEND_URL = _backend_url()

EXAMPLES = [
    "Can the police arrest me without telling me why?",
    "What does Article 21 say?",
    "What does Article 14 say about equality?",
    "What are the Fundamental Rights?",
    "Who is the current Prime Minister of India?",
    "Can the State impose reasonable restrictions on freedom of speech?",
]

# ---------------------------------------------------------------------------
# Indian Emerald theme CSS
# ---------------------------------------------------------------------------
INDIAN_CSS = """
<style>
/* CSS variables — Indian flag palette + emerald */
:root {
  --saffron: #FF9933;
  --saffron-dark: #E68A2B;
  --white: #FFFFFF;
  --emerald: #138808;
  --emerald-light: #2E7D32;
  --navy: #000080;
  --ink: #1A1A1A;
  --muted: #666666;
  --border: #E0E0E0;
}

/* App background */
.stApp {
  background: var(--white);
}

/* Tricolour header band */
.header-band {
  width: 100%;
  height: 10px;
  display: flex;
}
.header-band .saffron { flex: 1; background: var(--saffron); }
.header-band .white   { flex: 1; background: var(--white); border-bottom: 1px solid var(--border); }
.header-band .emerald { flex: 1; background: var(--emerald); }

/* Page title */
.app-title {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--emerald);
  letter-spacing: 0.2px;
  margin-top: 1.2rem;
  margin-bottom: 0.25rem;
}
.app-subtitle {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F7F7F7 0%, #FFFFFF 100%);
}
section[data-testid="stSidebar"] .sidebar-content {
  color: var(--ink);
}
.sidebar-header {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--emerald);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Primary button — emerald */
div.stButton > button[kind="primary"] {
  background: var(--emerald) !important;
  color: white !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1rem !important;
  transition: background 0.2s ease, transform 0.1s ease !important;
}
div.stButton > button[kind="primary"]:hover {
  background: var(--emerald-light) !important;
  transform: translateY(-1px);
}

/* Secondary buttons — saffron outline */
div.stButton > button:not([kind="primary"]) {
  background: white !important;
  color: var(--saffron-dark) !important;
  border: 1.5px solid var(--saffron) !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  padding: 0.45rem 0.9rem !important;
  transition: all 0.2s ease !important;
}
div.stButton > button:not([kind="primary"]):hover {
  background: var(--saffron) !important;
  color: white !important;
  border-color: var(--saffron-dark) !important;
}

/* Answer card */
.answer-card {
  background: linear-gradient(135deg, #FFFFFF 0%, #F1F8F2 100%);
  border: 1px solid var(--border);
  border-left: 5px solid var(--emerald);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  margin: 1rem 0;
  color: var(--ink);
  line-height: 1.6;
}
.answer-card strong {
  color: var(--emerald);
}

/* Abstention banner — saffron */
.abstain-banner {
  background: linear-gradient(90deg, var(--saffron) 0%, #FFB066 100%);
  color: #4A3700;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: 1rem 0;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.abstain-banner .icon { font-size: 1.4rem; }

/* Found banner — emerald */
.found-banner {
  background: linear-gradient(90deg, var(--emerald) 0%, #2E7D32 100%);
  color: white;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: 1rem 0;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.found-banner .icon { font-size: 1.4rem; }

/* Citation chips */
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 1rem 0;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: #F1F8F2;
  color: var(--emerald);
  border: 1.5px solid var(--emerald);
  border-radius: 20px;
  padding: 0.35rem 0.85rem;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}
.chip:hover {
  background: var(--emerald);
  color: white;
  border-color: var(--emerald-light);
}
.chip-label {
  pointer-events: none;
}

/* Retrieved chunk cards */
.chunk-card {
  background: white;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin: 0.6rem 0;
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.chunk-card:hover {
  box-shadow: 0 4px 12px rgba(19,136,8,0.08);
  border-color: var(--emerald-light);
}
.chunk-rank {
  display: inline-block;
  background: var(--emerald);
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
  border-radius: 4px;
  padding: 0.12rem 0.5rem;
  margin-right: 0.5rem;
}
.chunk-score {
  font-size: 0.78rem;
  color: var(--muted);
  font-family: monospace;
}
.chunk-arts {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--navy);
  margin-right: 0.5rem;
}
.chunk-text {
  margin-top: 0.5rem;
  color: var(--ink);
  font-size: 0.9rem;
  line-height: 1.5;
}

/* Exact article shortcut pill */
.shortcut-pill {
  background: #FFF4E0;
  border: 1px solid var(--saffron);
  border-radius: 6px;
  padding: 0.6rem 0.9rem;
  margin: 0.8rem 0;
  font-size: 0.9rem;
  color: #4A3700;
}

/* Flagged citation warning */
.flagged-note {
  background: #FFF7ED;
  border: 1px dashed var(--saffron);
  border-radius: 6px;
  padding: 0.6rem 0.9rem;
  margin: 0.8rem 0;
  font-size: 0.85rem;
  color: #4A3700;
}

/* Footer meta */
.footer-meta {
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 1.2rem;
  border-top: 1px solid var(--border);
  padding-top: 0.6rem;
}

/* Example selector styling */
.stSelectbox > div > div {
  border-color: var(--emerald) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Constitution of India — RAG QA",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(INDIAN_CSS, unsafe_allow_html=True)

# Tricolour header band
st.markdown(
    '<div class="header-band">'
    '<div class="saffron"></div>'
    '<div class="white"></div>'
    '<div class="emerald"></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Constitution of India <span style="color:var(--saffron)">—</span> RAG Question Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Answers only from the indexed Constitution, with Article citations in [Article N] format.</div>', unsafe_allow_html=True)


def api_get(path: str, timeout: int = 10):
    try:
        return requests.get(f"{BACKEND_URL}{path}", timeout=timeout), None
    except Exception as exc:  # pragma: no cover - network
        return None, str(exc)


def api_post(path: str, payload: dict, timeout: int = 120):
    try:
        return requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout), None
    except Exception as exc:  # pragma: no cover - network
        return None, str(exc)


def fetch_json(path: str):
    resp, err = api_get(path)
    if resp is None:
        return None, err
    if resp.status_code >= 400:
        return None, resp.text
    return resp.json(), None


def ask_backend(question: str, cfg: dict):
    payload = {
        "question": question,
        "chunking": cfg["chunking"],
        "retriever": cfg["retriever"],
        "top_k": cfg["top_k"],
        "rerank": cfg["rerank"],
        "abstain_threshold": cfg["abstain_threshold"],
        "use_retrieval": cfg["use_retrieval"],
    }
    resp, err = api_post("/ask", payload)
    if resp is None:
        return None, err
    if resp.status_code >= 400:
        return None, resp.text
    return resp.json(), None


def fetch_article(number: str):
    return fetch_json(f"/article/{number}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
health, err = fetch_json("/health")
if health is None:
    st.error(
        f"Backend not reachable at `{BACKEND_URL}`. Start it with:\n"
        "`uvicorn backend.main:app --reload`\n\n{err}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — retrieval settings (Indian tricolour accents)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Retrieval settings</div>', unsafe_allow_html=True)
    chunking = st.selectbox(
        "Chunking", ["article", "fixed256", "fixed512"], index=0,
        help="article = one Article per chunk; fixed256/512 = fixed-size chunks",
    )
    retriever = st.selectbox(
        "Retriever", ["hybrid", "bm25", "dense"], index=0,
        help="hybrid = BM25 + dense via Reciprocal Rank Fusion",
    )
    top_k = st.slider("Top-k chunks", 1, 20, 5, help="How many passages to retrieve")
    rerank = st.checkbox(
        "Cross-encoder reranker (top-20 → top-5)",
        value=False,
        help="Rescores candidates with a cross-encoder for better precision",
    )
    abstain_threshold = st.slider(
        "Abstain threshold (cosine)", 0.0, 1.0, 0.25, 0.05,
        help="If best similarity is below this, say 'Not found in the Constitution'",
    )
    use_retrieval = st.checkbox(
        "Use retrieval (uncheck = no-RAG ablation)",
        value=True,
        help="Disable retrieval to compare RAG vs parametric knowledge",
    )
    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")
    st.caption(f"Articles indexed: {health.get('articles_loaded', '?') if health else '?'}")

cfg = {
    "chunking": chunking,
    "retriever": retriever,
    "top_k": top_k,
    "rerank": rerank,
    "abstain_threshold": abstain_threshold,
    "use_retrieval": use_retrieval,
}

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
st.subheader("Ask a question about the Constitution of India")
example = st.selectbox(
    "Example questions",
    ["— pick one —"] + EXAMPLES,
    help="Choose a sample question to try the system",
)
default_q = "" if example == "— pick one —" else example
question = st.text_input(
    "Your question",
    value=default_q,
    placeholder="e.g. What does Article 21 say?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Handle question
# ---------------------------------------------------------------------------
if ask_clicked and question.strip():
    with st.spinner("Retrieving passages and generating answer…"):
        data, err = ask_backend(question, cfg)

    if data is None:
        st.error(f"Request failed: {err}")
    else:
        # ---- abstention / found banner ----
        if data["abstained"]:
            st.markdown(
                '<div class="abstain-banner">'
                '<span class="icon">⚠️</span>'
                '<span><b>Not found in the Constitution</b></span>'
                '</div>',
                unsafe_allow_html=True,
            )
            if data.get("abstain_reason"):
                st.caption(data["abstain_reason"])
        else:
            st.markdown(
                '<div class="found-banner">'
                '<span class="icon">✅</span>'
                '<span><b>Answer found in the Constitution</b></span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ---- exact article shortcut ----
        if data.get("exact_article"):
            ea = data["exact_article"]
            st.markdown(
                f'<div class="shortcut-pill">'
                f'⚡ Exact-lookup shortcut: <b>Article {ea["article"]}</b> '
                f'(Part {ea["part"]}) — also went through retrieval.'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ---- answer card ----
        st.markdown(
            f'<div class="answer-card">{data["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # ---- cited articles as chips ----
        if data["cited_articles"]:
            st.write("**Cited Articles** (click to open full text)")
            chip_cols = st.columns(min(len(data["cited_articles"]), 5))
            for i, art in enumerate(data["cited_articles"]):
                with chip_cols[i % len(chip_cols)]:
                    if st.button(f"[Article {art}]", key=f"chip_{art}_{i}",
                                 use_container_width=True):
                        st.session_state["open_article"] = art

        # ---- flagged citations ----
        if data["flagged_citations"]:
            st.markdown(
                '<div class="flagged-note">'
                f'⚠️ Citation check flagged (not in retrieved passages): '
                + ", ".join(f"[Article {a}]" for a in data["flagged_citations"])
                + '</div>',
                unsafe_allow_html=True,
            )

        # ---- retrieved chunks expander ----
        with st.expander(
            f"Retrieved passages ({len(data['retrieved'])})", expanded=False
        ):
            for c in data["retrieved"]:
                arts = ", ".join(f"[Article {a}]" for a in c["articles"])
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<span class="chunk-rank">#{c["rank"]}</span>'
                    f'<span class="chunk-arts">{arts}</span>'
                    f'<span class="chunk-score">score {c["score"]:.4f}</span>'
                    f'<span style="color:var(--muted);font-size:0.78rem;">Part {c["part"]}</span>'
                    f'<div class="chunk-text">{c["text"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ---- footer meta ----
        st.markdown(
            f'<div class="footer-meta">'
            f'Latency: {data["latency_ms"]:.0f} ms  •  '
            f'chunking: {data["config_used"]["chunking"]}  •  '
            f'retriever: {data["config_used"]["retriever"]}  •  '
            f'rerank: {data["config_used"]["rerank"]}  •  '
            f'LLM mode: {data["config_used"]["llm_mode"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Open article detail (from chip click)
# ---------------------------------------------------------------------------
open_art = st.session_state.get("open_article")
if open_art:
    st.divider()
    st.subheader(f"Article {open_art} — full text")
    art, err = fetch_article(str(open_art))
    if art is None:
        st.error(f"Could not load Article {open_art}: {err}")
    else:
        st.markdown(
            f'<div class="chunk-card">'
            f'<div style="font-weight:700;color:var(--emerald);font-size:1.05rem;margin-bottom:0.4rem;">'
            f'Article {art["article"]} — {art["title"]}</div>'
            f'<div style="color:var(--muted);font-size:0.85rem;">Part {art["part"]}</div>'
            f'<div style="margin-top:0.6rem;color:var(--ink);line-height:1.6;">{art["text"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if st.button("Close article detail"):
        st.session_state.pop("open_article", None)
        st.rerun()
