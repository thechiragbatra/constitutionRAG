"""Stage 6: Streamlit frontend for the Constitution of India RAG QA system.

Talks to the FastAPI backend over HTTP (POST /ask, GET /article/{n}, /health).
Run:  streamlit run app.py
Backend must be running:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import requests
import streamlit as st


def _backend_url():
    try:
        return st.secrets.get("BACKEND_URL", "http://localhost:8000")
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


def api_get(path: str, timeout: int = 10):
    try:
        return requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
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


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Constitution of India RAG QA", page_icon="\u270d")
st.title("Constitution of India \u2014 RAG Question Answering")
st.caption("Answers only from the indexed Constitution, with Article citations.")

# health
health, err = fetch_json("/health")
if health is None:
    st.error(f"Backend not reachable at `{BACKEND_URL}`. Start it with: "
             "`uvicorn backend.main:app --reload`\n\n{err}")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Retrieval settings")
chunking = st.sidebar.selectbox(
    "Chunking", ["article", "fixed256", "fixed512"], index=0
)
retriever = st.sidebar.selectbox(
    "Retriever", ["hybrid", "bm25", "dense"], index=0
)
top_k = st.sidebar.slider("Top-k chunks", 1, 20, 5)
rerank = st.sidebar.checkbox("Cross-encoder reranker (top-20 \u2192 top-5)", value=False)
abstain_threshold = st.sidebar.slider(
    "Abstain threshold (cosine)", 0.0, 1.0, 0.25, 0.05
)
use_retrieval = st.sidebar.checkbox("Use retrieval (uncheck = no-RAG ablation)", value=True)
st.sidebar.divider()
st.sidebar.caption(f"Backend: `{BACKEND_URL}`")
st.sidebar.caption(f"Articles indexed: {health.get('articles_loaded', '?')}")

# ---------------------------------------------------------------- question
st.subheader("Ask a question")
example = st.selectbox("Example questions", ["\u2014 pick one \u2014"] + EXAMPLES)
default_q = "" if example == "\u2014 pick one \u2014" else example
question = st.text_input("Your question", value=default_q,
                         placeholder="e.g. What does Article 21 say?")

col1, col2 = st.columns([1, 4])
ask_clicked = col1.button("Ask", type="primary", use_container_width=True)

cfg = {
    "chunking": chunking,
    "retriever": retriever,
    "top_k": top_k,
    "rerank": rerank,
    "abstain_threshold": abstain_threshold,
    "use_retrieval": use_retrieval,
}

if ask_clicked and question.strip():
    with st.spinner("Retrieving and generating..."):
        data, err = ask_backend(question, cfg)
    if data is None:
        st.error(f"Request failed: {err}")
    else:
        # ---- abstention banner ----
        if data["abstained"]:
            st.error(
                "\u26a0 **Not found in the Constitution** \u2014 "
                f"{data.get('abstain_reason', '')}"
            )
        else:
            st.success("Answer found in the Constitution.")

        # ---- answer ----
        st.subheader("Answer")
        st.write(data["answer"])

        # ---- cited articles as chips ----
        if data["cited_articles"]:
            st.write("**Cited Articles:**")
            cols = st.columns(min(len(data["cited_articles"]), 4))
            for i, art in enumerate(data["cited_articles"]):
                with cols[i % len(cols)]:
                    if st.button(f"[Article {art}]", key=f"chip_{art}_{i}",
                                 use_container_width=True):
                        st.session_state["open_article"] = art

        # ---- flagged citations ----
        if data["flagged_citations"]:
            st.warning(
                "Citation check flagged (not in retrieved passages): "
                + ", ".join(f"[Article {a}]" for a in data["flagged_citations"])
            )

        # ---- exact article shortcut ----
        if data.get("exact_article"):
            ea = data["exact_article"]
            st.info(
                f"Exact-lookup shortcut hit: Article {ea['article']} "
                f"(Part {ea['part']})."
            )

        # ---- retrieved chunks ----
        with st.expander(f"Retrieved chunks ({len(data['retrieved'])})", expanded=False):
            for c in data["retrieved"]:
                arts = ", ".join(f"[Article {a}]" for a in c["articles"])
                st.markdown(
                    f"**Rank {c['rank']}** \u00b7 score `{c['score']:.4f}` "
                    f"\u00b7 {arts} \u00b7 Part {c['part']}"
                )
                st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
                st.divider()

        st.caption(f"Latency: {data['latency_ms']:.0f} ms \u00b7 "
                   f"chunking={data['config_used']['chunking']} \u00b7 "
                   f"retriever={data['config_used']['retriever']} \u00b7 "
                   f"LLM mode={data['config_used']['llm_mode']}")

# ---------------------------------------------------------------- open article detail
open_art = st.session_state.get("open_article")
if open_art:
    st.divider()
    st.subheader(f"Article {open_art} \u2014 full text")
    art, err = fetch_article(str(open_art))
    if art is None:
        st.error(f"Could not load Article {open_art}: {err}")
    else:
        st.write(f"**Part {art['part']}** \u00b7 {art['title']}")
        st.write(art["text"])
    if st.button("Close article detail"):
        st.session_state.pop("open_article", None)
        st.rerun()
