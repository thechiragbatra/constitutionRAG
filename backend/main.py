"""Stage 5: FastAPI backend.

Endpoints:
  POST   /ask          - question (+ optional config overrides) -> answer + citations
  GET    /article/{n}  - exact Article lookup
  GET    /config       - current configuration
  GET    /health       - service status

Indexes are loaded once at startup (default chunking + embedding model) and
retrievers for other variants are cached on demand.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.retrieve import (
    Retriever, best_score, get_retriever, should_abstain, verify_citations,
)
from src.generate import Generator, build_prompt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_config()
app = FastAPI(title="Constitution of India RAG QA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- in-memory article index for exact lookup ----
ARTICLES: dict[str, list[dict]] = {}
_ARTICLES_LOADED = False


def _load_articles() -> None:
    global _ARTICLES_LOADED
    if _ARTICLES_LOADED:
        return
    processed_dir = ROOT / CFG["paths"]["processed_dir"]
    path = processed_dir / "articles.jsonl"
    if path.exists():
        import json
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ARTICLES.setdefault(rec["article"], []).append(rec)
    _ARTICLES_LOADED = True


# ---- default retriever (loaded at startup) ----
_DEFAULT_RETRIEVER: Optional[Retriever] = None


def default_retriever() -> Retriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = get_retriever(
            CFG["chunking"]["default"],
            CFG["embedding"]["default"],
        )
    return _DEFAULT_RETRIEVER


# ---- request / response models ----
class AskRequest(BaseModel):
    question: str = Field(..., description="User question")
    chunking: Optional[str] = Field(None, description="article | fixed256 | fixed512")
    retriever: Optional[str] = Field(None, description="bm25 | dense | hybrid")
    top_k: Optional[int] = Field(None, description="number of chunks to retrieve")
    rerank: Optional[bool] = Field(None, description="apply cross-encoder reranker")
    abstain_threshold: Optional[float] = Field(None, description="abstain score threshold")
    use_retrieval: Optional[bool] = Field(
        None, description="False => no-retrieval ablation"
    )


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    articles: list[str]
    part: str
    title: str


class AskResponse(BaseModel):
    question: str
    answer: str
    cited_articles: list[str]
    flagged_citations: list[str]
    retrieved: list[ChunkOut]
    abstained: bool
    abstain_reason: Optional[str] = None
    exact_article: Optional[dict] = None
    latency_ms: float
    config_used: dict


class ArticleOut(BaseModel):
    article: str
    part: str
    title: str
    text: str


class ConfigOut(BaseModel):
    config: dict


class HealthOut(BaseModel):
    status: str
    articles_loaded: int
    indexes_ready: bool


# ---- helpers ----
ARTICLE_QUERY_RE = re.compile(
    r"(?:^|\s)(?:article\s+)?([0-9]{1,3}[A-Z]?)\b(?:\s|$|\.|,|\?)",
    re.IGNORECASE,
)


def _exact_article_lookup(question: str) -> Optional[dict]:
    """Return the Article record if the question is an exact Article lookup."""
    _load_articles()
    # Prefer explicit "Article N" phrasing.
    m = re.search(r"article\s+([0-9]{1,3}[A-Z]?)\b", question, re.IGNORECASE)
    if not m:
        # Bare number like "21" or "What does 21 say".
        m = re.search(r"\b([0-9]{1,3}[A-Z]?)\b", question)
    if not m:
        return None
    num = m.group(1)
    matches = ARTICLES.get(num)
    if not matches:
        return None
    rec = matches[0]
    return {
        "article": rec["article"],
        "part": rec["part"],
        "title": rec["title"],
        "text": rec["text"],
    }


def _run_retrieval(retriever: Retriever, question: str, top_k: int,
                   retriever_kind: str, rerank: bool) -> list[dict]:
    candidates = retriever.search(question, top_k=top_k, retriever=retriever_kind)
    if rerank and CFG["reranker"]["enabled"]:
        candidates = retriever.rerank(question, candidates)
    return candidates


# ---- routes ----
@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    _load_articles()
    return HealthOut(
        status="ok",
        articles_loaded=len(ARTICLES),
        indexes_ready=True,
    )


@app.get("/config", response_model=ConfigOut)
def config() -> ConfigOut:
    return ConfigOut(config=CFG)


@app.get("/article/{number}", response_model=ArticleOut)
def get_article(number: str) -> ArticleOut:
    _load_articles()
    matches = ARTICLES.get(number)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Article {number} not found")
    rec = matches[0]
    return ArticleOut(
        article=rec["article"],
        part=rec["part"],
        title=rec["title"],
        text=rec["text"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.time()
    _load_articles()

    # Resolve config overrides with defaults.
    chunking = req.chunking or CFG["chunking"]["default"]
    retriever_kind = req.retriever or CFG["retrieval"]["default_retriever"]
    top_k = req.top_k if req.top_k is not None else CFG["retrieval"]["default_top_k"]
    rerank = req.rerank if req.rerank is not None else CFG["reranker"]["enabled"]
    threshold = req.abstain_threshold if req.abstain_threshold is not None else CFG["abstain"]["score_threshold"]
    use_retrieval = req.use_retrieval if req.use_retrieval is not None else True

    # Exact-lookup shortcut (Article 21, what does Article 14 say, etc.)
    exact = _exact_article_lookup(req.question)

    # Retrieval (skipped in no-retrieval ablation).
    retrieved: list[dict] = []
    if use_retrieval:
        retriever = get_retriever(
            chunking, CFG["embedding"]["default"]
        ) if chunking != CFG["chunking"]["default"] else default_retriever()
        retrieved = _run_retrieval(retriever, req.question, top_k,
                                   retriever_kind, rerank)

    # If an exact article was found, make sure it's in retrieved (dedupe by chunk_id).
    if exact:
        already = any(c["chunk_id"] == f"art_{exact['part']}_{exact['article']}"
                      for c in retrieved)
        if not already:
            retrieved.insert(0, {
                "chunk_id": f"art_{exact['part']}_{exact['article']}",
                "text": exact["text"],
                "score": 1.0,
                "rank": 0,
                "articles": [exact["article"]],
                "part": exact["part"],
                "title": exact["title"],
            })

    # Abstention decision (uses dense cosine, a 0..1 relevance signal).
    abstained = False
    abstain_reason = None
    if CFG["abstain"]["enabled"] and use_retrieval:
        best_cosine = retriever.dense_best_score(req.question) if retrieved else 0.0
        # Even when reranking, fall back to cosine for the abstention threshold
        # so the 0..1 config threshold stays meaningful.
        if should_abstain(best_cosine, threshold):
            abstained = True
            abstain_reason = (
                f"best cosine similarity {best_cosine:.3f} below threshold {threshold}"
            )

    # Generation.
    generator = Generator(CFG)
    if abstained:
        answer = "Not found in the Constitution"
        cited_articles: list[str] = []
        flagged: list[str] = []
    else:
        answer = generator.generate(
            req.question, retrieved, use_retrieval=use_retrieval
        )
        valid, flagged = verify_citations(answer, retrieved)
        cited_articles = valid

    latency_ms = (time.time() - start) * 1000.0

    return AskResponse(
        question=req.question,
        answer=answer,
        cited_articles=cited_articles,
        flagged_citations=flagged,
        retrieved=[
            ChunkOut(
                chunk_id=c["chunk_id"],
                text=c["text"],
                score=c["score"],
                rank=c["rank"],
                articles=c["articles"],
                part=c["part"],
                title=c["title"],
            )
            for c in retrieved[:top_k]
        ],
        abstained=abstained,
        abstain_reason=abstain_reason,
        exact_article=exact,
        latency_ms=round(latency_ms, 2),
        config_used={
            "chunking": chunking,
            "retriever": retriever_kind,
            "top_k": top_k,
            "rerank": rerank,
            "abstain_threshold": threshold,
            "use_retrieval": use_retrieval,
            "llm_mode": generator.mode,
        },
    )
