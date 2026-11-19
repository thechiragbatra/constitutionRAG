"""Stage 5: FastAPI backend — single source of truth.

Endpoints:
  POST   /api/ask          - question + config overrides -> answer + citations
  GET    /api/article/{n}  - exact Article text
  GET    /api/config       - current configuration
  GET    /api/health       - service status

Loads indexes once at startup. Imports heavy ML deps lazily so the same file
can be deployed on Vercel (serverless) and run locally.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
ARTICLES: Dict[str, List[dict]] = {}
_ARTICLES_LOADED = False


def _load_articles() -> None:
    global _ARTICLES_LOADED
    if _ARTICLES_LOADED:
        return
    processed_dir = ROOT / CFG["paths"]["processed_dir"]
    path = processed_dir / "articles.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ARTICLES.setdefault(rec["article"], []).append(rec)
    _ARTICLES_LOADED = True


# ---- cached retriever (one per process/cold start) ----
_DEFAULT_RETRIEVER: Optional[Any] = None


def default_retriever() -> Any:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        from src.retrieve import get_retriever
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
    use_retrieval: Optional[bool] = Field(None, description="False => no-retrieval ablation")


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    articles: List[str]
    part: str
    title: str


class AskResponse(BaseModel):
    question: str
    answer: str
    cited_articles: List[str]
    flagged_citations: List[str]
    retrieved: List[ChunkOut]
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
def _exact_article_lookup(question: str) -> Optional[dict]:
    """Return the Article record if the question is an exact Article lookup."""
    _load_articles()
    m = re.search(r"article\s+([0-9]{1,3}[A-Z]?)\b", question, re.IGNORECASE)
    if not m:
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


def _run_retrieval(retriever: Any, question: str, top_k: int,
                   retriever_kind: str, rerank: bool) -> List[dict]:
    candidates = retriever.search(question, top_k=top_k, retriever=retriever_kind)
    if rerank and CFG["reranker"]["enabled"]:
        candidates = retriever.rerank(question, candidates)
    return candidates


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _cited_articles_in_text(text: str) -> List[str]:
    return sorted(set(re.findall(r"\[Article\s+([0-9A-Za-z]+)\]", text, re.IGNORECASE)))


def _verify_citations(answer: str, retrieved: List[dict]) -> tuple:
    retrieved_arts = set()
    for c in retrieved:
        retrieved_arts.update(str(a) for a in c["articles"])
    cited = _cited_articles_in_text(answer)
    valid = [a for a in cited if a in retrieved_arts]
    flagged = [a for a in cited if a not in retrieved_arts]
    return valid, flagged


def _extractive_answer(question: str, retrieved: List[dict]) -> str:
    if not retrieved:
        return "Not found in the Constitution."
    parts = []
    seen = []
    for c in retrieved:
        for a in c["articles"]:
            if a not in seen:
                seen.append(a)
        parts.append(f"[Article {', '.join(c['articles'])}] {c['text']}")
    answer = "\n\n".join(parts)
    if len(answer) > 1200:
        answer = answer[:1200] + "..."
    return answer


def _should_abstain(best_score_val: float, threshold: float) -> bool:
    return best_score_val < threshold


# ---- routes ----
@app.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    _load_articles()
    return HealthOut(
        status="ok",
        articles_loaded=len(ARTICLES),
        indexes_ready=True,
    )


@app.get("/api/config", response_model=ConfigOut)
def config() -> ConfigOut:
    return ConfigOut(config=CFG)


@app.get("/api/article/{number}", response_model=ArticleOut)
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


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.time()
    _load_articles()

    chunking = req.chunking or CFG["chunking"]["default"]
    retriever_kind = req.retriever or CFG["retrieval"]["default_retriever"]
    top_k = req.top_k if req.top_k is not None else CFG["retrieval"]["default_top_k"]
    rerank = req.rerank if req.rerank is not None else CFG["reranker"]["enabled"]
    threshold = req.abstain_threshold if req.abstain_threshold is not None else CFG["abstain"]["score_threshold"]
    use_retrieval = req.use_retrieval if req.use_retrieval is not None else True

    exact = _exact_article_lookup(req.question)

    # Retrieval (skipped in no-retrieval ablation).
    retrieved: List[dict] = []
    retriever = None
    if use_retrieval:
        from src.retrieve import get_retriever
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
        if _should_abstain(best_cosine, threshold):
            abstained = True
            abstain_reason = (
                f"best cosine similarity {best_cosine:.3f} below threshold {threshold}"
            )

    # Generation.
    from src.generate import Generator
    generator = Generator(CFG)
    if abstained:
        answer = "Not found in the Constitution"
        cited_articles: List[str] = []
        flagged: List[str] = []
    else:
        answer = generator.generate(
            req.question, retrieved, use_retrieval=use_retrieval
        )
        valid, flagged = _verify_citations(answer, retrieved)
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
