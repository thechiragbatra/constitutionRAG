"""Stage 2: Retrieval over the saved indexes.

Three retrievers:
  * BM25        - keyword overlap (rank_bm25). Good for exact Article/keyword matches.
  * dense       - semantic similarity via FAISS + sentence-transformers.
  * hybrid      - Reciprocal Rank Fusion (RRF) of BM25 + dense ranks.

Optional cross-encoder (cross-encoder/ms-marco-MiniLM-L-12-v2) rescores the
top-20 candidates down to top-5.

Heavy deps (faiss, sentence-transformers) are imported lazily so this module
can be imported on Vercel (serverless, 500 MB bundle limit) where only BM25
is available.

Every result is a dict: {chunk_id, text, score, articles, part, title, rank}.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _load_json_meta(index_dir: Path, slug: str) -> dict:
    with (index_dir / f"{slug}.meta.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Retriever:
    """Loads BM25 index eagerly; FAISS + sentence-transformers loaded lazily.

    On environments without faiss/torch (e.g. Vercel serverless), BM25-only
    retrieval is used. Dense/hybrid methods raise ImportError if called.
    """

    def __init__(self, variant: str, model_name: str, config: Optional[dict] = None):
        self.cfg = config or load_config()
        self.variant = variant
        self.model_name = model_name
        self.slug = model_slug(model_name)
        self.index_dir = ROOT / self.cfg["paths"]["index_dir"] / variant / self.slug

        # ---- BM25 (eager, always available) ----
        with (self.index_dir.parent / "bm25.pkl").open("rb") as fh:
            bm25_pkg = pickle.load(fh)
        self.bm25 = bm25_pkg["bm25"]
        self.bm25_ids = bm25_pkg["ids"]
        self.bm25_corpus = bm25_pkg["corpus"]

        # ---- Metadata (eager, from JSON - no model loading) ----
        self.meta = _load_json_meta(self.index_dir, self.slug)
        self.ids = self.meta["ids"]
        self.texts = self.meta["texts"]
        self.articles = self.meta["articles"]
        self.parts = self.meta["parts"]
        self.titles = self.meta["titles"]
        self.id_to_pos = {cid: i for i, cid in enumerate(self.ids)}

        # ---- FAISS index (lazy) ----
        self.index = None
        self._faiss_loaded = False

        # lazy models
        self._embed_model = None
        self._cross_encoder = None

    # ---------------------------------------------------------------- FAISS (lazy)
    def _ensure_faiss(self) -> None:
        """Load FAISS index on first dense access."""
        if self._faiss_loaded:
            return
        import faiss  # type: ignore
        self.index = faiss.read_index(str(self.index_dir / f"{self.slug}.faiss"))
        self._faiss_loaded = True

    @property
    def has_faiss(self) -> bool:
        """True if FAISS index is available (requires faiss package)."""
        try:
            import faiss  # type: ignore
            return True
        except ImportError:
            return False

    # ---------------------------------------------------------------- models
    @property
    def embed_model(self):
        from sentence_transformers import SentenceTransformer  # type: ignore
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.model_name, device="cpu")
        return self._embed_model

    @property
    def cross_encoder(self):
        from sentence_transformers import CrossEncoder  # type: ignore
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(
                self.cfg["reranker"]["model"], device="cpu"
            )
        return self._cross_encoder

    # ---------------------------------------------------------------- search
    def _result(self, pos: int, score: float, rank: int) -> dict:
        return {
            "chunk_id": self.ids[pos],
            "text": self.texts[pos],
            "score": float(score),
            "rank": rank,
            "articles": self.articles[pos],
            "part": self.parts[pos],
            "title": self.titles[pos],
        }

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized = _tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, pos in enumerate(top, start=1):
            if scores[pos] <= 0:
                continue
            results.append(self._result(int(pos), float(scores[pos]), rank))
        return results

    def dense_search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.has_faiss:
            raise ImportError("faiss-cpu not installed; dense retrieval unavailable")
        self._ensure_faiss()
        q_emb = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, pos = self.index.search(q_emb, top_k)
        results = []
        for rank, (p, s) in enumerate(zip(pos[0], scores[0]), start=1):
            if p < 0:
                continue
            results.append(self._result(int(p), float(s), rank))
        return results

    def dense_best_score(self, query: str) -> float:
        """Top cosine similarity (0..1) for the query — used as abstention signal.

        Falls back to best BM25 score if FAISS is unavailable.
        """
        if not self.has_faiss:
            # BM25 fallback: normalize best score to [0, 1] roughly
            tokenized = _tokenize(query)
            scores = self.bm25.get_scores(tokenized)
            best = float(scores.max()) if len(scores) else 0.0
            return min(best / 10.0, 1.0)  # rough normalization
        self._ensure_faiss()
        q_emb = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, _ = self.index.search(q_emb, 1)
        return float(scores[0][0]) if scores[0][0] >= 0 else 0.0

    def hybrid_search(self, query: str, top_k: int = 10,
                      rrf_k: Optional[int] = None) -> list[dict]:
        """Reciprocal Rank Fusion of BM25 and dense rankings."""
        if not self.has_faiss:
            # BM25-only fallback
            return self.bm25_search(query, top_k)
        if rrf_k is None:
            rrf_k = self.cfg["retrieval"]["rrf_k"]
        bm25_res = self.bm25_search(query, top_k=top_k * 3)
        dense_res = self.dense_search(query, top_k=top_k * 3)

        bm25_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_res)}
        dense_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_res)}

        fused = {}
        for cid in set(bm25_rank) | set(dense_rank):
            score = 0.0
            if cid in bm25_rank:
                score += 1.0 / (rrf_k + bm25_rank[cid])
            if cid in dense_rank:
                score += 1.0 / (rrf_k + dense_rank[cid])
            fused[cid] = score

        meta_by_id = {cid: i for i, cid in enumerate(self.ids)}
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            pos = meta_by_id[cid]
            results.append(self._result(pos, score, rank))
        return results

    def search(self, query: str, top_k: int = 10,
               retriever: Optional[str] = None) -> list[dict]:
        retriever = retriever or self.cfg["retrieval"]["default_retriever"]
        if retriever == "bm25":
            return self.bm25_search(query, top_k)
        if retriever == "dense":
            return self.dense_search(query, top_k)
        return self.hybrid_search(query, top_k)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-encoder rescore: take top-20, return top-5."""
        if not self.has_faiss:
            return candidates[: self.cfg["reranker"]["final_top_n"]]
        cfg = self.cfg["reranker"]
        top_n = cfg["rerank_top_n"]
        final_n = cfg["final_top_n"]
        pool = candidates[:top_n]
        if not pool:
            return []
        pairs = [[query, c["text"]] for c in pool]
        ce_scores = self.cross_encoder.predict(pairs)
        for c, s in zip(pool, ce_scores):
            c["ce_score"] = float(s)
            c["score"] = float(s)
        pool.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(pool[:final_n], start=1):
            c["rank"] = i
        return pool[:final_n]


# ---------------------------------------------------------------- helpers
def best_score(candidates: list[dict]) -> float:
    """Highest score among candidates (after reranking if applied)."""
    if not candidates:
        return 0.0
    return max(float(c.get("score", 0.0)) for c in candidates)


def should_abstain(best_score_val: float, threshold: float) -> bool:
    """True when the best retrieval score is below the abstain threshold."""
    return best_score_val < threshold


def cited_articles_in_text(text: str) -> list[str]:
    """Extract Article numbers cited as [Article N] in a generated answer."""
    import re
    return sorted(set(re.findall(r"\[Article\s+([0-9A-Za-z]+)\]", text, re.IGNORECASE)))


def verify_citations(answer: str, retrieved: list[dict]) -> tuple:
    """Return (valid_cited, flagged_not_retrieved)."""
    retrieved_arts = set()
    for c in retrieved:
        retrieved_arts.update(str(a) for a in c["articles"])
    cited = cited_articles_in_text(answer)
    valid = [a for a in cited if a in retrieved_arts]
    flagged = [a for a in cited if a not in retrieved_arts]
    return valid, flagged


@lru_cache(maxsize=32)
def get_retriever(variant: str, model_name: str) -> Retriever:
    return Retriever(variant, model_name)
