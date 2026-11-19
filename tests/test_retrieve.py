"""Pytest suite: exact-lookup, RRF fusion, abstain behaviour.

Run:  pytest tests/
Requires indexes built (run src/index.py first).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.retrieve import (  # noqa: E402
    Retriever, should_abstain, verify_citations, cited_articles_in_text,
)
from src.generate import build_prompt  # noqa: E402
from backend.main import _exact_article_lookup  # noqa: E402


@pytest.fixture(scope="module")
def ret():
    return Retriever("article", "sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------- exact lookup
def test_exact_lookup_article_21():
    res = _exact_article_lookup("What does Article 21 say?")
    assert res is not None
    assert res["article"] == "21"
    assert "Protection of life and personal liberty" in res["title"]
    assert "deprived of his life" in res["text"]


def test_exact_lookup_unknown():
    assert _exact_article_lookup("Who is the current Prime Minister?") is None


# ---------------------------------------------------------------- RRF fusion
def test_hybrid_returns_results(ret):
    res = ret.hybrid_search("can the police arrest without telling why", top_k=5)
    assert len(res) > 0
    for r in res:
        assert set(r.keys()) >= {"chunk_id", "text", "score", "rank", "articles"}
    # ranks should be 1..n
    assert [r["rank"] for r in res] == list(range(1, len(res) + 1))


def test_bm25_finds_arrest_article(ret):
    res = ret.bm25_search("arrest without informing grounds", top_k=5)
    arts = [a for r in res for a in r["articles"]]
    assert "22" in arts, f"Article 22 (arrest protection) should rank high, got {arts}"


# ---------------------------------------------------------------- abstain
def test_should_abstain_below_threshold():
    assert should_abstain(0.10, 0.25) is True
    assert should_abstain(0.50, 0.25) is False


def test_verify_citations_flags_missing(ret):
    retrieved = [{"chunk_id": "x", "articles": ["21"], "text": "life and liberty"}]
    answer = "This is covered by [Article 21] and also [Article 999]."
    valid, flagged = verify_citations(answer, retrieved)
    assert "21" in valid
    assert "999" in flagged


def test_cited_articles_in_text():
    assert cited_articles_in_text("See [Article 21] and [Article 14].") == ["14", "21"]
    assert cited_articles_in_text("No citations here.") == []


# ---------------------------------------------------------------- prompt
def test_build_prompt_contains_citation_format():
    prompt = build_prompt("Q", [{"chunk_id": "a", "text": "life and liberty",
                                 "articles": ["21"]}])
    assert "[Article 21]" in prompt
    assert "Q" in prompt
