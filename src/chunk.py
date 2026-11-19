"""Stage 1: Segment the cleaned Constitution text into Articles and chunks.

Real PDF layout (Government of India edition):
  1. Name and territory of the Union.—(1) India, that is Bharat, ...
  21. Protection of life and personal liberty.—No person shall be ...

An Article start is a line matching ``<number>. <Title>—<body>`` (title only is
also accepted). Amendment footnotes (``1. Subs. by ...``) are filtered out and
attached to the preceding Article. Every record has: chunk_id, articles (list),
part, title, text. chunk_id is ``art_<part>_<article>`` so it stays unique even
though Article numbers repeat across Parts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

# optional [footnote marker] then <number>.<Title> [— <body>]
ARTICLE_START_RE = re.compile(
    r"^\s*(?:\[\d+\]\s*)?(\d{1,3}[A-Z]?)\.\s+(.{2,100}?)(?:\s*\u2014\s*(.+))?\s*$",
    re.IGNORECASE,
)
PART_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+)\b", re.IGNORECASE)
# Titles beginning with these are amendment footnotes, not real Articles.
FOOTNOTE_HEADS = (
    "subs.", "ins.", "rep.", "omit.", "sub-clause", "sub-heading",
    "the words", "added by", "art.",
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def token_count(text: str) -> int:
    """Rough token estimate: whitespace-separated tokens."""
    return len(text.split())


def is_footnote_title(title: str) -> bool:
    return any(title.lower().startswith(h) for h in FOOTNOTE_HEADS)


def segment_articles(text: str) -> list[dict]:
    """Split clean.txt into Article records: {article, part, title, text}."""
    lines = text.split("\n")
    articles = []
    current_part = ""
    current_article = None
    current_article_part = ""
    current_article_title = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_article, current_article_title, buffer
        if current_article is not None and buffer:
            articles.append({
                "article": current_article,
                "part": current_article_part,
                "title": current_article_title,
                "text": "\n".join(buffer).strip(),
            })
        buffer = []
        current_article_title = ""

    for line in lines:
        m_part = PART_RE.match(line)
        if m_part and not ARTICLE_START_RE.match(line):
            current_part = m_part.group(1).upper()
            continue
        m_art = ARTICLE_START_RE.match(line)
        if m_art:
            title = m_art.group(2).strip()
            if is_footnote_title(title):
                # Amendment footnote: attach to preceding article, do not start a new one.
                if current_article is not None:
                    buffer.append(line)
                continue
            flush()
            current_article = m_art.group(1)
            current_article_part = current_part
            current_article_title = title
            buffer = [line]
        else:
            if current_article is not None:
                buffer.append(line)
    flush()

    # De-duplicate by chunk id (Article numbers repeat across Parts).
    seen = set()
    unique = []
    for a in articles:
        cid = f"art_{a['part']}_{a['article']}"
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(a)
    return unique


def build_chunk_records(articles: list[dict], variant: str) -> list[dict]:
    records = []
    if variant == "article":
        for art in articles:
            records.append({
                "chunk_id": f"art_{art['part']}_{art['article']}",
                "articles": [art["article"]],
                "part": art["part"],
                "title": art["title"],
                "text": art["text"],
            })
    else:
        size = int(variant.replace("fixed", ""))
        for art in articles:
            base = f"{variant}_a{art['part']}_{art['article']}"
            parts = fixed_chunks(art["text"], size)
            for j, ptext in enumerate(parts):
                records.append({
                    "chunk_id": f"{base}_{j}",
                    "articles": [art["article"]],
                    "part": art["part"],
                    "title": art["title"],
                    "text": ptext,
                })
    return records


def fixed_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split text into fixed-size token chunks (approx. words)."""
    tokens = text.split()
    if not tokens:
        return []
    step = size - overlap if 0 <= overlap < size else size
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i + size])
        if chunk:
            chunks.append(chunk)
        if i + size >= len(tokens):
            break
    return chunks


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    clean_path = processed_dir / "clean.txt"

    if not clean_path.exists():
        print(f"[chunk] {clean_path} not found. Run src/parse.py first.")
        return 1

    text = clean_path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"[chunk] {clean_path} is empty.")
        return 1

    articles = segment_articles(text)
    print(f"[chunk] Segmented {len(articles)} Articles")

    processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(articles, processed_dir / "articles.jsonl")

    for variant in cfg["chunking"]["variants"]:
        records = build_chunk_records(articles, variant)
        out = processed_dir / f"chunks_{variant}.jsonl"
        write_jsonl(records, out)
        print(f"[chunk] Wrote {len(records)} chunks -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
