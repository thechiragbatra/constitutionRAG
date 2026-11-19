"""Stage 1: Segment the cleaned Constitution text into Articles and chunks.

Reads data/processed/clean.txt and writes:
  data/processed/articles.jsonl        (one Article per record)
  data/processed/chunks_article.jsonl  (Article-level chunks)
  data/processed/chunks_fixed256.jsonl (fixed ~256-token chunks)
  data/processed/chunks_fixed512.jsonl (fixed ~512-token chunks)

Every record has: chunk_id, articles (list of Article numbers), part, title, text.

Article detection looks for lines starting with "Article <N>" (case-insensitive).
Part detection looks for "PART <roman>". Titles are best-effort and may be empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

ARTICLE_RE = re.compile(r"^\s*Article\s+([0-9]+[A-Za-z]?)\b", re.IGNORECASE)
PART_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+)\b", re.IGNORECASE)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def token_count(text: str) -> int:
    """Rough token estimate: whitespace-separated tokens."""
    return len(text.split())


def segment_articles(text: str) -> list[dict]:
    """Split clean.txt into Article records: {article, part, title, text, start_line}."""
    lines = text.split("\n")
    articles = []
    current_part = ""
    current_article = None
    current_article_part = ""
    buffer = []

    def flush():
        nonlocal current_article, current_article_part, buffer
        if current_article is not None and buffer:
            articles.append({
                "article": current_article,
                "part": current_article_part,
                "title": "",
                "text": "\n".join(buffer).strip(),
            })
        buffer = []

    for line in lines:
        m_part = PART_RE.match(line)
        if m_part and not ARTICLE_RE.match(line):
            current_part = m_part.group(1).upper()
            continue
        m_art = ARTICLE_RE.match(line)
        if m_art:
            flush()
            current_article = m_art.group(1)
            current_article_part = current_part  # snapshot part at article start
            buffer = [line]
        else:
            if current_article is not None:
                buffer.append(line)
    flush()
    return articles


def fixed_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split text into fixed-size token chunks (approx. words)."""
    tokens = text.split()
    if not tokens:
        return []
    step = size - overlap if overlap < size else size
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i + size])
        if chunk:
            chunks.append(chunk)
        if i + size >= len(tokens):
            break
    return chunks


def build_chunk_records(articles: list[dict], variant: str) -> list[dict]:
    records = []
    if variant == "article":
        for art in articles:
            records.append({
                "chunk_id": f"art_{art['article']}",
                "articles": [art["article"]],
                "part": art["part"],
                "title": art["title"],
                "text": art["text"],
            })
    else:
        size = int(variant.replace("fixed", ""))
        idx = 0
        for art in articles:
            parts = fixed_chunks(art["text"], size)
            for j, ptext in enumerate(parts):
                # Find which articles this fixed chunk touches (its own article here).
                records.append({
                    "chunk_id": f"{variant}_a{art['article']}_{j}",
                    "articles": [art["article"]],
                    "part": art["part"],
                    "title": art["title"],
                    "text": ptext,
                })
                idx += 1
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    clean_path = processed_dir / "clean.txt"

    if not clean_path.exists():
        print(f"[chunk] {clean_path} not found. Run src/parse.py first (or add clean.txt).")
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
