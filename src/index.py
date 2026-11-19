"""Stage 1: Build and save BM25 + FAISS indexes.

For every chunking variant (article, fixed256, fixed512) and every embedding
model (all-MiniLM-L6-v2, bge-small-en-v1.5, multilingual-e5-small):
  * BM25 index over chunk texts  -> data/indexes/<variant>/bm25.pkl
  * FAISS IndexFlatIP (cosine via L2-normalized vectors)
    -> data/indexes/<variant>/<model_slug>.faiss  +  <model_slug>.meta.json

Deterministic: fixed seed, no shuffling, stable chunk ordering.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

# Limit threads before loading torch / faiss to avoid OpenMP-MKL conflicts on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_chunks(processed_dir: Path, variant: str) -> list[dict]:
    path = processed_dir / f"chunks_{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing chunk file: {path}. Run src/chunk.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25(chunks: list[dict], out_dir: Path) -> None:
    corpus = [c["text"] for c in chunks]
    tokenized = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenized)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump({"bm25": bm25, "corpus": corpus, "ids": [c["chunk_id"] for c in chunks]}, fh)
    print(f"[index] BM25 saved -> {out_dir / 'bm25.pkl'} ({len(chunks)} chunks)")


def build_faiss(chunks: list[dict], model_name: str, out_dir: Path) -> None:
    model = SentenceTransformer(model_name, device="cpu")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
    index.add(embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(model_name)
    faiss.write_index(index, str(out_dir / f"{slug}.faiss"))
    meta = {
        "model": model_name,
        "dim": int(dim),
        "ids": [c["chunk_id"] for c in chunks],
        "texts": texts,
        "articles": [c["articles"] for c in chunks],
        "parts": [c["part"] for c in chunks],
        "titles": [c["title"] for c in chunks],
    }
    with (out_dir / f"{slug}.meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    print(f"[index] FAISS ({model_name}) saved -> {out_dir / f'{slug}.faiss'} ({dim}D, {len(chunks)} vecs)")


def main() -> int:
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    index_dir = ROOT / cfg["paths"]["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)
    variants = cfg["chunking"]["variants"]
    models = cfg["embedding"]["models"]

    np.random.seed(cfg["seed"])

    for variant in variants:
        chunks = load_chunks(processed_dir, variant)
        variant_dir = index_dir / variant
        build_bm25(chunks, variant_dir)
        for m in models:
            build_faiss(chunks, m["name"], variant_dir / model_slug(m["name"]))

    print("[index] All indexes built under", index_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
