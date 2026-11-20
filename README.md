# Constitution of India — RAG Question Answering

End-to-end Retrieval-Augmented Generation (RAG) QA over the Constitution of India.
A user asks a question (e.g. *"Can the police arrest me without telling me why?"*),
the system finds the relevant Articles, a language model answers citing Article
numbers like `[Article 22]`, and if the Constitution does not contain the answer
it says **"Not found in the Constitution"** instead of guessing.

Built for **CSEG4134 NLP lab project** (final review, report and viva on 27 Nov 2026).
Designed so both team members can open any file and explain every design choice.

---

## Quick Start

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/thechiragbatra/constitutionRAG.git
cd constitutionRAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env if you want to use an API-based LLM (optional)
# The system works fully with llm_mode=none (extractive fallback)
```

### 3. Prepare Data & Build Indexes

**Option A: If you have processed data already** (included in repo):
```bash
# Build indexes from existing processed data
python src/index.py
```

**Option B: Starting from PDF** (if you have constitution.pdf):
```bash
# Place your PDF in data/raw/
# Then run the full pipeline:
python src/parse.py      # PDF → clean.txt
python src/chunk.py      # clean.txt → articles.jsonl + chunks
python src/index.py      # Build BM25 + FAISS indexes
```

### 4. Verify Setup

```bash
python setup_check.py
```

This will validate all dependencies, data files, and indexes are ready.

### 5. Run the Application

**Terminal 1 — Start Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Start Frontend:**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 6. Run Tests (Optional)

```bash
pytest tests/ -v
```

---

## Project structure

```
constitution-rag/
├── data/
│   ├── raw/                 # Place constitution.pdf here
│   ├── processed/           # clean.txt + articles.jsonl + chunks_*.jsonl (generated)
│   └── eval/
│       └── qa_set.jsonl     # 10-question sample eval set (you build the real 100–120 set)
├── src/
│   ├── parse.py             # Stage 0: PDF → clean.txt
│   ├── chunk.py             # Stage 1: Article segmentation → jsonl variants
│   ├── index.py             # Stage 1: BM25 + FAISS indexes (saved to disk)
│   ├── retrieve.py          # Stage 2: BM25 / dense / hybrid RRF + reranker + abstention
│   ├── generate.py          # Stage 3: Citation-grounded generation (api / local / none)
│   └── evaluate.py          # Stage 7: Config-driven ablations + metrics + graphs
├── backend/
│   └── main.py              # Stage 5: FastAPI (POST /ask, GET /article/{n}, /config, /health)
├── app.py                   # Stage 6: Indian Emerald theme Streamlit frontend
├── tests/
│   └── test_retrieve.py     # pytest: exact lookup, RRF fusion, abstain behaviour
├── results/                 # Eval CSVs + matplotlib graphs (generated)
├── report/                  # Place your lab report here
├── config.yaml              # Single settings file (all ablations run from here)
├── requirements.txt         # Pinned dependencies
├── .env.example             # Copy to .env, add your API key
├── Procfile                 # Render / Railway deploy
├── runtime.txt              # Python version pin
└── README.md
```

---

## How the system works (stage by stage)

### Stage 0 — `src/parse.py`  (PDF → clean text)
Reads `data/raw/constitution.pdf` (Government of India edition) with PyMuPDF,
extracts text page by page, normalises whitespace/control characters, and writes
`data/processed/clean.txt`. If no PDF is present it exits gracefully and leaves
any existing `clean.txt` untouched.

**Viva line:** *"We use PyMuPDF (fitz) because it handles the diglot
(Hindi–English) Government of India PDF reliably and preserves page breaks so
Article boundaries stay detectable."*

---

### Stage 1 — `src/chunk.py` + `src/index.py`  (segmentation + indexing)

**`src/chunk.py`** segments `clean.txt` into Articles. The real PDF uses the
layout `<number>. <Title>—<body>` (e.g. `21. Protection of life and personal
liberty.—No person shall be ...`). A regex detects that pattern, captures the
Part (`PART I`…`PART XXII`), and attaches amendment footnotes
(`1. Subs. by …`) to the preceding Article instead of creating fake Articles.
Output files (each record: `chunk_id, articles, part, title, text`):
- `articles.jsonl` — one Article per record
- `chunks_article.jsonl` — Article-level chunks
- `chunks_fixed256.jsonl` — fixed ~256-token chunks
- `chunks_fixed512.jsonl` — fixed ~512-token chunks

**`src/index.py`** builds, for every chunking variant × every embedding model:
- **BM25** (`rank_bm25`) → `bm25.pkl`
- **FAISS** `IndexFlatIP` on L2-normalised vectors (= cosine similarity) → `.faiss` + `.meta.json`

Embedding models: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `multilingual-e5-small`.
Fixed seed (42) makes everything deterministic.

**Viva lines:**
- *"chunk_id is `art_<part>_<article>` so Article numbers that repeat across Parts stay unique."*
- *"FAISS IndexFlatIP with L2-normalised vectors gives exact cosine similarity — no approximation, fully explainable."*
- *"One FAISS index per embedding model per chunking variant lets us ablate each factor independently."*

---

### Stage 2 — `src/retrieve.py`  (BM25 / dense / hybrid + reranker)

Three retrievers, each returning top-k chunks with scores:
- **BM25** — keyword overlap. Catches exact Article numbers and legal terms.
- **Dense** — semantic similarity via the embedding model + FAISS cosine.
- **Hybrid** — Reciprocal Rank Fusion (RRF) of BM25 + dense ranks, so neither
  retriever dominates. RRF score = `1/(k + rank_bm25) + 1/(k + rank_dense)`.

**Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-12-v2`): rescores
the top-20 candidates and returns the top-5, improving precision.

**Abstention signal:** `dense_best_score()` returns the top cosine similarity
(0–1). If it is below the config threshold, the system abstains.

**Citation verification:** every `[Article N]` in the generated answer is checked
against the Articles in the retrieved passages; mismatches are flagged.

**Viva lines:**
- *"RRF lets us combine keyword and semantic search without training weights — it is parameter-light and robust."*
- *"The cross-encoder is a bi-encoder-aware second stage: cheap first pass (BM25+dense), precise second pass (cross-encoder)."*
- *"We abstain on cosine similarity (0–1 range) rather than RRF score, because RRF scores are tiny and not comparable across queries."*

---

### Stage 3 — `src/generate.py`  (LLM switching + no-retrieval ablation)

Builds a prompt that instructs the model to answer **only from the retrieved
passages** and cite Articles as `[Article N]`. If the passages do not contain
the answer, the model must say so.

LLM mode is switchable via environment variable `RAG_LLM_MODE` (or config):
- **`none`** (default) — extractive fallback: builds the answer straight from
  retrieved passages with citations. No API key needed; runs anywhere.
- **`api`** — free API tier (`RAG_API_BASE`, `RAG_API_MODEL`, `RAG_API_KEY`).
- **`local`** — Qwen2.5-1.5B-Instruct / 3B on CPU via `transformers`.

`use_retrieval=false` disables retrieval entirely for the **RAG vs no-retrieval
ablation** required in the lab.

**Viva line:** *"The extractive fallback means the demo works without any API
key or GPU — important for the college lab environment."*

---

### Stage 4 — Abstention (inside `src/retrieve.py` + `backend/main.py`)

If the best cosine similarity is below `config.yaml: abstain.score_threshold`,
the backend returns `"Not found in the Constitution"` with no generated answer.
Every cited Article is then verified against retrieved passages; any Article
cited but not retrieved is flagged. This prevents hallucination on questions
like *"Who is the current Prime Minister?"* which the Constitution cannot answer.

---

### Stage 5 — `backend/main.py`  (FastAPI)

Endpoints, indexes loaded once at startup:
- `GET /health` — service status + articles loaded
- `GET /config` — current configuration
- `GET /article/{number}` — exact Article text (e.g. `/article/21`)
- `POST /ask` — JSON `{question, chunking, retriever, top_k, rerank,
  abstain_threshold, use_retrieval}` → `{answer, cited_articles,
  flagged_citations, retrieved[], abstained, abstain_reason, exact_article,
  latency_ms, config_used}`

**Exact-lookup shortcut:** questions like `"Article 21"` or `"What does Article
14 say"` hit the in-memory Article index directly and are also injected into
retrieval so the citation chain stays intact.

Pydantic models validate request/response. CORS enabled so the Streamlit
frontend can call it from another origin.

**Viva line:** *"Loading indexes once at startup keeps per-request latency low
(~1–2 s on CPU) — critical for a live demo during the viva."*

---

### Stage 6 — `app.py`  (Indian Emerald theme Streamlit frontend)

Frontend talks to FastAPI over HTTP. Theme colours are the Indian flag palette
with **emerald green** as primary:
- Saffron `#FF9933`, White `#FFFFFF`, Emerald `#138808`, Navy `#000080`

Features:
- Tricolour header band
- Question box + example-question selector
- Sidebar: chunking, retriever, top-k, reranker, abstain threshold, no-RAG toggle
- Answer card with emerald left border
- **Cited Articles shown as clickable chips** that open the full Article text
- Expandable retrieved-chunks list with scores, ranks, Articles, Part
- **Clear saffron banner when the system abstains**
- Latency and config-used footer
- Article detail panel opened from chips

**Viva line:** *"The emerald theme mirrors the Indian flag so the interface
visually connects to the Constitution of India, and every chip is a live HTTP
call to `/article/{n}` for the full text."*

---

### Stage 7 — `src/evaluate.py`  (evaluation + ablations)

Reads `data/eval/qa_set.jsonl` (fields: `question, gold_articles,
reference_answer, answerable`) and writes CSVs + matplotlib graphs to `results/`.

Metrics:
- **Recall@k**, **MRR**, **nDCG@k** — retrieval quality
- **token F1**, **ROUGE-L** — answer similarity to reference
- **citation precision / recall** — are cited Articles correct and complete?
- **abstention precision / recall** — does abstention correctly catch
  non-answerable questions?

Ablations (one factor at a time, all from `config.yaml`):
- chunking, retriever, reranker, top-k, embedding model, RAG vs no-retrieval,
  abstain-threshold sweep

Run: `python src/evaluate.py` → `results/summary.csv` + per-variant CSVs +
PNG graphs.

---

## Run commands (one-time setup then stages)

```bash
# 0. Create venv and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Parse the Constitution PDF
#    Put your PDF at data/raw/constitution.pdf
.venv/bin/python src/parse.py [path/to/constitution.pdf]

# 2. Chunk into Articles + fixed variants
.venv/bin/python src/chunk.py

# 3. Build BM25 + FAISS indexes (saved to data/indexes/)
.venv/bin/python src/index.py

# 4. Run the pytest suite
.venv/bin/python -m pytest tests/

# 5. Start the FastAPI backend
.venv/bin/uvicorn backend.main:app --reload   # http://localhost:8000

# 6. Start the Streamlit frontend (another terminal)
.venv/bin/streamlit run app.py                # http://localhost:8501

# 7. Run evaluation (config-driven ablations, CSVs + graphs -> results/)
.venv/bin/python src/evaluate.py
```

### Example `curl` test

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Can the police arrest me without telling me why?","top_k":5}'
```

---

## Configuration (`config.yaml`)

Single YAML file drives the whole project — change one value and re-run the
relevant stage; no code edits for ablations.

| Section | Key | Meaning |
|---|---|---|
| `chunking.variants` | list | `article`, `fixed256`, `fixed512` |
| `chunking.default` | string | default chunking variant |
| `embedding.models` | list | name + dim per embedding model |
| `embedding.default` | string | default embedding model |
| `retrieval.default_retriever` | `bm2nd`/`dense`/`hybrid` | default retriever |
| `retrieval.default_top_k` | int | default number of chunks |
| `retrieval.rrf_k` | int | RRF constant (default 60) |
| `reranker.model` | string | cross-encoder model |
| `reranker.enabled` | bool | rerank on by default |
| `abstain.score_threshold` | float | cosine threshold (default 0.25) |
| `generation.llm_mode` | `none`/`api`/`local` | LLM backend |
| `generation.api_key_env` | string | env var name for API key |
| `evaluation.ks` | list | k values for Recall@k / nDCG@k |

---

## LLM setup (`.env`)

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# edit .env and set RAG_API_KEY (only needed for llm_mode=api)
```

The system runs fully in `llm_mode=none` (extractive fallback) with no key and
no GPU — suitable for college lab machines.

---

## Deployment (Render / Railway)

The backend is deploy-ready:
- `Procfile` — `web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- `runtime.txt` — Python version pin
- `.env.example` — copy to `.env` and add `RAG_API_KEY` if using the API tier

Render/Railway install `requirements.txt`, run the Procfile command, and expose
the FastAPI service. Point the Streamlit `BACKEND_URL` secret at the deployed
backend URL.

---

## Viva talking points (one line per design choice, open any file)

1. **BM25** — keyword overlap catches exact Article numbers and legal terms.
2. **Dense (FAISS)** — semantic similarity catches paraphrased questions;
   IndexFlatIP on L2-normalised vectors = exact cosine.
3. **Hybrid (RRF)** — merges BM25 + dense ranks so neither dominates; robust to
   both keyword and semantic queries.
4. **Cross-encoder reranker** — rescores top-20 with ms-marco-MiniLM, returns top-5.
5. **Abstention** — if best cosine < threshold, answer "Not found in the Constitution"
   rather than guess; every cited Article is verified against retrieved passages.
6. **No-retrieval ablation** — `use_retrieval=false` measures RAG vs parametric
   knowledge for the lab comparison.
7. **Exact-lookup shortcut** — "Article 21" / "What does Article 14 say" hits the
   article index directly and also goes through retrieval.
8. **Single `config.yaml`** — all ablations run from one script; reproducible.
9. **Fixed seed 42** — deterministic chunk ordering, no shuffling.
10. **Never invent Article text** — everything shown comes from the indexed corpus.

---

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers: exact-lookup (Article 21 / unknown), RRF fusion structure, BM25 finds
Article 22 for arrest queries, abstain threshold logic, citation verification
flags missing Articles, prompt contains citation format.

---

## Repository

https://github.com/thechiragbatra/constitutionRAG

Branch: `main`
