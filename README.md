# Constitution of India — RAG Question Answering

End-to-end Retrieval-Augmented QA over the Constitution of India. A user asks a
question (e.g. "Can the police arrest me without telling me why?"), the system
retrieves the relevant Articles, a language model answers citing Article numbers
like [Article 22], and if the Constitution does not contain the answer it says
"Not found in the Constitution" instead of guessing.

Built for CSEG4134 NLP lab project. CPU-friendly; embeddings can be recomputed
on free Colab/Kaggle GPU.

## Project structure

```
constitution-rag/
  data/{raw,processed,eval}/        raw PDF, processed text + chunk jsonl, eval set
  src/{parse,chunk,index,retrieve,generate,evaluate}.py   pipeline stages
  backend/main.py                    FastAPI service
  app.py                             Streamlit frontend
  tests/                             pytest suite
  results/                           eval CSVs + matplotlib graphs
  config.yaml                        single settings file (ablations run from here)
  requirements.txt                   pinned dependencies
  README.md
```

## Stage-by-stage run commands

Use the project virtualenv (recreated from `requirements.txt`):

```bash
# 0. Create venv + install deps (one time)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Parse PDF -> data/processed/clean.txt
#    Put your Constitution PDF at data/raw/constitution.pdf
.venv/bin/python src/parse.py [path/to/constitution.pdf]

# 2. Chunk -> articles.jsonl + chunks_article/fixed256/fixed512.jsonl
.venv/bin/python src/chunk.py

# 3. Index -> BM25 + FAISS per chunking variant x embedding model (saved to data/indexes/)
.venv/bin/python src/index.py

# 4. Run pytest
.venv/bin/python -m pytest tests/

# 5. Start FastAPI backend
.venv/bin/uvicorn backend.main:app --reload   # http://localhost:8000

# 6. Start Streamlit frontend (in another terminal)
.venv/bin/streamlit run app.py                # http://localhost:8501

# 7. Evaluation (config-driven ablations, writes CSVs + graphs to results/)
.venv/bin/python src/evaluate.py
```

## Backend API

- `GET /health` — service status
- `GET /config` — current configuration
- `GET /article/{number}` — exact Article text (e.g. `/article/21`)
- `POST /ask` — JSON body `{question, chunking, retriever, top_k, rerank, abstain_threshold, use_retrieval}`
  returns `{answer, cited_articles, flagged_citations, retrieved[], abstained, abstain_reason, exact_article, latency_ms, config_used}`

Example:
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"Can the police arrest me without telling me why?","top_k":5}'
```

## Configuration (`config.yaml`)

Single YAML file drives everything (seed, paths, chunking variants, embedding
models, retrieval, reranker, abstain threshold, generation, evaluation). Change
one value and re-run the relevant stage — no code edits needed for ablations.

Generation LLM mode (set via env or config `generation.llm_mode`):
- `none` (default) — extractive fallback from retrieved passages, no key needed
- `api` — free API tier (`RAG_API_BASE`, `RAG_API_MODEL`, `RAG_API_KEY`)
- `local` — Qwen2.5-1.5B-Instruct / 3B on CPU (`RAG_LOCAL_MODEL`)

Add your API key to `.env` (see `.env.example`).

## Deployment (Render / Railway)

The app is deploy-ready:

- `Procfile` — `web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- `runtime.txt` — Python version
- `.env.example` — copy to `.env` and add `RAG_API_KEY` if using the API tier

Render/Railway will install `requirements.txt`, run the Procfile command, and
expose the FastAPI service. Point the Streamlit `BACKEND_URL` secret at the
deployed backend URL.

## Viva talking points (one line per design choice)

- **BM25**: keyword overlap; catches exact Article numbers and legal terms.
- **Dense (FAISS)**: semantic similarity via sentence-transformers; catches
  paraphrased questions. IndexFlatIP with L2-normalized vectors = cosine.
- **Hybrid (RRF)**: Reciprocal Rank Fusion merges BM25 + dense ranks so neither
  retriever dominates; robust to both keyword and semantic queries.
- **Cross-encoder reranker**: rescores top-20 with ms-marco-MiniLM, returns top-5.
- **Abstention**: if best cosine similarity < threshold (config), answer
  "Not found in the Constitution" rather than guess.
- **Citation verification**: every [Article N] in the answer is checked against
  retrieved passages; mismatches are flagged.
- **No-retrieval ablation**: `use_retrieval=false` measures RAG vs parametric
  knowledge for the lab comparison.
- **Exact-lookup shortcut**: "Article 21" / "What does Article 14 say" hits the
  article index directly and also goes through retrieval.
- **Single config.yaml**: all ablations (chunking, retriever, reranker, top-k,
  embedding model, RAG vs no-retrieval, abstain threshold) run from one script.

## Reproducibility

- Fixed random seed (`config.yaml: seed: 42`)
- Deterministic chunk ordering, no shuffling
- Indexes saved to disk; backend loads them once at startup
