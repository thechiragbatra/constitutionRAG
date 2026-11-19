"""Stage 7: Evaluation.

Reads data/eval/qa_set.jsonl (question, gold_articles, reference_answer, answerable)
and writes CSVs + matplotlib graphs to results/.

Metrics: Recall@k, MRR, nDCG@k, token F1, ROUGE-L, citation precision/recall,
abstention precision/recall.

Ablations (one factor at a time, all from config.yaml):
  chunking, retriever, reranker, top_k, embedding_model, rag_vs_noretrieval,
  abstain_threshold.

Run:  python src/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path so `src.` imports work when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import re
import csv

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.retrieve import get_retriever, verify_citations, cited_articles_in_text
from src.generate import Generator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
QA_PATH = ROOT / "data" / "eval" / "qa_set.jsonl"
RESULTS_DIR = ROOT / "results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_qa_set(path: Path = QA_PATH) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- metrics
def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def token_f1(pred: str, ref: str) -> float:
    p, r = _token_set(pred), _token_set(ref)
    if not p or not r:
        return 0.0
    inter = len(p & r)
    prec = inter / len(p)
    rec = inter / len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 via longest common subsequence (token level)."""
    ptok = re.findall(r"\w+", pred.lower())
    rtok = re.findall(r"\w+", ref.lower())
    if not ptok or not rtok:
        return 0.0
    m, n = len(ptok), len(rtok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (
                dp[i][j] + 1 if ptok[i] == rtok[j] else max(dp[i][j + 1], dp[i + 1][j])
            )
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def recall_at_k(retrieved_arts: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0  # nothing to retrieve for non-answerable
    return 1.0 if any(a in gold_set for a in retrieved_arts[:k]) else 0.0


def mrr(retrieved_arts: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    for i, a in enumerate(retrieved_arts, start=1):
        if a in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_arts: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    dcg = 0.0
    for i, a in enumerate(retrieved_arts[:k], start=1):
        rel = 1.0 if a in gold_set else 0.0
        dcg += rel / math.log2(i + 1)
    ideal = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def question_retrieved_arts(retrieved: list[dict]) -> list[str]:
    arts = []
    for c in retrieved:
        arts.extend(str(a) for a in c["articles"])
    # preserve order, dedupe
    seen = set()
    out = []
    for a in arts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def evaluate_one(retriever, generator, q: dict, top_k: int,
                 retriever_kind: str, rerank: bool, threshold: float,
                 use_retrieval: bool) -> dict:
    """Run one question and return per-question metric dict."""
    retrieved = []
    if use_retrieval:
        from src.retrieve import Retriever  # local import to avoid cycles
        retrieved = retriever.search(q["question"], top_k=top_k,
                                     retriever=retriever_kind)
        if rerank:
            retrieved = retriever.rerank(q["question"], retrieved)

    ret_arts = question_retrieved_arts(retrieved)
    gold = q["gold_articles"]
    answerable = bool(q["answerable"])

    if not use_retrieval:
        answer = generator.generate(q["question"], [], use_retrieval=False)
    else:
        answer = generator.generate(q["question"], retrieved, use_retrieval=True)

    abstained = False
    if retrieved:
        from src.retrieve import should_abstain
        try:
            best = retriever.dense_best_score(q["question"])
        except Exception:
            best = 0.0
        if should_abstain(best, threshold):
            abstained = True

    if abstained:
        answer = "Not found in the Constitution"

    valid, flagged = verify_citations(answer, retrieved)
    cited = valid

    return {
        "question": q["question"],
        "answerable": answerable,
        "gold_articles": ",".join(gold),
        "retrieved_articles": ",".join(ret_arts[:top_k]),
        "recall_k": recall_at_k(ret_arts, gold, top_k),
        "mrr": mrr(ret_arts, gold),
        "ndcg_k": ndcg_at_k(ret_arts, gold, top_k),
        "token_f1": token_f1(answer, q["reference_answer"]),
        "rouge_l": rouge_l(answer, q["reference_answer"]),
        "cited_articles": ",".join(cited),
        "citation_precision": (len(set(cited) & set(gold)) / len(cited)
                               if cited else (1.0 if not gold else 0.0)),
        "citation_recall": (len(set(cited) & set(gold)) / len(gold)
                            if gold else (1.0 if not cited else 0.0)),
        "abstained": abstained,
        "answer": answer[:200],
    }


def aggregate(rows: list[dict], k: int) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        f"Recall@{k}": round(np.mean([r["recall_k"] for r in rows]), 4),
        "MRR": round(np.mean([r["mrr"] for r in rows]), 4),
        f"nDCG@{k}": round(np.mean([r["ndcg_k"] for r in rows]), 4),
        "token_F1": round(np.mean([r["token_f1"] for r in rows]), 4),
        "ROUGE-L": round(np.mean([r["rouge_l"] for r in rows]), 4),
        "citation_precision": round(np.mean([r["citation_precision"] for r in rows]), 4),
        "citation_recall": round(np.mean([r["citation_recall"] for r in rows]), 4),
        "abstention_rate": round(np.mean([1 if r["abstained"] else 0 for r in rows]), 4),
    }


def abstention_prf(rows: list[dict]) -> dict:
    """Precision/Recall of abstention as 'predict not answerable'."""
    pred_not_ans = [r for r in rows if r["abstained"]]
    truly_not_ans = [r for r in rows if not r["answerable"]]
    tp = sum(1 for r in pred_not_ans if not r["answerable"])
    prec = tp / len(pred_not_ans) if pred_not_ans else 0.0
    rec = tp / len(truly_not_ans) if truly_not_ans else 0.0
    return {"abstention_precision": round(prec, 4),
            "abstention_recall": round(rec, 4)}


# ---------------------------------------------------------------- run
def run_config(retriever, generator, qa_set, cfg: dict) -> list[dict]:
    return [evaluate_one(retriever, generator, q,
                         cfg["top_k"], cfg["retriever"], cfg["rerank"],
                         cfg["abstain_threshold"], cfg["use_retrieval"])
            for q in qa_set]


def main() -> int:
    cfg = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qa_set = load_qa_set()
    k = cfg["evaluation"]["ks"][0]

    generator = Generator(cfg)
    base = {
        "chunking": cfg["chunking"]["default"],
        "retriever": cfg["retrieval"]["default_retriever"],
        "top_k": cfg["retrieval"]["default_top_k"],
        "rerank": cfg["reranker"]["enabled"],
        "abstain_threshold": cfg["abstain"]["score_threshold"],
        "use_retrieval": True,
    }

    # ---- ablations: one factor at a time ----
    ablation_specs = {
        "chunking": [("article",), ("fixed256",), ("fixed512",)],
        "retriever": [("bm25",), ("dense",), ("hybrid",)],
        "reranker": [("no_rerank", False), ("rerank", True)],
        "top_k": [("k3", 3), ("k5", 5), ("k10", 10)],
        "embedding_model": [(m["name"],) for m in cfg["embedding"]["models"]],
        "abstain_threshold": [("t0.10", 0.10), ("t0.25", 0.25),
                              ("t0.40", 0.40), ("t0.60", 0.60)],
    }

    summary_rows = []
    retrievers = {}  # (chunking, model) -> Retriever

    def get_ret(chunking: str, model_name: str):
        key = (chunking, model_name)
        if key not in retrievers:
            retrievers[key] = get_retriever(chunking, model_name)
        return retrievers[key]

    def run_variant(name: str, **overrides) -> dict:
        c = dict(base)
        c.update(overrides)
        ret = get_ret(c["chunking"], cfg["embedding"]["default"])
        rows = run_config(ret, generator, qa_set, c)
        agg = aggregate(rows, k)
        agg.update(abstention_prf(rows))
        agg["ablation"] = name
        for ov_k, ov_v in overrides.items():
            agg[ov_k] = ov_v
        # save per-question csv
        qpath = RESULTS_DIR / f"qaresults_{name}.csv"
        with qpath.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        summary_rows.append(agg)
        return agg

    # base
    run_variant("base")
    # ablation factors
    for factor, variants in ablation_specs.items():
        for variant in variants:
            if factor == "chunking":
                run_variant(f"chunk_{variant[0]}", chunking=variant[0])
            elif factor == "retriever":
                run_variant(f"retr_{variant[0]}", retriever=variant[0])
            elif factor == "reranker":
                run_variant(f"rerank_{variant[0]}", rerank=variant[1])
            elif factor == "top_k":
                run_variant(f"topk_{variant[0]}", top_k=variant[1])
            elif factor == "embedding_model":
                ret = get_ret(cfg["chunking"]["default"], variant[0])
                rows = run_config(ret, generator, qa_set, base)
                agg = aggregate(rows, k)
                agg.update(abstention_prf(rows))
                agg["ablation"] = f"emb_{variant[0].split('/')[-1]}"
                agg["embedding_model"] = variant[0]
                summary_rows.append(agg)
                qpath = RESULTS_DIR / f"qaresults_emb_{variant[0].split('/')[-1]}.csv"
                with qpath.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
                    w.writeheader()
                    w.writerows(rows)
            elif factor == "abstain_threshold":
                run_variant(f"thresh_{variant[0]}",
                            abstain_threshold=variant[1])

    # RAG vs no-retrieval
    run_variant("rag", use_retrieval=True)
    run_variant("noretrieval", use_retrieval=False)

    # ---- summary CSV ----
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_rows:
        fieldnames = []
        for row in summary_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)

    # ---- graphs ----
    _graphs(summary_rows, RESULTS_DIR)

    print(f"[evaluate] Wrote {len(summary_rows)} summary rows -> {summary_path}")
    print(f"[evaluate] Graphs -> {RESULTS_DIR}")
    return 0


def _graphs(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import pandas as pd
    df = pd.DataFrame(summary_rows)

    # Metric vs ablation for each factor
    metrics = ["Recall@1", "MRR", "nDCG@1", "token_F1", "ROUGE-L",
               "citation_precision", "citation_recall", "abstention_rate"]

    # base line
    base_row = df[df["ablation"] == "base"]
    base_val = {m: float(base_row[m].iloc[0]) for m in metrics if m in df.columns}

    # factor comparison charts
    factors = {
        "chunking": df[df["ablation"].str.startswith("chunk_")],
        "retriever": df[df["ablation"].str.startswith("retr_")],
        "top_k": df[df["ablation"].str.startswith("topk_")],
        "abstain_threshold": df[df["ablation"].str.startswith("thresh_")],
    }

    for factor, sub in factors.items():
        if sub.empty:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 3.5))
            labels = [a.replace(factor + "_", "") for a in sub["ablation"]]
            vals = sub[metric].astype(float)
            ax.bar(labels, vals)
            ax.axhline(base_val.get(metric, 0), color="red", linestyle="--",
                       label=f"base ({base_val.get(metric,0):.3f})")
            ax.set_title(f"{metric} by {factor}")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{metric}_by_{factor}.png", dpi=120)
            plt.close(fig)

    # RAG vs no-retrieval
    comp = df[df["ablation"].isin(["rag", "noretrieval"])]
    if not comp.empty:
        for metric in metrics[:4]:
            if metric not in comp.columns:
                continue
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(comp["ablation"], comp[metric].astype(float))
            ax.set_title(f"{metric}: RAG vs no-retrieval")
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(out_dir / f"{metric}_rag_vs_noretrieval.png", dpi=120)
            plt.close(fig)

    print(f"[evaluate] Graphs saved to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
