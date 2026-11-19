"""Stage 3: Generation.

Builds a citation-grounded prompt from retrieved passages and calls an LLM.
The LLM is switchable via config/env:
  * api   - free API tier (base url / model / key from env)
  * local - small local model (Qwen2.5-1.5B-Instruct or 3B), CPU
  * none  - no-retrieval ablation: LLM answers from parametric knowledge only

If no LLM is configured/reachable, an extractive fallback builds the answer
directly from the retrieved passages (so the demo runs without any key).
Every answer cites Articles as [Article N].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


SYSTEM_INSTRUCTION = (
    "You are a careful legal assistant answering ONLY from the provided "
    "Constitution of India passages. Do not invent Article text. "
    "Cite every claim with the Article number in brackets like [Article 21]. "
    "If the passages do not contain the answer, say so plainly."
)


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Compose the generation prompt from the question and retrieved chunks."""
    ctx_parts = []
    for c in retrieved:
        arts = ", ".join([f"[Article {a}]" for a in c["articles"]])
        ctx_parts.append(f"Passage ({arts}): {c['text']}")
    context = "\n\n".join(ctx_parts) if ctx_parts else "No relevant passages found."
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Retrieved Constitution passages:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer (cite Article numbers like [Article 21]):"
    )


def build_no_retrieval_prompt(question: str) -> str:
    """Prompt for the no-retrieval ablation (no context provided)."""
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"No Constitution passages were retrieved for this question.\n"
        f"Question: {question}\n"
        f"Answer:"
    )


class Generator:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or load_config()
        gen = self.cfg["generation"]
        self.mode = os.getenv("RAG_LLM_MODE", gen["llm_mode"])
        self.api_base = os.getenv("RAG_API_BASE", gen["api_base"])
        self.api_model = os.getenv("RAG_API_MODEL", gen["api_model"])
        self.api_key = os.getenv(gen["api_key_env"], "")
        self.local_model = os.getenv("RAG_LOCAL_MODEL", gen["local_model"])
        self.max_new_tokens = int(gen["max_new_tokens"])
        self.temperature = float(gen["temperature"])
        self._local_pipe = None

    # ---------------------------------------------------------------- LLM calls
    def _call_api(self, prompt: str) -> str:
        import requests  # light, optional dependency
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_local(self, prompt: str) -> str:
        from transformers import pipeline
        if self._local_pipe is None:
            self._local_pipe = pipeline(
                "text-generation",
                model=self.local_model,
                device=-1,  # CPU
                torch_dtype="auto",
            )
        out = self._local_pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()

    def _extractive(self, question: str, retrieved: list[dict]) -> str:
        """Fallback: build a cited answer straight from retrieved passages."""
        if not retrieved:
            return "Not found in the Constitution."
        parts = []
        seen_arts = []
        for c in retrieved:
            for a in c["articles"]:
                if a not in seen_arts:
                    seen_arts.append(a)
            parts.append(f"[Article {', '.join(c['articles'])}] {c['text']}")
        answer = "\n\n".join(parts)
        if len(answer) > 1200:
            answer = answer[:1200] + "..."
        return answer

    # ---------------------------------------------------------------- entrypoint
    def generate(self, question: str, retrieved: list[dict],
                 use_retrieval: bool = True) -> str:
        """Return an answer string. use_retrieval=False => no-retrieval ablation."""
        prompt = build_prompt(question, retrieved) if use_retrieval else build_no_retrieval_prompt(question)

        if self.mode == "api" and self.api_key:
            try:
                return self._call_api(prompt)
            except Exception as exc:  # pragma: no cover - network
                return f"(API error: {exc})\n\n" + self._extractive(question, retrieved)

        if self.mode == "local":
            try:
                return self._call_local(prompt)
            except Exception as exc:  # pragma: no cover - model load
                return f"(Local model error: {exc})\n\n" + self._extractive(question, retrieved)

        # default / 'none' with no LLM: extractive grounded answer
        return self._extractive(question, retrieved)
