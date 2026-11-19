"""Stage 0: Parse the Constitution PDF into a single clean text file.

Reads data/raw/constitution.pdf (if present) and writes data/processed/clean.txt.
Each page's text is extracted, normalized (whitespace collapsed, control chars
removed), and concatenated with form-feed separators so Article boundaries stay
detectable by src/chunk.py.

Usage:
    python src/parse.py [pdf_path]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

try:  # PyMuPDF (imported as fitz in older releases)
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    fitz = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalize(text: str) -> str:
    """Collapse whitespace and strip control characters, keep page structure."""
    text = text.replace("\x0c", "\n")  # form feed -> newline
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", text)  # drop control chars
    text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines
    return text.strip()


def parse_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF, page by page."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed; cannot parse PDF")
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n\n--PAGE BREAK--\n\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(argv[0]) if argv else raw_dir / "constitution.pdf"
    out_path = processed_dir / "clean.txt"

    if not pdf_path.exists():
        print(f"[parse] PDF not found at {pdf_path}; leaving {out_path} untouched.")
        print("[parse] Add data/raw/constitution.pdf and re-run, or place clean.txt manually.")
        return 0

    text = parse_pdf(pdf_path)
    text = normalize(text)
    out_path.write_text(text, encoding="utf-8")
    print(f"[parse] Wrote {len(text)} chars to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
