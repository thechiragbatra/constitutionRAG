#!/usr/bin/env python3
"""Setup validation script for Constitution RAG project.

Checks that all dependencies, data files, and indexes are in place.
Run this after initial setup to verify the project is ready to use.

Usage:
    python setup_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_mark(passed: bool) -> str:
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def main() -> int:
    print("=" * 70)
    print("Constitution RAG — Setup Validation")
    print("=" * 70)
    
    ROOT = Path(__file__).resolve().parent
    issues = []
    warnings = []
    
    # 1. Check Python version
    print(f"\n{check_mark(True)} Python version: {sys.version.split()[0]}")
    
    # 2. Check required dependencies
    print("\n📦 Checking dependencies...")
    required_modules = [
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("fitz", "pymupdf"),
        ("rank_bm25", "rank-bm25"),
        ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("requests", "requests"),
    ]
    
    for module, package in required_modules:
        try:
            __import__(module)
            print(f"  {check_mark(True)} {package}")
        except ImportError:
            print(f"  {check_mark(False)} {package} — MISSING")
            issues.append(f"Install {package}: pip install {package}")
    
    # 3. Check config file
    print("\n⚙️  Checking configuration...")
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        print(f"  {check_mark(True)} config.yaml exists")
    else:
        print(f"  {check_mark(False)} config.yaml — MISSING")
        issues.append("config.yaml not found in project root")
    
    # 4. Check .env file
    env_path = ROOT / ".env"
    if env_path.exists():
        print(f"  {check_mark(True)} .env file exists")
    else:
        print(f"  {check_mark(False)} .env file — MISSING")
        warnings.append("Copy .env.example to .env (not required for llm_mode=none)")
    
    # 5. Check data directories
    print("\n📁 Checking data structure...")
    data_dirs = ["data/raw", "data/processed", "data/eval", "data/indexes"]
    for d in data_dirs:
        path = ROOT / d
        if path.exists():
            print(f"  {check_mark(True)} {d}/")
        else:
            print(f"  {check_mark(False)} {d}/ — MISSING")
            issues.append(f"Create directory: mkdir -p {d}")
    
    # 6. Check constitution.pdf
    pdf_path = ROOT / "data/raw/constitution.pdf"
    if pdf_path.exists():
        print(f"  {check_mark(True)} data/raw/constitution.pdf ({pdf_path.stat().st_size // 1024} KB)")
    else:
        print(f"  {check_mark(False)} data/raw/constitution.pdf — MISSING")
        warnings.append(
            "Place constitution.pdf in data/raw/ or manually add data/processed/clean.txt"
        )
    
    # 7. Check processed files
    print("\n📄 Checking processed data...")
    processed_files = [
        "data/processed/clean.txt",
        "data/processed/articles.jsonl",
        "data/processed/chunks_article.jsonl",
        "data/processed/chunks_fixed256.jsonl",
        "data/processed/chunks_fixed512.jsonl",
    ]
    processed_ok = True
    for f in processed_files:
        path = ROOT / f
        if path.exists():
            print(f"  {check_mark(True)} {f}")
        else:
            print(f"  {check_mark(False)} {f} — MISSING")
            processed_ok = False
    
    if not processed_ok:
        issues.append("Run pipeline: python src/parse.py && python src/chunk.py")
    
    # 8. Check indexes
    print("\n🔍 Checking indexes...")
    index_variants = ["article", "fixed256", "fixed512"]
    indexes_ok = True
    for variant in index_variants:
        bm25_path = ROOT / f"data/indexes/{variant}/bm25.pkl"
        if bm25_path.exists():
            print(f"  {check_mark(True)} {variant}/bm25.pkl")
        else:
            print(f"  {check_mark(False)} {variant}/bm25.pkl — MISSING")
            indexes_ok = False
    
    if not indexes_ok:
        issues.append("Build indexes: python src/index.py")
    
    # 9. Check eval dataset
    qa_path = ROOT / "data/eval/qa_set.jsonl"
    if qa_path.exists():
        print(f"\n  {check_mark(True)} Evaluation dataset: data/eval/qa_set.jsonl")
    else:
        print(f"\n  {check_mark(False)} Evaluation dataset — MISSING")
        warnings.append("data/eval/qa_set.jsonl not found (optional for basic usage)")
    
    # Summary
    print("\n" + "=" * 70)
    if not issues and not warnings:
        print(f"{GREEN}✓ All checks passed! Your setup is ready.{RESET}")
        print("\nNext steps:")
        print("  1. Start backend:  uvicorn backend.main:app --reload")
        print("  2. Start frontend: streamlit run app.py")
        print("  3. Run tests:      pytest tests/")
        return 0
    
    if warnings:
        print(f"\n{YELLOW}⚠ Warnings:{RESET}")
        for w in warnings:
            print(f"  • {w}")
    
    if issues:
        print(f"\n{RED}✗ Issues found:{RESET}")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\n{RED}Please fix the issues above before running the application.{RESET}")
        return 1
    
    print(f"\n{YELLOW}Setup has warnings but should work.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
