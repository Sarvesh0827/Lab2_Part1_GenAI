"""Paths and model names. Override with environment variables on the 4090."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS / "figures"
CHUNKS_PATH = ARTIFACTS / "chunks.jsonl"
EXCLUDED_PATH = ARTIFACTS / "excluded_low_trust.jsonl"
READINGS_PATH = ARTIFACTS / "figure_readings.json"
INDEX_DIR = ARTIFACTS / "index"

# Open weights that fit a 24 GB 4090 one model at a time.
# VLM reads chart images once. The text model answers questions.
VLM_NAME = os.environ.get("MM_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
LLM_NAME = os.environ.get("MM_LLM", "Qwen/Qwen2.5-7B-Instruct")
EMBED_NAME = os.environ.get("MM_EMBED", "BAAI/bge-large-en-v1.5")
RERANK_NAME = os.environ.get("MM_RERANK", "BAAI/bge-reranker-base")
CLIP_NAME = os.environ.get("MM_CLIP", "openai/clip-vit-base-patch32")

USE_4BIT = os.environ.get("MM_4BIT", "0") == "1"
SKIP_VLM = os.environ.get("MM_SKIP_VLM", "0") == "1"
SKIP_CLIP = os.environ.get("MM_SKIP_CLIP", "0") == "1"
SKIP_RERANK = os.environ.get("MM_SKIP_RERANK", "0") == "1"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DOCS = [
    {
        "doc_id": "DOC1",
        "file": "Document.pdf",
        "trust_level": "high",
        "title": "Blanchard chapters 1-2, fall 2011 forecasts",
    },
    {
        "doc_id": "DOC2",
        "file": "Document2_Supplement_A_Outturns_2011-2013.pdf",
        "trust_level": "high",
        "title": "Supplement A outturns and revisions, March 2014",
    },
    {
        "doc_id": "DOC3",
        "file": "Document3_Briefing_B_Labour_Prices_Policy.pdf",
        "trust_level": "medium",
        "title": "Briefing B re-estimates; Appendix C is low trust",
    },
]

TRUST_RANK = {"high": 3, "medium": 2, "low": 1}


def data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for pdf in kaggle.rglob("Document.pdf"):
            return pdf.parent
    return ROOT


def questions_path() -> Path:
    env = os.environ.get("QUESTIONS_PATH")
    if env:
        return Path(env)
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        found = list(kaggle.rglob("Questions.csv"))
        if found:
            return found[0]
    return data_dir() / "Questions.csv"


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def submission_path() -> Path:
    env = os.environ.get("SUBMISSION_PATH")
    if env:
        return Path(env)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/submission.csv")
    return ROOT / "submission.csv"
