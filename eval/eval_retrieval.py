"""Recall of the hybrid retriever on eval/golden_questions.csv.

This file is not Questions.csv. Task 1.4 asks for a separate 25-question set.
Recall@5 is 1 when a top-5 chunk has the expected document and, when set, the label.
A second pass with dense retrieval only is the baseline the lab write-up asks for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmrag.retrieve import _dense_hits, load, search

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden_questions.csv"


def hit(chunks: list[dict], doc: str, label: str) -> bool:
    for chunk in chunks[:5]:
        if chunk["doc_id"] != doc:
            continue
        if label and chunk.get("label") != label:
            continue
        return True
    return False


def main() -> None:
    golden = pd.read_csv(GOLDEN, dtype=str, keep_default_na=False)
    state = load()
    hybrid_hits = 0
    dense_hits = 0
    rows = []
    for _, row in golden.iterrows():
        if row["id"] == "G25":
            # Appendix C must stay out of the index entirely.
            blob = " ".join(c["text"] for c in state["chunks"]).lower()
            ok = "lehman" not in blob
            hybrid_hits += int(ok)
            dense_hits += int(ok)
            rows.append({"id": "G25", "hybrid": ok, "dense": ok})
            continue
        found = search(row["question"], row["question"], top_k=5)
        dense_idx = _dense_hits(row["question"], 5)
        dense_chunks = [state["chunks"][i] for i in dense_idx]
        h = hit(found, row["expected_doc"], row["expected_label"])
        d = hit(dense_chunks, row["expected_doc"], row["expected_label"])
        hybrid_hits += int(h)
        dense_hits += int(d)
        rows.append({"id": row["id"], "hybrid": h, "dense": d, "top": found[0]["chunk_id"] if found else ""})
    n = len(golden)
    summary = {
        "n": n,
        "hybrid_recall_at_5": round(hybrid_hits / n, 3),
        "dense_only_recall_at_5": round(dense_hits / n, 3),
        "rows": rows,
    }
    out = ROOT / "artifacts" / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n", "hybrid_recall_at_5", "dense_only_recall_at_5")}, indent=2))


if __name__ == "__main__":
    main()
