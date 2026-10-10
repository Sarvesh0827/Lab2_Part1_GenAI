"""Turn a model JSON object into a row validate_submission.py will accept."""

from __future__ import annotations

import json
import re

from mmrag.ingest import find_labels

STATUS = {"answered", "not_found", "conflict", "false_premise", "session_override"}
LABEL = r"(Figure|Table) ([12]|[A-C])-\d+( \(corrected\))?"
FIG_RE = re.compile(rf"^(NONE|{LABEL}(,{LABEL})*)$")
CIT_RE = re.compile(rf"^(NONE|DOC[1-3]:(p\d+|{LABEL})(,DOC[1-3]:(p\d+|{LABEL}))*)$")

OVERRIDE_RE = re.compile(
    r"\b(assume|assumption|for this conversation only|for this chat only|this chat only|new data point)\b",
    re.IGNORECASE,
)
PERMANENT_RE = re.compile(r"\b(permanently|update the knowledge base|change the knowledge base)\b", re.IGNORECASE)
IN_DOCS_RE = re.compile(
    r"\b(in (your|the) documents|does document \d|document \d itself reports|is that .{0,40} in )\b",
    re.IGNORECASE,
)


def intent(question: str) -> str:
    if PERMANENT_RE.search(question):
        return "permanent"
    if IN_DOCS_RE.search(question) and not OVERRIDE_RE.search(question):
        return "lookup"
    if OVERRIDE_RE.search(question):
        return "override"
    return "lookup"


def retrieval_query(question: str, history: list[dict]) -> str:
    if len(question.split()) <= 14 and history:
        return history[-1]["question"] + "\n" + question
    return question


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw.replace("\n", " "))
        except json.JSONDecodeError:
            return {}


def normalize_figure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]
    labels: list[str] = []
    for part in parts:
        if not part or part.upper() == "NONE":
            continue
        found = find_labels(part) or find_labels(part.replace("-", " "))
        for label in found:
            if label not in labels:
                labels.append(label)
    return labels


def _citations(chunk: dict) -> list[str]:
    cites = [f"{chunk['doc_id']}:p{chunk['page']}"]
    if chunk.get("label"):
        cites.append(f"{chunk['doc_id']}:{chunk['label']}")
    return cites


def repair(parsed: dict, question: str, chunks: list[dict], why: str) -> dict:
    known_figures = set()
    known_cites = set()
    for chunk in chunks:
        if chunk.get("label"):
            known_figures.add(chunk["label"])
            known_cites.add(f"{chunk['doc_id']}:{chunk['label']}")
        known_cites.add(f"{chunk['doc_id']}:p{chunk['page']}")

    named = find_labels(question)
    figures = normalize_figure_list(parsed.get("figure_references"))
    if named:
        figures = [lab for lab in named if lab in known_figures] or named
    else:
        figures = [lab for lab in figures if lab in known_figures]

    raw_cites = parsed.get("citations") or []
    if isinstance(raw_cites, str):
        raw_cites = [p.strip() for p in raw_cites.split(",")]
    cites: list[str] = []
    for cite in raw_cites:
        cite = re.sub(r"\s+", " ", str(cite).strip())
        cite = cite.replace(" :", ":").replace(": ", ":")
        if cite in known_cites and cite not in cites:
            cites.append(cite)
    if named:
        for lab in figures:
            for chunk in chunks:
                if chunk.get("label") == lab:
                    token = f"{chunk['doc_id']}:{lab}"
                    if token not in cites:
                        cites.insert(0, token)
                    page = f"{chunk['doc_id']}:p{chunk['page']}"
                    if page not in cites:
                        cites.append(page)
    if not cites and why != "not_found":
        for chunk in chunks[:2]:
            for token in _citations(chunk):
                if token not in cites:
                    cites.append(token)

    status = str(parsed.get("status") or "").strip()
    if why == "override":
        status = "session_override"
    elif why == "permanent":
        status = "answered"
    elif status not in STATUS:
        status = "not_found" if not chunks else "answered"
    if status == "not_found" and not named:
        figures, cites = [], []

    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        if why == "permanent":
            answer = "The knowledge base is read-only. It was not changed, and this conversation cannot update it."
        elif status == "not_found":
            answer = "That fact is not in the knowledge base."
        else:
            answer = "The retrieved pages do not state a single supported value."

    fig_s = ",".join(figures) if figures else "NONE"
    cit_s = ",".join(cites) if cites else "NONE"
    if not FIG_RE.match(fig_s):
        fig_s = "NONE"
    if not CIT_RE.match(cit_s):
        cit_s = ",".join(c for c in cites if CIT_RE.match(c)) or "NONE"
    if status not in STATUS:
        status = "answered"
    return {
        "answer": answer,
        "figure_references": fig_s,
        "citations": cit_s,
        "status": status,
    }
