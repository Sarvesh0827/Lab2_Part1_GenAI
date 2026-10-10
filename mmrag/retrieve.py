"""Hybrid retrieval: exact figure/table lookup, BM25, dense vectors, CLIP, rerank.

Conversation text is never added to these indexes.
"""

from __future__ import annotations

import json
import pickle
import re

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mmrag.config import BGE_QUERY_PREFIX, CLIP_NAME, EMBED_NAME, INDEX_DIR, RERANK_NAME, SKIP_RERANK
from mmrag.ingest import find_labels

_STATE: dict = {}


def tokenize(text: str) -> list[str]:
    return [tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if tok]


def load() -> dict:
    if _STATE:
        return _STATE
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
    if any(c.get("trust_level") == "low" for c in chunks):
        raise RuntimeError("Index contains low-trust chunks; rebuild after ingest.")
    dense = faiss.read_index(str(INDEX_DIR / "dense.index"))
    with (INDEX_DIR / "bm25.pkl").open("rb") as handle:
        bm25_blob = pickle.load(handle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = SentenceTransformer(EMBED_NAME, device=device)
    by_label: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        if chunk.get("label"):
            by_label.setdefault(chunk["label"], []).append(i)
    _STATE.update(
        {
            "chunks": chunks,
            "dense": dense,
            "bm25": bm25_blob["bm25"],
            "embed": embed,
            "by_label": by_label,
        }
    )
    clip_path = INDEX_DIR / "clip.index"
    if clip_path.exists():
        from transformers import CLIPModel, CLIPProcessor

        _STATE["clip_index"] = faiss.read_index(str(clip_path))
        _STATE["clip_ids"] = json.loads((INDEX_DIR / "clip_ids.json").read_text())
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _STATE["clip_model"] = CLIPModel.from_pretrained(CLIP_NAME).to(clip_device)
        _STATE["clip_processor"] = CLIPProcessor.from_pretrained(CLIP_NAME)
        _STATE["clip_device"] = clip_device
    return _STATE


def labels_in_question(question: str) -> list[str]:
    return find_labels(question)


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _dense_hits(query: str, topn: int) -> list[int]:
    state = load()
    vector = state["embed"].encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    _, ids = state["dense"].search(vector, topn)
    return [int(i) for i in ids[0] if i >= 0]


def _bm25_hits(query: str, topn: int) -> list[int]:
    state = load()
    scores = state["bm25"].get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:topn]
    return [int(i) for i in order if scores[i] > 0]


def _clip_hits(query: str, topn: int = 3) -> list[int]:
    state = load()
    if "clip_model" not in state:
        return []
    inputs = state["clip_processor"](text=[query], return_tensors="pt", padding=True).to(state["clip_device"])
    with torch.inference_mode():
        emb = state["clip_model"].get_text_features(**inputs)
    if not torch.is_tensor(emb):
        emb = emb.text_embeds if hasattr(emb, "text_embeds") else emb.pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)
    vector = emb.detach().float().cpu().numpy().astype("float32")
    _, ids = state["clip_index"].search(vector, topn)
    id_to_chunk = {c["chunk_id"]: i for i, c in enumerate(state["chunks"])}
    hits = []
    for local in ids[0]:
        if local < 0:
            continue
        chunk_id = state["clip_ids"][int(local)]
        if chunk_id in id_to_chunk:
            hits.append(id_to_chunk[chunk_id])
    return hits


def _rerank(query: str, idxs: list[int], keep: int) -> list[int]:
    if SKIP_RERANK or len(idxs) <= keep:
        return idxs[:keep]
    state = load()
    if "reranker" not in state:
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state["reranker"] = CrossEncoder(RERANK_NAME, device=device)
    chunks = state["chunks"]
    pairs = [(query, chunks[i]["text"][:1800]) for i in idxs]
    scores = state["reranker"].predict(pairs)
    order = np.argsort(scores)[::-1]
    return [idxs[int(i)] for i in order[:keep]]


def search(query: str, question: str, top_k: int = 6) -> list[dict]:
    state = load()
    chunks = state["chunks"]
    named = labels_in_question(question)
    forced: list[int] = []
    for label in named:
        forced.extend(state["by_label"].get(label, []))

    dense_ids = _dense_hits(query, 24)
    bm25_ids = _bm25_hits(query, 24)
    rankings = [dense_ids, bm25_ids]
    # CLIP is for questions that do not already name a figure.
    if not any(label.startswith("Figure") for label in named):
        rankings.append(_clip_hits(query, 3))
    fused = rrf([r for r in rankings if r])
    # Exact label matches outrank fuzzy hits. This is what "named figure retrieved exactly" requires.
    for idx in forced:
        fused[idx] = fused.get(idx, 0.0) + 1.0
    ranked = [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
    # Keep forced chunks in front, then rerank the rest.
    rest = [i for i in ranked if i not in forced]
    rest = _rerank(query, rest[:20], top_k)
    ordered = []
    for idx in forced + rest:
        if idx not in ordered:
            ordered.append(idx)
    selected = []
    for idx in ordered[: max(top_k, len(forced))]:
        chunk = dict(chunks[idx])
        chunk["score"] = float(fused.get(idx, 0.0))
        if chunk["trust_level"] == "low":
            continue
        selected.append(chunk)
    return selected[:8]

