"""Build a FAISS dense index, a BM25 index, and a CLIP index over figure images.

Metadata on every vector: doc_id, page, figure label, trust_level.
Low-trust chunks are not in the chunk file, so they cannot enter the index.
"""

from __future__ import annotations

import json
import pickle

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mmrag.config import (
    CHUNKS_PATH,
    CLIP_NAME,
    EMBED_NAME,
    INDEX_DIR,
    SKIP_CLIP,
    resolve,
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if tok]


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open() as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    bad = [c for c in chunks if c.get("trust_level") == "low"]
    if bad:
        raise RuntimeError(f"Refusing to index {len(bad)} low-trust chunks.")
    return chunks


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_clip(chunks: list[dict]) -> None:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    figures = [c for c in chunks if c["type"] == "figure" and c.get("image_path")]
    if not figures:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    vectors = []
    ids = []
    for chunk in tqdm(figures, desc="clip"):
        image = Image.open(resolve(chunk["image_path"])).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            emb = model.get_image_features(**inputs)
        if not torch.is_tensor(emb):
            emb = emb.image_embeds if hasattr(emb, "image_embeds") else emb.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        vectors.append(emb[0].detach().float().cpu().numpy())
        ids.append(chunk["chunk_id"])
    matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(INDEX_DIR / "clip.index"))
    (INDEX_DIR / "clip_ids.json").write_text(json.dumps(ids))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build() -> None:
    chunks = load_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(chunks)} chunks with {EMBED_NAME} on {device}")
    model = SentenceTransformer(EMBED_NAME, device=device)
    texts = []
    for chunk in chunks:
        label = chunk["label"] or chunk["type"]
        texts.append(
            f"{chunk['doc_id']} page {chunk['page']} {label} trust {chunk['trust_level']}\n{chunk['text']}"
        )
    vectors = embed_texts(model, texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "dense.index"))

    tokenized = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    with (INDEX_DIR / "bm25.pkl").open("wb") as handle:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, handle)

    if SKIP_CLIP:
        print("MM_SKIP_CLIP=1, skipping CLIP figure index")
    else:
        print(f"CLIP figure index with {CLIP_NAME}")
        build_clip(chunks)
    print(f"index ready in {INDEX_DIR}")


if __name__ == "__main__":
    build()
