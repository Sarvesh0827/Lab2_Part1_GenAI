# DATA 266 Lab 2 Part 1 — multimodal RAG

This folder is meant to be copied to the 4090 machine and run there. It builds a local knowledge base from the three PDFs, reads every figure with a vision model, and writes `submission.csv`.

No question from `Questions.csv` is hard-coded. Hard-coded answers score 0 on the hidden set.

## What it does

1. **Ingest.** PyMuPDF pulls page text, tables, and a PNG of every numbered figure. Each chunk stores `doc_id`, page, figure or table label, and trust. Appendix C (low trust) is saved under `artifacts/excluded_low_trust.jsonl` and is not embedded. Instructions printed inside a document are stripped.
2. **Read figures.** `Qwen2.5-VL-7B-Instruct` transcribes each chart (series, years, values) into the figure chunk. That is what makes a named figure answerable. Near-identical pairs (A-2/A-3, A-4/A-5, A-6/A-7, B-3/B-4, B-5 vs B-5 corrected) stay on separate chunks.
3. **Index.** `bge-large-en-v1.5` vectors in FAISS, BM25, and CLIP embeddings of the figure images. Metadata on every row: document, page, label, trust.
4. **Answer.** Hybrid retrieval (exact label match, then dense + BM25 + CLIP, then a reranker). `Qwen2.5-7B-Instruct` returns the JSON fields the validator wants. Chat facts live only in a dict keyed by `conversation_id` and are discarded when that conversation ends. A request to edit the knowledge base permanently is refused.

Status values follow the lab rules: a revision or outturn replaces a forecast, disagreement without a revision note is `conflict` (higher trust wins), a false assumption is `false_premise`, a user value for this chat only is `session_override`, and a missing fact is `not_found`.

## On the 4090

```bash
cd data-266-fall-2026-lab-2-part-1-multimodal-rag
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` writes `submission.csv` and runs `validate_submission.py`. The first run downloads the three models (about 20 GB). Later runs reuse the Hugging Face cache.

Useful switches:

| Variable | Meaning |
| --- | --- |
| `MM_4BIT=1` | Load the VLM and the answer model in 4-bit if memory gets tight |
| `MM_SKIP_CLIP=1` | Skip the CLIP index |
| `MM_SKIP_RERANK=1` | Skip the cross-encoder |
| `MM_SKIP_VLM=1` | Debug without chart reading (figure questions will be weak) |

After the index exists, retrieval quality on the separate 25-question set is:

```bash
python eval/eval_retrieval.py
```

## Kaggle

The notebook has to run end to end with no manual step, within 90 minutes, on a Kaggle GPU. Use the same commands. Point the data at the attached input if the PDFs are not in the working directory:

```bash
export DATA_DIR=/kaggle/input/<competition-slug>
export QUESTIONS_PATH=/kaggle/input/<competition-slug>/Questions.csv
export SUBMISSION_PATH=/kaggle/working/submission.csv
bash run_all.sh
```

Internet must be on for the first model download, or add the models as a Kaggle dataset and set `HF_HOME`. Tag the notebook `p1-v1.0` and submit that output. One final submission per team.

Do not fine-tune on `Questions.csv`. The hidden set is the same kind of question, including reworded ones, and task-specific rules score 0.

## Layout

```
mmrag/ingest.py            PDF text, tables, figure crops, trust filter
mmrag/describe_figures.py  VLM chart transcription
mmrag/build_index.py       FAISS + BM25 + CLIP
mmrag/retrieve.py          exact label, hybrid, rerank
mmrag/answer.py            conversation memory and submission.csv
eval/golden_questions.csv  25 questions that are not the leaderboard file
```
