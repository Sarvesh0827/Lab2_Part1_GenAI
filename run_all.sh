#!/usr/bin/env bash
# Run this on the 4090 after the CUDA PyTorch install in the README.
set -euo pipefail
cd "$(dirname "$0")"

python -m mmrag.ingest
python -m mmrag.describe_figures
python -m mmrag.build_index
python -m mmrag.answer
python validate_submission.py "${SUBMISSION_PATH:-submission.csv}" "${QUESTIONS_PATH:-Questions.csv}"
echo "Submission is valid. On Kaggle, tag the notebook p1-v1.0 and submit that run."
