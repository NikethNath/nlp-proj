#!/usr/bin/env bash
# Every Phase 3 mitigation, in order. Needs the Phase 2 setup (README) and a
# CUDA GPU with 8 GB. Results go to results/mitigate/; model outputs are cached
# under data/eval, so an interrupted run picks up where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv/bin/python}
QWEN=Qwen/Qwen2.5-7B-Instruct

# Training on romanized FLORES dev, evaluated on devtest.
$PY -m latscript.mitigate.train retrieval
$PY -m latscript.eval.retrieval --encoders data/models/LaBSE-romanized --phase3
$PY -m latscript.mitigate.train mt
$PY -m latscript.eval.translate nllb data/models/nllb-200-distilled-600M-romanized --phase3

# Input rewriting: back-transliteration (bt) and round-trip normalization (rt).
$PY -m latscript.mitigate.transforms
$PY -m latscript.mitigate.fidelity
$PY -m latscript.eval.retrieval --variant bt
$PY -m latscript.eval.retrieval --variant rt
$PY -m latscript.eval.translate nllb facebook/nllb-200-distilled-600M --variant bt
$PY -m latscript.eval.translate llm $QWEN --4bit --subset core --batch-size 8 --variant bt
$PY -m latscript.eval.translate llm $QWEN --4bit --subset core --batch-size 8 --variant rt

# Offensive-language detection: normalization at inference, script-augmented training.
$PY -m latscript.eval.classify zeroshot $QWEN --4bit --rt
$PY -m latscript.eval.classify finetune google/muril-base-cased --rt
$PY -m latscript.eval.classify finetune google/muril-base-cased --rt --augment

$PY -m latscript.mitigate.report
