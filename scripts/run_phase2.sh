#!/usr/bin/env bash
# Every Phase 2 evaluation, in the order the results were produced. Needs the
# built corpus (see README), a CUDA GPU with 8 GB, and requirements-eval.txt.
# Each step writes results/<task>/*.json; model outputs are cached under
# data/eval, so an interrupted run picks up where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}

$PY -m latscript.eval.tokenization
$PY -m latscript.eval.retrieval
$PY -m latscript.eval.surprisal
$PY -m latscript.eval.translate nllb facebook/nllb-200-distilled-600M
# 1.3B at batch 8 is too slow for the full devtest on an 8 GB GPU.
$PY -m latscript.eval.translate nllb facebook/nllb-200-distilled-1.3B --subset core --batch-size 8
$PY -m latscript.eval.translate llm Qwen/Qwen2.5-7B-Instruct --4bit --subset core
$PY -m latscript.eval.classify finetune google/muril-base-cased
$PY -m latscript.eval.classify finetune FacebookAI/xlm-roberta-base
$PY -m latscript.eval.classify zeroshot Qwen/Qwen2.5-7B-Instruct --4bit
$PY -m latscript.eval.report
