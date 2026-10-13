#!/usr/bin/env bash
# Classroom demo wrapper: offline (models load from the local cache), quiet.
#   scripts/demo.sh results
#   scripts/demo.sh corpus --lang ml
#   scripts/demo.sh type --lang ml "njan innu vaikunneram kadappurathu pokum"
#   scripts/demo.sh classify --lang ta "2.06 sema serupadi to thalaivar"
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TQDM_DISABLE=1 PYTHONWARNINGS=ignore
exec "${EVAL_PY:-.venv-eval/bin/python}" -m latscript.demo "$@" 2> >(grep -v -E "Loading weights|^\s*$" >&2)
