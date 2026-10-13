#!/usr/bin/env bash
# Phase 2-3 end to end. Every step resumes if interrupted, and a failed step
# does not stop the ones after it. Sizes are those used for the reported
# results, on a shared 8-core laptop CPU; raise them on a GPU machine.
#   N_NLLB  FLORES devtest sentences for NLLB and its LoRA (default 200)
#   N_LLM   FLORES devtest sentences for Gemma (default 30); Qwen gets N_QWEN (20)
#   N_POS   offensive comments per language for LLM classification (15, +3 negatives each)
#   RUN_QWEN=1  also run Qwen3-4B (off by default: native script costs it ~4x the tokens)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${EVAL_PY:-.venv-eval/bin/python}"
SERVER="${LLAMA_SERVER:-$HOME/.local/llamacpp/llama-b11567/llama-server}"
MODELS="${MODEL_DIR:-$HOME/models}"
N_NLLB=${N_NLLB:-200}; N_LLM=${N_LLM:-30}; N_QWEN=${N_QWEN:-20}; N_POS=${N_POS:-15}
export PYTHONPATH=src
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }

serve() {
  pkill -x llama-server; sleep 2
  "$SERVER" -m "$MODELS/$1" -c 6144 --parallel 3 -t 8 --host 127.0.0.1 --port 8091 \
      >"${TMPDIR:-/tmp}/llama-server.log" 2>&1 &
  until curl -sf localhost:8091/health >/dev/null; do sleep 2; done
}

# 0. Inputs for every condition (IndicXlit runs in .venv-xlit). Skipped once done,
#    so the measured IndicXlit throughput is not overwritten by a warm-cache run.
[ -f data/derived/eval/xlit_cost.json ] || step .venv/bin/python -m latscript.eval.prepare

# 1. Dedicated MT model: every level and the two pipeline mitigations.
for L in ta ml; do
  step $PY -m latscript.eval.run --task mt --model nllb-600m --lang $L --n $N_NLLB --writers 3 --conds l1 l2c l2l l3
  step $PY -m latscript.eval.run --task mt --model nllb-600m --lang $L --n $N_NLLB --writers 2 --conds bx cn
done

# 2. Gemma-3-4B (Indic-capable open LLM) through llama.cpp.
serve gemma-3-4b-it-Q4_K_M.gguf
for L in ml ta; do
  step $PY -m latscript.eval.run --task mt --model gemma3-4b --chat gemma3-4b --lang $L --n $N_LLM --writers 2 --conds l1 l2c l3 bx
  step $PY -m latscript.eval.run --task clf --model gemma3-4b --chat gemma3-4b --lang $L --n $N_POS --writers 3 --conds l3 l1 l2 s3
done
for L in ml ta; do
  step $PY -m latscript.eval.run --task mt --model gemma3-4b --chat gemma3-4b --lang $L --n $N_LLM --writers 1 --conds cn fs
  step $PY -m latscript.eval.run --task clf --model gemma3-4b --chat gemma3-4b --lang $L --n $N_POS --conds cn te
done
pkill -x llama-server

# 3. Supervised classifiers on DravidianCodeMix.
for L in ml ta; do step $PY -m latscript.eval.train_clf --model charlr --lang $L; done
# MuRIL as a linear probe (frozen encoder + logistic regression): one-epoch CPU
# fine-tuning stayed at chance on Malayalam. On a GPU, --model muril fine-tunes.
for L in ml ta; do step $PY -m latscript.eval.train_clf --model murilprobe --lang $L; done

# 4. LoRA rung: adapt NLLB on romanized FLORES dev, evaluate on devtest.
[ -f data/models/nllb-lora/train.json ] || step $PY -m latscript.eval.train_lora --epochs 0.5
for L in ml ta; do
  step $PY -m latscript.eval.run --task mt --model nllb-600m --lang $L --n $N_NLLB --writers 2 \
      --conds l1 l2c l3 --adapter data/models/nllb-lora
done

# 5. Qwen3-4B (general open LLM), optional: its tokenizer makes native script ~4x longer.
if [ "${RUN_QWEN:-0}" = 1 ]; then
  serve Qwen3-4B-Instruct-2507-Q4_K_M.gguf
  for L in ml ta; do
    step $PY -m latscript.eval.run --task mt --model qwen3-4b --chat qwen3-4b --lang $L --n $N_QWEN --writers 2 --conds l1 l2c l3 bx
  done
  pkill -x llama-server
fi

# 6. Tokenizer fertility and the summary tables.
[ -f results/fertility.json ] || step $PY -m latscript.eval.fertility
step $PY -m latscript.eval.analyze
echo "=== $(date +%T) done"
