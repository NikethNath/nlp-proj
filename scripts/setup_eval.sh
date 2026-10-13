#!/usr/bin/env bash
# Evaluation environment (Phase 2-3): Python 3.12, CPU torch, transformers, peft,
# plus a prebuilt llama.cpp server and the two quantized chat models.
#   EVAL_VENV   venv path            (default .venv-eval)
#   LLAMA_DIR   llama.cpp install    (default ~/.local/llamacpp)
#   MODEL_DIR   GGUF directory       (default ~/models)
set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_VENV=${EVAL_VENV:-.venv-eval}
LLAMA_DIR=${LLAMA_DIR:-$HOME/.local/llamacpp}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-b11567}

uv venv --python 3.12 "$EVAL_VENV"
uv pip install --python "$EVAL_VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python "$EVAL_VENV/bin/python" transformers sentencepiece sacrebleu scikit-learn \
    peft tiktoken protobuf requests numpy pandas pytest

if [ ! -x "$LLAMA_DIR/llama-$LLAMA_BUILD/llama-server" ]; then
  mkdir -p "$LLAMA_DIR"
  tmp=$(mktemp -d)
  gh release download "$LLAMA_BUILD" -R ggml-org/llama.cpp -p "llama-$LLAMA_BUILD-bin-ubuntu-x64.tar.gz" -D "$tmp"
  tar xzf "$tmp"/*.tar.gz -C "$LLAMA_DIR"
  rm -rf "$tmp"
fi

mkdir -p "$MODEL_DIR"
"$EVAL_VENV/bin/hf" download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir "$MODEL_DIR"
"$EVAL_VENV/bin/hf" download unsloth/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir "$MODEL_DIR"
echo "ok: $EVAL_VENV, $LLAMA_DIR/llama-$LLAMA_BUILD, $MODEL_DIR"
