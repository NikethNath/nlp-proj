"""Tokenizer fertility (tokens per word) for each orthographic level.

The word count is the whitespace word count of the native sentence, so all
levels share a denominator. Only tokenizer files are downloaded. Gated
official repos are replaced by ungated mirrors carrying the same tokenizer.

Writes results/fertility.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from .prepare import DERIVED, read_jsonl

ROOT = Path(__file__).resolve().parents[3]
TOKENIZERS = {
    "GPT-4o (o200k)": "tiktoken:o200k_base",
    "Llama-3.1": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "Llama-4": "unsloth/Llama-4-Scout-17B-16E-Instruct",
    "Qwen3": "Qwen/Qwen3-8B",
    "Gemma-3": "unsloth/gemma-3-4b-it",
    "Sarvam-1": "sarvamai/sarvam-1",
    "MuRIL": "google/muril-base-cased",
    "NLLB-200": "facebook/nllb-200-distilled-600M",
}


def encoder(spec: str):
    if spec.startswith("tiktoken:"):
        import tiktoken
        enc = tiktoken.get_encoding(spec.split(":", 1)[1])
        return lambda t: len(enc.encode(t))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec)
    return lambda t: len(tok(t, add_special_tokens=False)["input_ids"])


def main() -> None:
    data = {lang: read_jsonl(DERIVED / f"flores_{lang}_devtest.jsonl") for lang in ("ml", "ta")}
    out = {}
    for name, spec in TOKENIZERS.items():
        try:
            count = encoder(spec)
        except Exception as exc:  # gated or missing - record and move on
            print(f"skip {name}: {type(exc).__name__}")
            continue
        out[name] = {}
        for lang, rows in data.items():
            words = sum(len(r["l1_native"].split()) for r in rows)
            f = lambda get: sum(count(get(r)) for r in rows) / words
            l3 = sum(f(lambda r, i=i: r["l3_writers"][i]) for i in range(5)) / 5
            out[name][lang] = {"l1": round(f(lambda r: r["l1_native"]), 2),
                               "l2c": round(f(lambda r: r["l2_canonical"]), 2),
                               "l2l": round(f(lambda r: r["l2_lexical"]), 2), "l3": round(l3, 2),
                               "english": round(sum(count(r["english"]) for r in rows) /
                                                sum(len(r["english"].split()) for r in rows), 2)}
        print(name, out[name])
    path = ROOT / "results" / "fertility.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
