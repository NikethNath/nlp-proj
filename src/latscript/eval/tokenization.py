"""How many subword tokens each orthography costs.

Fertility is tokens per native word. Words align one-to-one across the
conditions of a FLORES sentence, so this is a like-for-like count of what
the same content costs in each script. Lower is better; the gaps are reported
as extra tokens per word. Tokenizers that have an unknown token also report
the share of words that contain one.

    PYTHONPATH=src python -m latscript.eval.tokenization
"""
from __future__ import annotations

import argparse

import numpy as np

from ..translit.tokens import NATIVE_WORD, words
from .conditions import CONDITIONS, LANGS, RESULTS, load_rows, slug, texts, write_json
from .stats import as_mean_stats, decompose, mean_metric

TOKENIZERS = [
    "facebook/nllb-200-distilled-600M",
    "FacebookAI/xlm-roberta-base",
    "google/muril-base-cased",
    "ai4bharat/IndicBERTv2-MLM-only",
    "sentence-transformers/LaBSE",
    "Qwen/Qwen2.5-7B-Instruct",
    "sarvamai/sarvam-1",
    "tiktoken:o200k_base",
]


class Counter:
    def __init__(self, name: str):
        self.name = name
        if name.startswith("tiktoken:"):
            import tiktoken
            enc = tiktoken.get_encoding(name.split(":", 1)[1])
            self._encode = enc.encode_ordinary
            self.unk = None
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
            self._encode = lambda t: tok(t, add_special_tokens=False)["input_ids"]
            self.unk = tok.unk_token_id

    def count(self, text: str) -> tuple[int, int]:
        """(tokens, words containing an unknown token)."""
        ids = self._encode(text)
        if self.unk is None:
            return len(ids), 0
        unk_words = sum(self.unk in self._encode(w) for w in words(text))
        return len(ids), unk_words


def evaluate(name: str, lang: str, rows: list[dict]) -> dict:
    counter = Counter(name)
    n_words = np.array([max(1, sum(bool(NATIVE_WORD[lang].search(w)) for w in words(r["l1_native"])))
                        for r in rows], dtype=float)
    fert, unk = {}, {}
    for cond in CONDITIONS:
        counts = [counter.count(texts(r)[cond]) for r in rows]
        tok = np.array([c[0] for c in counts], dtype=float)
        # Corpus fertility: total tokens over total words, as a ratio of sums.
        fert[cond] = np.stack([tok, n_words], axis=-1)
        unk[cond] = np.stack([np.array([c[1] for c in counts], float), n_words], axis=-1)
    out = {"tokenizer": name, "lang": lang,
           "fertility": decompose(fert, mean_metric, higher_is_better=False)}
    if counter.unk is not None:
        out["unk_word_share"] = decompose(unk, mean_metric, higher_is_better=False)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--tokenizers", nargs="+", default=TOKENIZERS)
    ap.add_argument("--subset", default="all")
    args = ap.parse_args(argv)

    for lang in args.langs:
        rows = load_rows(lang, subset=args.subset)
        for name in args.tokenizers:
            res = evaluate(name, lang, rows)
            write_json(RESULTS / "tokenization" / f"{slug(name)}_{lang}.json", res)
            lv = res["fertility"]["levels"]
            print(f"{lang} {slug(name):32s} tokens/word  L1 {lv['l1']['score']:.2f}  "
                  f"L2c {lv['l2_canonical']['score']:.2f}  L2l {lv['l2_lexical']['score']:.2f}  "
                  f"L3 {lv['l3']['score']:.2f}")


if __name__ == "__main__":
    main()
