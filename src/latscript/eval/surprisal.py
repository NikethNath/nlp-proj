"""How surprised a language model is by the same sentence in each orthography.

The content of a FLORES sentence is fixed across conditions, so the total
negative log-likelihood of the whole sentence (in bits, summed over however
many tokens the tokenizer produces) is directly comparable between them,
unlike per-token perplexity. It is reported per native word. A model that
has learned romanized Malayalam well pays few extra bits for L2; one that
has learned a single romanization convention pays extra for L3.

    PYTHONPATH=src python -m latscript.eval.surprisal Qwen/Qwen2.5-1.5B
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from ..translit.tokens import NATIVE_WORD, words
from .conditions import CONDITIONS, LANGS, RESULTS, load_rows, slug, texts, write_json
from .stats import decompose, mean_metric

MODELS = ["Qwen/Qwen2.5-1.5B", "sarvamai/sarvam-1"]


class Scorer:
    def __init__(self, name: str, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda").eval()
        # Condition the first real token on a document boundary.
        self.prefix = self.tok.bos_token_id if self.tok.bos_token_id is not None \
            else self.tok.eos_token_id
        self.batch_size = batch_size

    def bits(self, items: list[str]) -> list[float]:
        torch = self.torch
        out = []
        for start in range(0, len(items), self.batch_size):
            ids = [[self.prefix] + self.tok(t, add_special_tokens=False)["input_ids"]
                   for t in items[start:start + self.batch_size]]
            width = max(map(len, ids))
            x = torch.full((len(ids), width), self.prefix, dtype=torch.long)
            mask = torch.zeros_like(x)
            for i, seq in enumerate(ids):
                x[i, :len(seq)] = torch.tensor(seq)
                mask[i, :len(seq)] = 1
            x, mask = x.cuda(), mask.cuda()
            with torch.inference_mode():
                logits = self.model(input_ids=x, attention_mask=mask).logits
                # One sequence at a time: float logits for a whole batch over a
                # 150k vocabulary do not fit next to the model on an 8 GB GPU.
                for i, seq in enumerate(ids):
                    n = len(seq) - 1
                    nll = torch.nn.functional.cross_entropy(
                        logits[i, :n].float(), x[i, 1:n + 1], reduction="sum")
                    out.append(nll.item() / math.log(2))
            del logits
        return out


def evaluate(scorer: Scorer, name: str, lang: str, rows: list[dict]) -> dict:
    n_words = np.array([max(1, sum(bool(NATIVE_WORD[lang].search(w)) for w in words(r["l1_native"])))
                        for r in rows], dtype=float)
    stats = {}
    for cond in CONDITIONS:
        b = np.array(scorer.bits([texts(r)[cond] for r in rows]))
        stats[cond] = np.stack([b, n_words], axis=-1)
    return {"model": name, "lang": lang, "n": len(rows), "unit": "bits per native word",
            "bits_per_word": decompose(stats, mean_metric, higher_is_better=False)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*", default=MODELS)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--subset", default="all")
    args = ap.parse_args(argv)

    for name in args.models:
        scorer = Scorer(name)
        for lang in args.langs:
            res = evaluate(scorer, name, lang, load_rows(lang, subset=args.subset))
            write_json(RESULTS / "surprisal" / f"{slug(name)}_{lang}.json", res)
            lv = res["bits_per_word"]["levels"]
            print(f"{lang} {slug(name):16s} bits/word  L1 {lv['l1']['score']:.1f}  "
                  f"L2c {lv['l2_canonical']['score']:.1f}  L2l {lv['l2_lexical']['score']:.1f}  "
                  f"L3 {lv['l3']['score']:.1f}")
        del scorer


if __name__ == "__main__":
    main()
