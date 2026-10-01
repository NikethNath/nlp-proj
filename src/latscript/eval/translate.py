"""Translation into English under each orthographic condition.

The FLORES English reference is the same for all eight inputs of a sentence,
so any change in chrF++ is caused by orthography alone.

Two kinds of model:

* ``nllb`` - NLLB-200 encoder-decoders. NLLB has no romanized Malayalam or
  Tamil language code, so every condition is tagged ``mal_Mlym``/``tam_Taml``,
  which is what a user of the model would do.
* ``llm`` - a chat model given one fixed prompt that names the language and
  says the text may be in native script or Latin letters. The prompt never
  says which, so the model gets identical instructions under every condition.

Hypotheses are cached in ``data/eval/mt`` and a run resumes where it stopped.

    PYTHONPATH=src python -m latscript.eval.translate nllb facebook/nllb-200-distilled-600M
    PYTHONPATH=src python -m latscript.eval.translate llm Qwen/Qwen2.5-7B-Instruct --subset core --4bit
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from .conditions import (CONDITIONS, EVAL_OUT, LANG_NAME, LANGS, VARIANTS, append_jsonl,
                         load_rows, read_jsonl, results_dir, slug, tag, texts, write_json)
from ..translit.distance import normalized_edit_distance
from .stats import as_mean_stats, decompose, level_scores, mean_metric

NLLB_CODE = {"ml": "mal_Mlym", "ta": "tam_Taml"}
SCRIPT_NAME = {"ml": "Malayalam", "ta": "Tamil"}

PROMPT = ("Translate the following {language} text into English. It may be written in "
          "{script} script or typed in Latin letters. Reply with the English translation only.\n\n"
          "{text}")


# --------------------------------------------------------------------------- scoring

def chrf_stats(hyps: list[str], refs: list[str]) -> np.ndarray:
    """Per-sentence chrF++ sufficient statistics, shaped (n, k)."""
    from sacrebleu.metrics import CHRF
    metric = CHRF(word_order=2)
    return np.array(metric._extract_corpus_statistics(hyps, [refs]), dtype=float)


def chrf_metric(summed: np.ndarray) -> float:
    from sacrebleu.metrics import CHRF
    return float(CHRF(word_order=2)._compute_score_from_stats(list(summed)).score)


# --------------------------------------------------------------------------- models

class NLLB:
    def __init__(self, name: str, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name, dtype=torch.float16).to("cuda").eval()
        self.batch_size = batch_size

    def translate(self, items: list[str], lang: str) -> list[str]:
        self.tok.src_lang = NLLB_CODE[lang]
        enc = self.tok(items, return_tensors="pt", padding=True, truncation=True,
                       max_length=256).to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc, forced_bos_token_id=self.tok.convert_tokens_to_ids("eng_Latn"),
                num_beams=4, max_new_tokens=256)
        return self.tok.batch_decode(out, skip_special_tokens=True)


class ChatLLM:
    def __init__(self, name: str, batch_size: int = 16, four_bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16) if four_bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, quantization_config=quant, device_map="cuda").eval()
        self.batch_size = batch_size

    def generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        chats = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        enc = self.tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def translate(self, items: list[str], lang: str) -> list[str]:
        prompts = [PROMPT.format(language=LANG_NAME[lang], script=SCRIPT_NAME[lang], text=t)
                   for t in items]
        return [clean(o) for o in self.generate(prompts)]


def clean(output: str) -> str:
    """First non-empty line, without a 'Translation:' label or wrapping quotes."""
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    text = lines[0] if lines else ""
    text = re.sub(r"^(english\s+)?translation\s*:\s*", "", text, flags=re.I)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.strip()


# --------------------------------------------------------------------------- run

def run(model, model_name: str, lang: str, rows: list[dict],
        variant: str | None = None) -> dict[tuple[str, str], str]:
    """Translate every input not already cached; return (id, source) -> hypothesis.

    The cache is keyed by source text, so Phase 3 variants reuse the L1
    translations and any rewritten input that happens to equal one already done.
    """
    path = EVAL_OUT / "mt" / f"{slug(model_name)}_{lang}.jsonl"
    done = {(h["id"], h["source"]) for h in read_jsonl(path)}
    todo, seen = [], set()
    for r in rows:
        for cond, text in texts(r, variant).items():
            if (r["id"], text) not in done and (r["id"], text) not in seen:
                seen.add((r["id"], text))
                todo.append((r["id"], cond, text))
    # Longest first, so the batch that runs out of memory is the first one.
    todo.sort(key=lambda x: -len(x[2]))
    for start in range(0, len(todo), model.batch_size):
        batch = todo[start:start + model.batch_size]
        hyps = model.translate([t for _, _, t in batch], lang)
        append_jsonl(path, ({"id": i, "condition": c, "source": t, "hyp": h}
                            for (i, c, t), h in zip(batch, hyps)))
        print(f"\r  {lang}: {min(start + model.batch_size, len(todo))}/{len(todo)}", end="", flush=True)
    if todo:
        print()
    return {(h["id"], h["source"]): h["hyp"] for h in read_jsonl(path)}


def is_copy(source: str, hyp: str, threshold: float = 0.3) -> bool:
    """The model echoed its (romanized) input instead of translating it."""
    a, b = source.lower().strip(), hyp.lower().strip()
    return bool(a) and normalized_edit_distance(a, b) < threshold


def score(hyps: dict[tuple[str, str], str], rows: list[dict], variant: str | None = None) -> dict:
    src = [texts(r, variant) for r in rows]
    hyp = {cond: [hyps[(r["id"], s[cond])] for r, s in zip(rows, src)] for cond in CONDITIONS}
    refs = [r["english"] for r in rows]
    stats = {cond: chrf_stats(hyp[cond], refs) for cond in CONDITIONS}
    copies = {cond: as_mean_stats([is_copy(s[cond], h) for s, h in zip(src, hyp[cond])])
              for cond in CONDITIONS}
    copy_rate = level_scores({c: v.sum(0) for c, v in copies.items()}, mean_metric)
    return {**decompose(stats, chrf_metric),
            "copy_rate": {k: round(v, 4) for k, v in copy_rate.items() if k != "l3_writers"}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["nllb", "llm"])
    ap.add_argument("model")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--subset", default="all", help="'all', 'core' or a sentence count")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--variant", choices=VARIANTS, help="Phase 3 rewritten inputs")
    ap.add_argument("--phase3", action="store_true",
                    help="write to results/mitigate (e.g. for a fine-tuned model)")
    args = ap.parse_args(argv)

    if args.kind == "nllb":
        model = NLLB(args.model, batch_size=args.batch_size or 32)
    else:
        model = ChatLLM(args.model, batch_size=args.batch_size or 16, four_bit=args.four_bit)

    for lang in args.langs:
        rows = load_rows(lang, subset=args.subset, variant=args.variant)
        hyps = run(model, args.model, lang, rows, args.variant)
        res = {"model": args.model, "lang": lang, "subset": args.subset, "metric": "chrF++",
               "variant": args.variant, **score(hyps, rows, args.variant)}
        write_json(results_dir("mt", args.phase3 or bool(args.variant)) /
                   f"{slug(args.model)}_{lang}_{args.subset}{tag(args.variant)}.json",
                   res)
        lv, gp = res["levels"], res["gaps"]
        print(f"{lang} {slug(args.model)} chrF++  L1 {lv['l1']['score']:.1f}  "
              f"L2c {lv['l2_canonical']['score']:.1f}  L2l {lv['l2_lexical']['score']:.1f}  "
              f"L3 {lv['l3']['score']:.1f}  | script(c) {gp['script_canonical']['loss']:.1f}  "
              f"spelling(c) {gp['spelling_canonical']['loss']:.1f}")


if __name__ == "__main__":
    main()
