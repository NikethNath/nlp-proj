"""Run one model on one task and language, over a list of orthographic conditions.

    python -m latscript.eval.run --task mt  --model nllb-600m --lang ml --n 400 --conds l1 l2c l2l l3 bx cn
    python -m latscript.eval.run --task mt  --model qwen3-4b --chat qwen --n 120 --writers 3 --conds l1 l2c l3 bx cn fs
    python -m latscript.eval.run --task clf --model qwen3-4b --chat qwen --conds l3 l1 l2 cn s3 te

Condition codes (see prepare.py): a code with per-writer variants (l3, bx, cn,
fs for MT; s3 for classification) expands to one input per writer. ``fs`` is
L3 with four romanized exemplars in the prompt; ``te`` translates the comment
to English and classifies the translation (two calls).

Outputs append to results/outputs/{task}_{model}_{lang}.jsonl, so an
interrupted run resumes. Subsets are prefixes of a fixed permutation, so a
larger --n is always a superset of a smaller one.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from .backends import NLLB, Chat
from .prepare import OUT as EVAL, DERIVED, flores_order, read_jsonl

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "results" / "outputs"
LANG = {"ml": "Malayalam", "ta": "Tamil"}
PER_WRITER = {"mt": {"l3", "bx", "cn", "fs"}, "clf": {"s3"}}
SOURCE = {"fs": "l3"}

MT_SYSTEM = ("You are a translator. Translate the user's {lang} text into English. The text "
             "may be written in {lang} script or typed in English letters. Reply with the "
             "English translation only.")
CLF_PROMPT = ("Here is a YouTube comment in {lang}. It may be typed in English letters and "
              "mixed with English.\n\n{text}\n\nIs this comment offensive (insulting, abusive, "
              "profane, or attacking a person or group)? Answer with one word: yes or no.")


def flores_subset(lang: str, n: int | None) -> list[dict]:
    rows = read_jsonl(EVAL / f"flores_{lang}.jsonl")
    order = flores_order(len(rows))
    return [rows[i] for i in order[:n or len(rows)]]


def dcm_subset(lang: str, n_pos: int | None, ratio: int = 3) -> list[dict]:
    """All positives up to n_pos, plus `ratio` negatives per positive (seeded)."""
    rows = read_jsonl(EVAL / f"dcm_{lang}.jsonl")
    if n_pos is None:
        return rows
    rng = random.Random(2026)
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    rng.shuffle(pos), rng.shuffle(neg)
    pos = pos[:n_pos]
    return sorted(pos + neg[:ratio * len(pos)], key=lambda r: r["id"])


def fewshot_exemplars(lang: str, k: int = 4) -> list[tuple[str, str]]:
    """Four short FLORES *dev* sentences in simulated L3, each by a different writer."""
    dev = read_jsonl(DERIVED / f"flores_{lang}_dev.jsonl")
    short = [r for r in dev if 8 <= len(r["l1_native"].split()) <= 14]
    picks = random.Random(7).sample(short, k)
    return [(r["l3_writers"][i % len(r["l3_writers"])], r["english"]) for i, r in enumerate(picks)]


def expand(task: str, conds: list[str], writers: int) -> list[str]:
    out = []
    for c in conds:
        out += [f"{c}_{i}" for i in range(writers)] if c in PER_WRITER[task] else [c]
    return out


def source_field(cond: str) -> str:
    cond = cond.removeprefix("lora_")
    base, _, i = cond.partition("_")
    base = SOURCE.get(base, base)
    return f"{base}_{i}" if i else base


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(r["id"], r["cond"]) for r in read_jsonl(path)}


def parse_yes_no(text: str) -> int | None:
    m = re.search(r"\b(yes|no)\b", text.lower())
    return None if not m else int(m.group(1) == "yes")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=["mt", "clf"], required=True)
    ap.add_argument("--model", required=True, help="name used in output files")
    ap.add_argument("--lang", choices=["ml", "ta"], required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--writers", type=int, default=5)
    ap.add_argument("--n", type=int, help="MT: sentences; clf: positives (3 negatives each)")
    ap.add_argument("--chat", help="model name served at --base-url (omit for NLLB)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    ap.add_argument("--api-key-env")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--adapter", help="LoRA adapter directory for NLLB")
    args = ap.parse_args(argv)

    rows = flores_subset(args.lang, args.n) if args.task == "mt" else dcm_subset(args.lang, args.n)
    conds = expand(args.task, args.conds, args.writers)
    if args.adapter:
        # Adapter outputs share the base model's file under a lora_ prefix.
        conds = [f"lora_{c}" for c in conds]
    path = OUTPUTS / f"{args.task}_{args.model}_{args.lang}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path)
    todo = [(r, c) for c in conds for r in rows if (r["id"], c) not in done]
    print(f"{path.name}: {len(rows)} items x {len(conds)} conditions, {len(todo)} to run",
          file=sys.stderr, flush=True)
    if not todo:
        return
    lang = LANG[args.lang]
    out = path.open("a", encoding="utf-8")

    def write(r, c, res, **extra):
        out.write(json.dumps({"id": r["id"], "cond": c, "out": res.text,
                              "seconds": round(res.seconds, 4), "ptok": res.prompt_tokens,
                              "ctok": res.completion_tokens, **extra}, ensure_ascii=False) + "\n")
        out.flush()

    if args.chat is None:
        if args.task != "mt":
            raise SystemExit("NLLB only translates")
        model = NLLB(args.nllb, adapter=args.adapter)
        for c in conds:
            # Length-sorted batches: one long sentence no longer stalls fifteen short ones.
            batch = sorted((r for r, cc in todo if cc == c), key=lambda r: len(r[source_field(c)]))
            if batch:
                for r, res in zip(batch, model.translate([r[source_field(c)] for r in batch], args.lang)):
                    write(r, c, res)
                print(f"  {c}: {len(batch)}", file=sys.stderr, flush=True)
        return

    chat = Chat(args.chat, args.base_url, args.api_key_env, args.parallel)
    shots = fewshot_exemplars(args.lang) if args.task == "mt" else []

    def mt_messages(text: str, fewshot: bool) -> list[dict]:
        msgs = [{"role": "system", "content": MT_SYSTEM.format(lang=lang)}]
        if fewshot:
            for src, tgt in shots:
                msgs += [{"role": "user", "content": src}, {"role": "assistant", "content": tgt}]
        return msgs + [{"role": "user", "content": text}]

    def clf_messages(text: str) -> list[dict]:
        return [{"role": "user", "content": CLF_PROMPT.format(lang=lang, text=text)}]

    if args.task == "clf" and any(c == "te" for _, c in todo):
        # Translate-to-English first: one MT call, then classify the English.
        te = [(r, c) for r, c in todo if c == "te"]
        jobs = [mt_messages(r["l3"], False) for r, _ in te]
        trans = dict(chat.map(jobs, 160))
        jobs2 = [[{"role": "user", "content": CLF_PROMPT.format(lang="English", text=trans[i].text)}]
                 for i in range(len(te))]
        for i, res in chat.map(jobs2, 4):
            r, c = te[i]
            t = trans[i]
            res.seconds += t.seconds
            res.prompt_tokens += t.prompt_tokens
            res.completion_tokens += t.completion_tokens
            write(r, c, res, pred=parse_yes_no(res.text), english=t.text)
        todo = [(r, c) for r, c in todo if c != "te"]

    if args.task == "mt":
        jobs = [mt_messages(r[source_field(c)], c.startswith("fs_")) for r, c in todo]
        max_tokens = 200
    else:
        jobs = [clf_messages(r[source_field(c)]) for r, c in todo]
        max_tokens = 4
    for n, (i, res) in enumerate(chat.map(jobs, max_tokens), 1):
        r, c = todo[i]
        extra = {"pred": parse_yes_no(res.text)} if args.task == "clf" else {}
        write(r, c, res, **extra)
        if n % 50 == 0:
            print(f"  {n}/{len(todo)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
