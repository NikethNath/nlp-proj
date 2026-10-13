"""LoRA mitigation rung: adapt NLLB-200 to romanized Malayalam and Tamil.

Training pairs come from FLORES *dev* (disjoint from the devtest sentences we
evaluate on): for each dev sentence, two randomly chosen simulated writers'
L3 and the canonical L2, each paired with the English reference. One adapter
serves both languages. The simulated writers are the same five people in dev
and devtest, so the adapter has seen their habits but never the test sentences;
this is noted as a limitation.

    python -m latscript.eval.train_lora --out data/models/nllb-lora
Then evaluate with run.py --adapter data/models/nllb-lora.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .backends import FLORES_CODE
from .prepare import DERIVED, read_jsonl

ROOT = Path(__file__).resolve().parents[3]


def pairs(lang: str, rng: random.Random) -> list[tuple[str, str, str]]:
    out = []
    for r in read_jsonl(DERIVED / f"flores_{lang}_dev.jsonl"):
        for w in rng.sample(range(len(r["l3_writers"])), 2):
            out.append((lang, r["l3_writers"][w], r["english"]))
        out.append((lang, r["l2_canonical"], r["english"]))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "models" / "nllb-lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model, tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                                             task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()

    rng = random.Random(0)
    data = pairs("ml", rng) + pairs("ta", rng)
    # Batches are single-language so the source language tag is right for every row.
    batches = []
    for lang in ("ml", "ta"):
        rows = [d for d in data if d[0] == lang]
        rng.shuffle(rows)
        batches += [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    rng.shuffle(batches)
    steps = int(len(batches) * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    model.train()
    t0 = time.perf_counter()
    for step in range(steps):
        chunk = batches[step % len(batches)]
        tok.src_lang = FLORES_CODE[chunk[0][0]]
        enc = tok([c[1] for c in chunk], text_target=[c[2] for c in chunk], truncation=True,
                  max_length=128, padding=True, return_tensors="pt")
        enc["labels"][enc["labels"] == tok.pad_token_id] = -100
        loss = model(**enc).loss
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
        if step % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  step {step}/{steps} loss {loss.item():.3f} ({el / max(step, 1):.2f}s/step)",
                  file=sys.stderr, flush=True)
    model.save_pretrained(args.out)
    (args.out / "train.json").write_text(json.dumps({
        "pairs": len(data), "steps": steps, "batch": args.batch, "rank": args.rank,
        "lr": args.lr, "cpu_hours": round((time.perf_counter() - t0) / 3600, 3),
        "threads": args.threads}, indent=1))
    print(args.out)


if __name__ == "__main__":
    main()
