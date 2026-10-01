"""Training-side mitigation: fine-tune on romanized text from FLORES dev.

Phase 2 found that models trained on native script lose most of their
competence on romanized input. The direct fix is to show them romanized text.
FLORES dev (997 sentences per language) was held out of Phase 2 for this.

Each dev sentence gives these training pairs with its English reference:
native script, both L2 romanizations, and simulated writers 0-2. Writers 3
and 4 are never trained on. Every writer keeps the same habits in dev and
devtest, so devtest L3 splits into *seen* writers (0-2), whose habits the
model has met, and *unseen* writers (3-4), which test generalization to new
spelling habits. Native script stays in training so the model can be checked
for forgetting it.

Both languages are trained into one model. Word embeddings are frozen: they
hold most of the parameters of these large-vocabulary models, and freezing
them fits training into 8 GB.

    PYTHONPATH=src python -m latscript.mitigate.train retrieval   # LaBSE
    PYTHONPATH=src python -m latscript.mitigate.train mt          # NLLB-200 600M

Models are saved under data/models/ and then evaluated on devtest with the
Phase 2 evaluation code (see scripts/run_phase3.sh).
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from ..eval.conditions import LANGS, ROOT, load_rows, texts
from ..eval.translate import NLLB_CODE

MODELS = ROOT / "data" / "models"
TRAIN_CONDITIONS = ("l1", "l2_canonical", "l2_lexical", "l3_0", "l3_1", "l3_2")
SEEN_WRITERS, UNSEEN_WRITERS = (0, 1, 2), (3, 4)


def pairs(langs=LANGS) -> list[tuple[str, str, str]]:
    """(lang, source, English) for every dev sentence and training condition."""
    out = []
    for lang in langs:
        for r in load_rows(lang, split="dev"):
            t = texts(r)
            out += [(lang, t[c], r["english"]) for c in TRAIN_CONDITIONS]
    return out


def train_retrieval(base: str = "sentence-transformers/LaBSE", epochs: int = 1,
                    batch_size: int = 32, lr: float = 2e-5, seed: int = 0) -> str:
    """Contrastive fine-tuning with in-batch negatives; returns the saved path."""
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    random.seed(seed); torch.manual_seed(seed)
    model = SentenceTransformer(base, device="cuda")
    for p in model[0].auto_model.embeddings.word_embeddings.parameters():
        p.requires_grad = False
    data = Dataset.from_dict({"anchor": [s for _, s, _ in pairs()],
                              "positive": [e for _, _, e in pairs()]})
    out = MODELS / "LaBSE-romanized"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr, warmup_ratio=0.1,
        bf16=True, seed=seed, save_strategy="no", logging_steps=50, report_to=[],
        # Several renderings of one sentence share an English positive; keep
        # them out of the same batch so none is used as another's negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(str(out))
    return str(out)


def train_mt(base: str = "facebook/nllb-200-distilled-600M", epochs: int = 2,
             batch_size: int = 16, lr: float = 5e-5, seed: int = 0, max_len: int = 128) -> str:
    """Seq2seq fine-tuning into English; returns the saved path."""
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base).cuda()
    model.get_input_embeddings().weight.requires_grad = False   # shared with the output layer
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01)

    data = pairs()
    steps = epochs * ((len(data) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    tok.tgt_lang = "eng_Latn"
    step = 0
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(data))
        for start in range(0, len(order), batch_size):
            batch = [data[i] for i in order[start:start + batch_size]]
            # NLLB takes one source language per call; batches mix both, so
            # tokenize each language's items with its own tag and pad together.
            ids, labels = [], []
            for lang, src, en in batch:
                tok.src_lang = NLLB_CODE[lang]
                enc = tok(src, text_target=en, truncation=True, max_length=max_len)
                ids.append(enc["input_ids"]); labels.append(enc["labels"])
            x = tok.pad({"input_ids": ids}, return_tensors="pt").to("cuda")
            width = max(map(len, labels))
            y = torch.tensor([l + [-100] * (width - len(l)) for l in labels]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f}", flush=True)
    out = MODELS / "nllb-200-distilled-600M-romanized"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return str(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=["retrieval", "mt"])
    args = ap.parse_args(argv)
    path = train_retrieval() if args.task == "retrieval" else train_mt()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
