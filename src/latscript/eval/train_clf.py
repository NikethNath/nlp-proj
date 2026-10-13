"""Supervised offensive-language classifiers on DravidianCodeMix.

* ``muril``   google/muril-base-cased fine-tuned on the cleaned training split
              (comments as their authors typed them, mostly romanized). The
              197k-entry embedding matrix is frozen so training fits in a few GB
              of RAM on CPU; every transformer layer is trained.
* ``murilprobe``  frozen MuRIL, mean-pooled last layer, class-balanced
              logistic regression on top (a linear probe). Fine-tuning MuRIL
              for one epoch on CPU did not leave chance level on Malayalam (it
              predicted every comment non-offensive), so the reported MuRIL
              numbers use this probe.
* ``charlr``  character 1-5-gram TF-IDF + logistic regression: a cheap
              baseline that is maximally sensitive to spelling.

Both are then evaluated on every condition of the romanized test comments
(see prepare.py) and written in the same format as run.py, so analyze.py
scores them like the zero-shot LLMs.

    python -m latscript.eval.train_clf --model muril --lang ta --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from .prepare import DERIVED, read_jsonl
from .run import OUTPUTS, dcm_subset

COND_FIELDS = ["l3", "l1", "l2", "cn"] + [f"s3_{i}" for i in range(5)]


def train_rows(lang: str) -> list[dict]:
    return [r for r in read_jsonl(DERIVED / f"dcm_{lang}_train.jsonl") if r["text"].strip()]


def write_outputs(model: str, lang: str, rows: list[dict], preds: dict, seconds: dict) -> None:
    path = OUTPUTS / f"clf_{model}_{lang}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in COND_FIELDS:
            for r, p in zip(rows, preds[c]):
                fh.write(json.dumps({"id": r["id"], "cond": c, "out": str(p), "pred": int(p),
                                     "seconds": seconds[c], "ptok": 0, "ctok": 0}) + "\n")
    print(path)


def run_charlr(lang: str) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train = train_rows(lang)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                          lowercase=True, max_features=300_000)
    X = vec.fit_transform([r["text"] for r in train])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(X, [int(r["offensive"]) for r in train])
    test = dcm_subset(lang, None)
    preds, secs = {}, {}
    for c in COND_FIELDS:
        t0 = time.perf_counter()
        preds[c] = clf.predict(vec.transform([r[c] for r in test]))
        secs[c] = round((time.perf_counter() - t0) / len(test), 5)
    write_outputs("charlr", lang, test, preds, secs)


def run_murilprobe(lang: str, batch: int, max_len: int, threads: int) -> None:
    """Frozen MuRIL + logistic regression; C and the decision threshold are chosen on dev."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(threads)
    name = "google/muril-base-cased"
    tok = AutoTokenizer.from_pretrained(name)
    enc_model = AutoModel.from_pretrained(name).eval()
    cache = DERIVED.parent / "cache" / "muril"
    cache.mkdir(parents=True, exist_ok=True)

    def embed(texts, key=None):
        path = cache / f"{key}.npy" if key else None
        if path and path.exists():
            return np.load(path)
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), batch):
                enc = tok(texts[s:s + batch], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt")
                h = enc_model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1)).numpy())
        X = np.concatenate(out)
        if path:
            np.save(path, X)
        return X

    t0 = time.perf_counter()
    train = train_rows(lang)
    dev = [r for r in read_jsonl(DERIVED / f"dcm_{lang}_dev.jsonl") if r["text"].strip()]
    Xtr = embed([r["text"] for r in train], f"{lang}_train")
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xdv = scaler.transform(Xtr), scaler.transform(embed([r["text"] for r in dev], f"{lang}_dev"))
    ytr = np.array([int(r["offensive"]) for r in train])
    ydv = np.array([int(r["offensive"]) for r in dev])
    best = (-1.0, None, None)
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Xtr, ytr)
        p = clf.predict_proba(Xdv)[:, 1]
        for th in np.arange(0.1, 0.95, 0.05):
            f = f1_score(ydv, p >= th, average="macro")
            if f > best[0]:
                best = (f, clf, float(th))
    dev_f1, clf, th = best
    train_hours = (time.perf_counter() - t0) / 3600
    print(f"  dev macro-F1 {dev_f1:.3f} at C={clf.C}, threshold {th:.2f}", file=sys.stderr, flush=True)
    test = dcm_subset(lang, None)
    preds, secs = {}, {}
    for c in COND_FIELDS:
        t1 = time.perf_counter()
        preds[c] = (clf.predict_proba(scaler.transform(embed([r[c] for r in test], f"{lang}_test_{c}")))[:, 1] >= th).astype(int)
        secs[c] = round((time.perf_counter() - t1) / len(test), 5)
        print(f"  {c}: positive rate {np.mean(preds[c]):.3f}", file=sys.stderr, flush=True)
    write_outputs("muril", lang, test, preds, secs)
    (OUTPUTS / f"clf_muril_{lang}.train.json").write_text(json.dumps({
        "method": "linear probe: frozen MuRIL mean-pooled, standardized, class-balanced logistic regression; "
                  "C and decision threshold chosen on the dev split by macro F1",
        "C": clf.C, "threshold": round(th, 2), "dev_macro_f1": round(dev_f1, 4),
        "train_comments": len(train), "dev_comments": len(dev), "max_len": max_len,
        "cpu_hours": round(train_hours, 3), "threads": threads}, indent=1))


def run_muril(lang: str, epochs: float, batch: int, max_len: int, lr: float, threads: int) -> None:
    import torch
    from torch import nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    name = "google/muril-base-cased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)

    train = train_rows(lang)
    random.Random(0).shuffle(train)
    y = np.array([int(r["offensive"]) for r in train])
    weights = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    steps = int(len(train) / batch * epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    t0 = time.perf_counter()
    for step in range(steps):
        start = (step * batch) % len(train)
        chunk = train[start:start + batch]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        logits = model(**enc).logits
        loss = loss_fn(logits, torch.tensor([int(r["offensive"]) for r in chunk]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  step {step}/{steps} loss {loss.item():.3f} ({el / max(step, 1):.2f}s/step)",
                  file=sys.stderr, flush=True)
    train_hours = (time.perf_counter() - t0) / 3600

    model.eval()
    test = dcm_subset(lang, None)
    preds, secs = {}, {}
    with torch.inference_mode():
        for c in COND_FIELDS:
            out, t0 = [], time.perf_counter()
            for s in range(0, len(test), 64):
                enc = tok([r[c] for r in test[s:s + 64]], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt")
                out += model(**enc).logits.argmax(-1).tolist()
            preds[c] = out
            secs[c] = round((time.perf_counter() - t0) / len(test), 5)
            print(f"  {c}: positive rate {np.mean(out):.3f}", file=sys.stderr, flush=True)
    write_outputs("muril", lang, test, preds, secs)
    meta = OUTPUTS / f"clf_muril_{lang}.train.json"
    meta.write_text(json.dumps({"train_comments": len(train), "epochs": epochs, "steps": steps,
                                "batch": batch, "max_len": max_len, "lr": lr,
                                "cpu_hours": round(train_hours, 3), "threads": threads}, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["muril", "murilprobe", "charlr"], required=True)
    ap.add_argument("--lang", choices=["ml", "ta"], required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    args = ap.parse_args(argv)
    out_name = "muril" if args.model == "murilprobe" else args.model
    if (OUTPUTS / f"clf_{out_name}_{args.lang}.jsonl").exists() and not args.force:
        print(f"clf_{out_name}_{args.lang}.jsonl exists; skipping (use --force)")
        return
    if args.model == "charlr":
        run_charlr(args.lang)
    elif args.model == "murilprobe":
        run_murilprobe(args.lang, 64, args.max_len, args.threads)
    else:
        run_muril(args.lang, args.epochs, args.batch, args.max_len, args.lr, args.threads)


if __name__ == "__main__":
    main()
