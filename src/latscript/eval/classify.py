"""Offensive-language detection on real romanized comments (DravidianCodeMix).

The test comments that were typed in Latin script are scored twice: as typed,
and after IndicXlit back-transliteration into native script. Same comment,
same label, different orthography. Only in-language comments are used and the
task is binary (offensive or not): back-transliterating a comment that is not
in the language produces noise, not a native-script version of it.

Two kinds of classifier, which answer different questions:

* ``finetune`` - an encoder fine-tuned on the cleaned train split (mostly Latin
  script, as the data comes). This is what a practitioner would deploy; it has
  seen romanized text in training and little native script, so it measures
  what back-transliterating *at inference* would do to it, not how much the
  script itself costs.
* ``zeroshot`` - a chat LLM asked whether the comment is offensive, with no
  task training. Here neither orthography is favoured by training data, so the
  typed-vs-native gap is the model's own orthographic handicap. Scored as
  AUROC of P(yes), which is independent of any decision threshold, plus
  macro-F1 at the model's own yes/no choice.

    PYTHONPATH=src python -m latscript.eval.classify finetune google/muril-base-cased
    PYTHONPATH=src python -m latscript.eval.classify zeroshot Qwen/Qwen2.5-7B-Instruct --4bit
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from .conditions import (DERIVED, LANG_NAME, LANGS, load_dcm, read_jsonl, results_dir, slug,
                         write_json)
from .stats import paired


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def offensive_f1(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, pos_label=1, zero_division=0))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def eval_set(lang: str) -> list[dict]:
    """Latin-script, in-language test comments that have a back-transliteration."""
    return [c for c in load_dcm(lang, "test")
            if c["script"] == "latin" and c["in_language"] and c["l1_backtranslit"]]


def mitigated(lang: str, split: str) -> dict[str, dict]:
    """Phase 3 rewrites of DravidianCodeMix comments (bt_text, rt_text) by id."""
    path = DERIVED / "mitigated" / f"dcm_{lang}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run latscript.mitigate.transforms")
    return {r["id"]: r for r in read_jsonl(path)}


def compare(y: np.ndarray, typed: dict, native: dict) -> dict:
    """Paired typed-vs-native comparison for every score both runs produced."""
    out = {}
    for key, fn, field in (("macro_f1", macro_f1, "pred"), ("offensive_f1", offensive_f1, "pred"),
                           ("auroc", auroc, "score")):
        if field in typed:
            a, b = typed[field], native[field]
            out[key] = paired(lambda i: fn(y[i], a[i]), lambda i: fn(y[i], b[i]), len(y))
    return out


# --------------------------------------------------------------------------- fine-tuning

def finetune(name: str, lang: str, seed: int, epochs: int = 3, lr: float = 2e-5,
             batch_size: int = 32, max_len: int = 128, augment: bool = False):
    """Fine-tune on train, pick the epoch by dev macro-F1; return a predict function.

    With `augment` (a Phase 3 mitigation), every Latin-script training comment
    is added a second time in its back-transliterated form, with the same
    label, so the model sees both scripts for the same content.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).cuda()

    def data(split):
        rows = [c for c in load_dcm(lang, split) if c["in_language"]]
        x, y = [c["text"] for c in rows], [int(c["offensive"]) for c in rows]
        if augment and split == "train":
            bt = mitigated(lang, "train")
            extra = [(bt[c["id"]]["bt_text"], int(c["offensive"])) for c in rows
                     if bt[c["id"]]["bt_text"]]
            x += [t for t, _ in extra]
            y += [l for _, l in extra]
        return x, np.array(y)

    train_x, train_y = data("train")
    dev_x, dev_y = data("dev")
    # Malayalam is ~4% offensive; weight the loss so the minority class is learned.
    pos = train_y.mean()
    weight = torch.tensor([0.5 / (1 - pos), 0.5 / pos], dtype=torch.float).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_x) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))

    def predict(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(texts), 128):
                enc = tok(texts[s:s + 128], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to("cuda")
                probs.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        p = np.concatenate(probs)
        return (p > 0.5).astype(int), p

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(train_x))
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            enc = tok([train_x[i] for i in b], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), torch.tensor(train_y[b]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        f1 = macro_f1(dev_y, predict(dev_x)[0])
        print(f"  {lang} seed {seed} epoch {epoch + 1}: dev macro-F1 {f1:.3f}")
        if f1 > best:
            best = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict, best


def run_finetune(name: str, lang: str, seeds: list[int], augment: bool = False,
                 with_rt: bool = False) -> dict:
    """Typed vs back-transliterated test comments; `with_rt` adds typed vs round-tripped."""
    rows = eval_set(lang)
    y = np.array([int(c["offensive"]) for c in rows])
    rt = mitigated(lang, "test") if with_rt else None
    per_seed = []
    for seed in seeds:
        predict, dev_f1 = finetune(name, lang, seed, augment=augment)
        tp, ts = predict([c["text"] for c in rows])
        np_, ns = predict([c["l1_backtranslit"] for c in rows])
        entry = {"seed": seed, "dev_macro_f1": round(dev_f1, 4),
                 **compare(y, {"pred": tp, "score": ts}, {"pred": np_, "score": ns})}
        if rt:
            rp, rs = predict([rt[c["id"]]["rt_text"] for c in rows])
            entry["rt"] = compare(y, {"pred": tp, "score": ts}, {"pred": rp, "score": rs})
        per_seed.append(entry)
        import torch; torch.cuda.empty_cache()
    return {"kind": "finetune", "model": name, "lang": lang, "n": len(y),
            "offensive": int(y.sum()), "augmented": augment,
            "a": "typed (Latin)", "b": "back-transliterated (native)",
            "rt": "round-tripped (Latin, normalized spelling)" if with_rt else None,
            "seeds": per_seed}


# --------------------------------------------------------------------------- zero-shot

ZS_PROMPT = ("Here is a YouTube comment in {language}. It may be written in {language} script "
             "or typed in Latin letters, and may mix in English.\n\nComment: {text}\n\n"
             "Is this comment offensive (insulting, abusive, profane or hateful)? "
             "Answer with one word: Yes or No.")


MAX_COMMENT_TOKENS = 384
TOKEN_BUDGET = 2048


def run_zeroshot(name: str, lang: str, four_bit: bool, batch_size: int = 16,
                 with_rt: bool = False) -> dict:
    import torch
    from .translate import ChatLLM

    llm = ChatLLM(name, batch_size=batch_size, four_bit=four_bit)
    tok = llm.tok
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]

    def prompt(text: str) -> str:
        # Native script costs Qwen ~10 tokens per word, so a few very long
        # comments would dominate memory; keep the first MAX_COMMENT_TOKENS.
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_COMMENT_TOKENS:
            text = tok.decode(ids[:MAX_COMMENT_TOKENS])
        return tok.apply_chat_template(
            [{"role": "user", "content": ZS_PROMPT.format(language=LANG_NAME[lang], text=text)}],
            tokenize=False, add_generation_prompt=True)

    def score(texts: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
        chats = [prompt(t) for t in texts]
        lengths = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in chats]
        # Batch by length, so padding stays small, under a token budget per batch.
        order = sorted(range(len(chats)), key=lambda i: lengths[i])
        batches, cur = [], []
        for i in order:
            if cur and (len(cur) + 1) * lengths[i] > TOKEN_BUDGET or len(cur) == batch_size:
                batches.append(cur); cur = []
            cur.append(i)
        batches.append(cur)
        p = np.zeros(len(chats))
        for b, idx in enumerate(batches):
            enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                # Only the next-token distribution is needed; full-sequence
                # logits would not fit on the GPU next to the 7B model.
                logits = llm.model(**enc, logits_to_keep=1).logits[:, -1].float()
            lp = torch.log_softmax(logits, -1)
            p_yes = torch.logsumexp(lp[:, yes], -1).exp()
            p_no = torch.logsumexp(lp[:, no], -1).exp()
            p[idx] = (p_yes / (p_yes + p_no)).cpu().numpy()
            if b % 20 == 0 or b == len(batches) - 1:
                print(f"\r  {lang} {label}: {b + 1}/{len(batches)} batches", end="", flush=True)
        print()
        return (p > 0.5).astype(int), p

    rows = eval_set(lang)
    y = np.array([int(c["offensive"]) for c in rows])
    tp, ts = score([c["text"] for c in rows], "typed")
    base = {"kind": "zeroshot", "model": name, "lang": lang, "n": len(y),
            "max_comment_tokens": MAX_COMMENT_TOKENS, "offensive": int(y.sum()),
            "a": "typed (Latin)"}
    if with_rt:
        # Phase 3: typed vs round-tripped only; typed vs native is in Phase 2.
        rt = mitigated(lang, "test")
        rp, rs = score([rt[c["id"]]["rt_text"] for c in rows], "round-tripped")
        return {**base, "rt": compare(y, {"pred": tp, "score": ts}, {"pred": rp, "score": rs})}
    np_, ns = score([c["l1_backtranslit"] for c in rows], "native")
    return {**base, "b": "back-transliterated (native)",
            "yes_rate": {"a": round(float(tp.mean()), 4), "b": round(float(np_.mean()), 4)},
            **compare(y, {"pred": tp, "score": ts}, {"pred": np_, "score": ns})}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["finetune", "zeroshot"])
    ap.add_argument("model")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--augment", action="store_true",
                    help="Phase 3: add back-transliterated copies of Latin training comments")
    ap.add_argument("--rt", action="store_true",
                    help="Phase 3: also score round-tripped (spelling-normalized) test comments")
    args = ap.parse_args(argv)
    phase3 = args.augment or args.rt
    suffix = ("_aug" if args.augment else "") + ("_rt" if args.rt else "")

    for lang in args.langs:
        if args.kind == "finetune":
            res = run_finetune(args.model, lang, args.seeds, augment=args.augment, with_rt=args.rt)
            m = res["seeds"]
            line = "  ".join(f"seed {s['seed']}: typed {s['macro_f1']['a']['score']:.3f} "
                             f"native {s['macro_f1']['b']['score']:.3f}"
                             + (f" rt {s['rt']['macro_f1']['b']['score']:.3f}" if "rt" in s else "")
                             for s in m)
        else:
            res = run_zeroshot(args.model, lang, args.four_bit, with_rt=args.rt)
            other, key = (res["rt"], "rt") if args.rt else (res, "native")
            line = (f"AUROC typed {other['auroc']['a']['score']:.3f} "
                    f"{key} {other['auroc']['b']['score']:.3f}  macro-F1 typed "
                    f"{other['macro_f1']['a']['score']:.3f} {key} {other['macro_f1']['b']['score']:.3f}")
        write_json(results_dir("classify", phase3) / f"{args.kind}_{slug(args.model)}_{lang}{suffix}.json",
                   res)
        print(f"{lang} {slug(args.model)} {line}")


if __name__ == "__main__":
    main()
