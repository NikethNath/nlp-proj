"""Score every output file and write results/summary.json and results/README.md.

Translation is scored with chrF++ against the FLORES English reference;
classification with macro and weighted F1 on the offensive / not-offensive
decision. Per-sentence (or per-comment) scores feed a paired bootstrap, which
gives a 95% interval and a two-sided p-value for each term of the
decomposition

    total    = L1 - L3      (native script vs. naturally spelled romanization)
    script   = L1 - L2      (change of script, spelling held fixed)
    spelling = L2 - L3      (spelling variance, script held fixed)

L3 is averaged over writers before differencing, so each item counts once.
Orthographic consistency is computed across the writer variants of the same
item: mean pairwise chrF++ between translations, and the prediction flip rate
for classification.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

from .prepare import OUT as EVAL, read_jsonl
from .run import OUTPUTS

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
CHRF_PP = CHRF(word_order=2)
B = 10_000
MT_GROUPS = ["l1", "l2c", "l2l", "l3", "bx", "cn", "fs", "lora_l1", "lora_l2c", "lora_l3"]
CLF_GROUPS = ["l1", "l2", "l3", "cn", "s3", "te"]


_PREFACE = re.compile(r"^.*\b(translation|translates?|means?)\b[^\n]*:\s*$", re.I | re.M)
_NOTE = re.compile(r"\n\s*\n\s*[*(]*\s*(note|explanation|literal|this translation)\b.*", re.I | re.S)


def clean(text: str) -> str:
    """Strip chat framing ("Here's the translation:", trailing notes, quotes, bold).

    Applied identically to every model's output before scoring; NLLB output is
    unaffected.
    """
    last = None
    for m in _PREFACE.finditer(text):
        last = m
    if last and text[last.end():].strip():
        text = text[last.end():]
    text = _NOTE.sub("", text).strip()
    text = text.replace("**", "").strip()
    if len(text) > 1 and text[0] in "\"“'" and text[-1] in "\"”'":
        text = text[1:-1].strip()
    return text


def group_of(cond: str) -> str:
    """'l3_2' -> 'l3', 'lora_l3_1' -> 'lora_l3', 'l2c' -> 'l2c'."""
    head, _, tail = cond.rpartition("_")
    return head if tail.isdigit() else cond


def bootstrap(a: np.ndarray, b: np.ndarray, stat=np.mean, seed: int = 0) -> dict:
    """Paired bootstrap of stat(a) - stat(b) over items."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    if stat is np.mean:
        diffs = a[idx].mean(1) - b[idx].mean(1)
    else:
        diffs = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    d = float(stat(a) - stat(b))
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(d, 2), "ci": [round(float(lo), 2), round(float(hi), 2)],
            "p": round(float(p), 4)}


def decompose(scores: dict[str, np.ndarray], l2: str, stat=np.mean) -> dict:
    l1, mid, l3 = scores["l1"], scores[l2], scores["l3"]
    out = {"total": bootstrap(l1, l3, stat), "script": bootstrap(l1, mid, stat),
           "spelling": bootstrap(mid, l3, stat)}
    t = out["total"]["diff"]
    out["spelling_share"] = round(out["spelling"]["diff"] / t, 3) if abs(t) > 1e-9 else None
    return out


# ------------------------------------------------------------------ translation
def analyze_mt(path: Path, xlit_cost: dict) -> dict:
    lang = path.stem.rsplit("_", 1)[1]
    refs = {r["id"]: r["ref"] for r in read_jsonl(EVAL / f"flores_{lang}.jsonl")}
    outs = read_jsonl(path)
    by = defaultdict(dict)                       # cond -> id -> row
    for r in outs:
        r["out"] = clean(r["out"])
        by[r["cond"]][r["id"]] = r
    groups = defaultdict(list)                   # group -> [cond]
    for c in by:
        groups[group_of(c)].append(c)
    # Items present in every base condition, so all comparisons are paired.
    base = [g for g in ("l1", "l2c", "l3") if g in groups]
    ids = sorted(set.intersection(*(set(by[c]) for g in base for c in groups[g])))
    if not ids:
        return {}

    def sent_scores(conds: list[str]) -> np.ndarray:
        per = [[CHRF_PP.sentence_score(by[c][i]["out"], [refs[i]]).score for i in ids] for c in conds]
        return np.mean(np.array(per), axis=0)

    def corpus(conds: list[str]) -> float:
        return float(np.mean([CHRF_PP.corpus_score([by[c][i]["out"] for i in ids],
                                                   [[refs[i] for i in ids]]).score for c in conds]))

    res = {"n": len(ids), "conditions": {}, "consistency": {}}
    sent = {}
    for g in MT_GROUPS:
        conds = sorted(c for c in groups.get(g, []) if all(i in by[c] for i in ids))
        if not conds:
            continue
        sent[g] = sent_scores(conds)
        rows = [by[c][i] for c in conds for i in ids]
        cost = {"seconds": round(float(np.mean([r["seconds"] for r in rows])), 3),
                "prompt_tokens": round(float(np.mean([r["ptok"] for r in rows])), 1),
                "output_tokens": round(float(np.mean([r["ctok"] for r in rows])), 1)}
        if g == "bx" and xlit_cost.get(lang, {}).get("en2indic_words_per_s"):
            c = xlit_cost[lang]
            cost["xlit_seconds_cold"] = round(c["latin_words_per_sentence"] / c["en2indic_words_per_s"], 3)
        res["conditions"][g] = {"chrf": round(corpus(conds), 2),
                                "sent_chrf": round(float(sent[g].mean()), 2),
                                "writers": len(conds), **cost}
        if len(conds) > 1:
            pair = [np.mean([(CHRF_PP.sentence_score(by[a][i]["out"], [by[b][i]["out"]]).score +
                              CHRF_PP.sentence_score(by[b][i]["out"], [by[a][i]["out"]]).score) / 2
                             for a, b in combinations(conds, 2)]) for i in ids]
            ident = [np.mean([by[a][i]["out"] == by[b][i]["out"] for a, b in combinations(conds, 2)])
                     for i in ids]
            res["consistency"][g] = {"pairwise_chrf": round(float(np.mean(pair)), 2),
                                     "identical_output_rate": round(float(np.mean(ident)), 3)}
    if not all(g in sent for g in ("l1", "l2c", "l3")):
        return {}                                # a run still in progress
    res["decomposition"] = {"canonical": decompose(sent, "l2c")}
    if "l2l" in sent:
        res["decomposition"]["lexical"] = decompose(sent, "l2l")
    # Each rung is compared with L3 from the same writers it was applied to.
    def l3_for(g):
        writers = [c.rsplit("_", 1)[1] for c in groups[g] if all(i in by[c] for i in ids)]
        return sent_scores([f"l3_{w}" for w in writers if f"l3_{w}" in by])

    res["mitigations"] = {g: bootstrap(sent[g], l3_for(g)) for g in ("bx", "cn", "fs", "lora_l3")
                          if g in sent}
    res["ceiling_gap"] = {g: bootstrap(sent["l1"], sent[g]) for g in ("bx", "cn", "fs", "lora_l3")
                          if g in sent}
    return res


# ------------------------------------------------------------------ classification
def f1(y: np.ndarray, p: np.ndarray, average: str = "macro") -> float:
    scores, support = [], []
    for k in (0, 1):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        support.append(np.sum(y == k))
    if average == "macro":
        return 100 * float(np.mean(scores))
    return 100 * float(np.average(scores, weights=support))


def analyze_clf(path: Path) -> dict:
    lang = path.stem.rsplit("_", 1)[1]
    labels = {r["id"]: r["label"] for r in read_jsonl(EVAL / f"dcm_{lang}.jsonl")}
    by = defaultdict(dict)
    for r in read_jsonl(path):
        by[r["cond"]][r["id"]] = r
    groups = defaultdict(list)
    for c in by:
        groups[group_of(c)].append(c)
    ids = sorted(set(by.get("l3", {})) & set(by.get("l1", {})) & set(by.get("l2", {})))
    if len(ids) < len(by.get("l3", {})) or not ids:
        return {}                                # a run still in progress
    if not ids:
        return {}
    y = np.array([labels[i] for i in ids])
    res = {"n": len(ids), "positives": int(y.sum()), "conditions": {}, "consistency": {}}
    preds = {}
    for g in CLF_GROUPS:
        conds = sorted(c for c in groups.get(g, []) if all(i in by[c] for i in ids))
        if not conds:
            continue
        raw = np.array([[by[c][i].get("pred") for i in ids] for c in conds], dtype=object)
        invalid = float(np.mean(raw == None))  # noqa: E711
        P = np.where(raw == None, 0, raw).astype(int)  # noqa: E711
        preds[g] = P
        rows = [by[c][i] for c in conds for i in ids]
        res["conditions"][g] = {
            "macro_f1": round(float(np.mean([f1(y, p) for p in P])), 2),
            "weighted_f1": round(float(np.mean([f1(y, p, "weighted") for p in P])), 2),
            "positive_rate": round(float(P.mean()), 3), "invalid_rate": round(invalid, 3),
            "writers": len(conds),
            "seconds": round(float(np.mean([r["seconds"] for r in rows])), 3),
            "prompt_tokens": round(float(np.mean([r["ptok"] for r in rows])), 1)}
        if len(conds) > 1:
            flip = np.mean([len(set(P[:, j])) > 1 for j in range(len(ids))])
            res["consistency"][g] = {"flip_rate": round(float(flip), 3)}

    def per_item(g):
        # Majority vote over writers keeps items paired for the bootstrap.
        return (preds[g].mean(0) >= 0.5).astype(int) if len(preds[g]) > 1 else preds[g][0]

    scores = {g: per_item(g) for g in preds}

    def boot(a, b):
        rng = np.random.default_rng(0)
        n = len(y)
        idx = rng.integers(0, n, size=(2000, n))
        diffs = np.array([f1(y[i], scores[a][i]) - f1(y[i], scores[b][i]) for i in idx])
        d = f1(y, scores[a]) - f1(y, scores[b])
        p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"diff": round(d, 2), "ci": [round(float(lo), 2), round(float(hi), 2)], "p": round(float(p), 4)}

    dec = {"total": boot("l1", "l3"), "script": boot("l1", "l2"), "spelling": boot("l2", "l3")}
    t = dec["total"]["diff"]
    dec["spelling_share"] = round(dec["spelling"]["diff"] / t, 3) if abs(t) > 1e-9 else None
    res["decomposition"] = {"canonical": dec}
    res["mitigations"] = {g: boot(g, "l3") for g in ("l1", "cn", "te") if g in scores}
    return res


# ------------------------------------------------------------------ report
def fmt_ci(d: dict) -> str:
    star = "*" if d["p"] < 0.05 else ""
    return f"{d['diff']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]{star}"


def share(value, total) -> str:
    """Spelling share is only meaningful when there is a real drop to share out."""
    if value is None or not total or total["diff"] <= 0 or total["p"] >= 0.05:
        return "n/a"
    return f"{value:.0%}"


def markdown(summary: dict) -> str:
    L = ["# Results", "",
         "Generated by `python -m latscript.eval.analyze` from `results/outputs/`. "
         "Intervals are 95% paired-bootstrap; `*` marks p < 0.05.", ""]
    mt = summary.get("mt", {})
    if mt:
        L += ["## Translation (FLORES devtest → English, chrF++)", "",
              "| Model | Lang | n | L1 | L2 canon | L2 human | L3 | Back-translit | Canonicalize | Few-shot | LoRA (L3) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, r in sorted(mt.items()):
            c = r["conditions"]
            cell = lambda g: f"{c[g]['chrf']:.1f}" if g in c else "–"
            m, lang = key.rsplit("_", 1)
            L.append(f"| {m} | {lang} | {r['n']} | " + " | ".join(
                cell(g) for g in ("l1", "l2c", "l2l", "l3", "bx", "cn", "fs", "lora_l3")) + " |")
        L += ["", "### Decomposition of the L1 → L3 drop (sentence chrF++ points)", "",
              "| Model | Lang | Total L1−L3 | Script L1−L2 | Spelling L2−L3 | Spelling share | Spelling share (human L2) |",
              "|---|---|---|---|---|---|---|"]
        for key, r in sorted(mt.items()):
            d = r["decomposition"]["canonical"]
            lx = r["decomposition"].get("lexical", {}).get("spelling_share")
            m, lang = key.rsplit("_", 1)
            L.append(f"| {m} | {lang} | {fmt_ci(d['total'])} | {fmt_ci(d['script'])} | "
                     f"{fmt_ci(d['spelling'])} | {share(d['spelling_share'], d['total'])} | "
                     f"{share(lx, r['decomposition'].get('lexical', {}).get('total'))} |")
        L += ["", "### Orthographic consistency (mean pairwise chrF++ between outputs for different spellings of the same sentence)", "",
              "| Model | Lang | L3 | Back-translit | Canonicalize | Few-shot | L3 identical outputs |", "|---|---|---|---|---|---|---|"]
        for key, r in sorted(mt.items()):
            c = r["consistency"]
            cell = lambda g: f"{c[g]['pairwise_chrf']:.1f}" if g in c else "–"
            m, lang = key.rsplit("_", 1)
            L.append(f"| {m} | {lang} | {cell('l3')} | {cell('bx')} | {cell('cn')} | {cell('fs')} | "
                     f"{c['l3']['identical_output_rate']:.0%} |" if "l3" in c else "")
        L += ["", "### Mitigation ladder (gain over L3, sentence chrF++; cost per sentence)", "",
              "| Model | Lang | Mitigation | Gain over L3 | Gap to L1 ceiling | Seconds | Prompt tokens |", "|---|---|---|---|---|---|---|"]
        names = {"bx": "Back-transliterate", "cn": "Canonicalize", "fs": "Few-shot (4)", "lora_l3": "LoRA"}
        for key, r in sorted(mt.items()):
            m, lang = key.rsplit("_", 1)
            for g, d in r["mitigations"].items():
                c = r["conditions"][g]
                extra = f" + {c['xlit_seconds_cold']:.2f} xlit" if "xlit_seconds_cold" in c else ""
                L.append(f"| {m} | {lang} | {names[g]} | {fmt_ci(d)} | {fmt_ci(r['ceiling_gap'][g])} | "
                         f"{c['seconds']:.2f}{extra} | {c['prompt_tokens']:.0f} |")
        L.append("")
    clf = summary.get("clf", {})
    if clf:
        L += ["## Offensive-language classification (DravidianCodeMix, romanized test comments)", "",
              "| Model | Lang | n (pos) | L1 (back-translit) | L2 | L3 (as typed) | Canonicalize | Translate first | Flip rate (5 spellings) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key, r in sorted(clf.items()):
            c = r["conditions"]
            cell = lambda g: f"{c[g]['macro_f1']:.1f}" if g in c else "–"
            m, lang = key.rsplit("_", 1)
            flip = r["consistency"].get("s3", {}).get("flip_rate")
            L.append(f"| {m} | {lang} | {r['n']} ({r['positives']}) | {cell('l1')} | {cell('l2')} | {cell('l3')} | "
                     f"{cell('cn')} | {cell('te')} | {'–' if flip is None else f'{flip:.1%}'} |")
        L += ["", "Macro F1. Decomposition (macro-F1 points):", "",
              "| Model | Lang | Total L1−L3 | Script L1−L2 | Spelling L2−L3 |", "|---|---|---|---|---|"]
        for key, r in sorted(clf.items()):
            d = r["decomposition"]["canonical"]
            m, lang = key.rsplit("_", 1)
            L.append(f"| {m} | {lang} | {fmt_ci(d['total'])} | {fmt_ci(d['script'])} | {fmt_ci(d['spelling'])} |")
        L.append("")
    fert = summary.get("fertility")
    if fert:
        L += ["## Tokenizer fertility (tokens per word, FLORES devtest)", "",
              "| Tokenizer | ml L1 | ml L2 | ml L3 | ta L1 | ta L2 | ta L3 |", "|---|---|---|---|---|---|---|"]
        for tok, d in fert.items():
            L.append(f"| {tok} | " + " | ".join(f"{d[l][c]:.2f}" for l in ("ml", "ta") for c in ("l1", "l2c", "l3")) + " |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    cost_path = EVAL / "xlit_cost.json"
    xlit_cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}
    summary = {"mt": {}, "clf": {}}
    for path in sorted(OUTPUTS.glob("*.jsonl")):
        task, rest = path.stem.split("_", 1)
        res = analyze_mt(path, xlit_cost) if task == "mt" else analyze_clf(path)
        if res:
            summary[task][rest] = res
            print(f"scored {path.name}")
    fert = RESULTS / "fertility.json"
    if fert.exists():
        summary["fertility"] = json.loads(fert.read_text())
    summary["xlit_cost"] = xlit_cost
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (RESULTS / "README.md").write_text(markdown(summary))
    print(RESULTS / "README.md")


if __name__ == "__main__":
    main()
