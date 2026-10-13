"""Classroom demo. Runs on a laptop CPU; every mode works without a network.

    python -m latscript.demo results                    # main tables, no model needed
    python -m latscript.demo corpus --lang ml           # one FLORES sentence at every level, translated live
    python -m latscript.demo type --lang ml "nammal innu vaikunneram kadappurathu pokum"
    python -m latscript.demo classify --lang ta "padam semma mokka da"

``corpus --offline`` shows the saved translations instead of running NLLB.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import textwrap
import time
from pathlib import Path

from .translit import indicxlit
from .translit.tokens import LATIN_WORD, NATIVE_WORD, map_matches, map_words

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "derived" / "eval"
LANG = {"ml": "Malayalam", "ta": "Tamil"}
W = 100


def rule(title: str = "") -> None:
    print(("── " + title + " ").ljust(W, "─") if title else "─" * W)


def row(label: str, text: str, extra: str = "") -> None:
    lines = textwrap.wrap(text, W - 26) or [""]
    for i, chunk in enumerate(lines):
        print(f"{label if i == 0 else '':<14}{chunk:<{W - 26}}  {extra if i == 0 else ''}")


def say(msg: str) -> None:
    print(f"[{msg}]", flush=True)


def load_nllb():
    say("loading NLLB-200 600M (about 15 s)...")
    from .eval.backends import NLLB
    return NLLB()


def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).sentence_score(hyp, [ref]).score


class Respeller:
    """Back-transliterates typed words and re-spells them as the simulated writers would."""

    def __init__(self, lang: str):
        say("loading the Dakshina spelling model (about 10 s)...")
        from .eval.prepare import Canonicalizer
        from .translit.generate import Builder
        self.lang = lang
        self.builder = Builder(lang, k=5, use_indicxlit=False)
        self.canon = Canonicalizer(lang, self.builder.lexicon)
        self.canon.to_native.update(indicxlit._read_cache(indicxlit._cache_path("en2indic", lang)))

    def to_native(self, text: str, use_xlit: bool = False) -> str:
        if use_xlit:
            unknown = {w.lower() for w in LATIN_WORD.findall(text)} - self.canon.to_native.keys()
            if unknown:
                say(f"running IndicXlit on {len(unknown)} new words (about a minute)...")
                self.canon.to_native.update(indicxlit.table(unknown, self.lang, "en2indic"))
        return map_matches(text, LATIN_WORD, lambda w: self.canon.to_native.get(w.lower(), w))

    def respell(self, native: str, writer: int) -> str:
        w = self.builder.writers[writer]
        return map_words(native, lambda t: self.builder.speller.spell(t, w)
                         if NATIVE_WORD[self.lang].search(t) else t)


# ------------------------------------------------------------------ modes
def mode_results(_args) -> None:
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    rule("Translation, FLORES → English (chrF++, higher is better)")
    print(f"{'model':<12}{'lang':<6}{'native':>8}{'canon.':>8}{'natural':>9}{'script':>9}{'spelling':>10}"
          f"{'back-xlit':>11}{'agree':>8}")
    for key, r in sorted(s["mt"].items()):
        c, d = r["conditions"], r["decomposition"]["canonical"]
        m, lang = key.rsplit("_", 1)
        print(f"{m:<12}{lang:<6}{c['l1']['chrf']:>8.1f}{c['l2c']['chrf']:>8.1f}{c['l3']['chrf']:>9.1f}"
              f"{d['script']['diff']:>+9.1f}{d['spelling']['diff']:>+10.1f}"
              f"{c.get('bx', {}).get('chrf', float('nan')):>11.1f}"
              f"{r['consistency'].get('l3', {}).get('pairwise_chrf', float('nan')):>8.1f}")
    print("script = native − canonical;  spelling = canonical − natural;  "
          "agree = chrF++ between outputs for two spellings")
    rule("Offensive-comment classification (macro F1) and flip rate across 5 spellings")
    for key, r in sorted(s["clf"].items()):
        c = r["conditions"]
        m, lang = key.rsplit("_", 1)
        flip = r["consistency"].get("s3", {}).get("flip_rate")
        print(f"{m:<12}{lang:<6}as typed {c['l3']['macro_f1']:5.1f}   native {c['l1']['macro_f1']:5.1f}   "
              f"flip rate {flip:.0%}" if flip is not None else "")
    rule()


def mode_corpus(args) -> None:
    from .eval.prepare import flores_order, read_jsonl
    rows = read_jsonl(EVAL / f"flores_{args.lang}.jsonl")
    order = flores_order(len(rows))
    if args.idx is None:
        r = rows[order[random.Random(args.seed).randrange(200)]]
    else:
        r = next(x for x in rows if x["idx"] == args.idx)
    levels = [("L1 native", "l1"), ("L2 canonical", "l2c"), ("L3 writer 1", "l3_0"),
              ("L3 writer 2", "l3_1"), ("L3 writer 3", "l3_2"), ("back-xlit w1", "bx_0")]
    levels = [(n, f) for n, f in levels if f in r]
    if args.offline:
        saved = {}
        for x in read_jsonl(ROOT / "results" / "outputs" / f"mt_nllb-600m_{args.lang}.jsonl"):
            if x["id"] == r["id"]:
                saved[x["cond"]] = x["out"]
        hyps = [saved.get(f, "(not in the saved sample: run without --offline)") for _, f in levels]
    else:
        model = load_nllb()
        t0 = time.perf_counter()
        hyps = [h.text for h in model.translate([r[f] for _, f in levels], args.lang)]
        say(f"translated {len(levels)} versions in {time.perf_counter() - t0:.1f} s")
    rule(f"{LANG[args.lang]} FLORES sentence {r['idx']}: the same meaning, written {len(levels)} ways")
    for (name, f) in levels:
        row(name, r[f])
    rule("NLLB-200's English translation of each, with chrF++ against the reference")
    row("reference", r["ref"])
    print()
    for (name, f), h in zip(levels, hyps):
        row(name, h, f"chrF++ {chrf(h, r['ref']):5.1f}" if not h.startswith("(not") else "")
    rule()


def mode_type(args) -> None:
    sp = Respeller(args.lang)
    native = sp.to_native(args.text, use_xlit=args.xlit)
    versions = [("you typed", args.text)]
    versions += [(f"writer {i + 1}", sp.respell(native, i)) for i in range(3)]
    versions += [("back-xlit", native)]
    model = load_nllb()
    hyps = [h.text for h in model.translate([v for _, v in versions], args.lang)]
    rule(f"Your {LANG[args.lang]} sentence, as other people might type it")
    for name, v in versions:
        row(name, v)
    rule("NLLB-200's English translation of each version")
    for (name, _), h in zip(versions, hyps):
        row(name, h)
    rule()
    left = [w for w in LATIN_WORD.findall(native)]
    if left:
        print(f"(not in the spelling dictionary, left in Latin letters: {', '.join(left)}; add --xlit to convert them)")


def mode_classify(args) -> None:
    path = ROOT / "data" / "cache" / f"demo_charlr_{args.lang}.pkl"
    if path.exists():
        vec, clf = pickle.loads(path.read_bytes())
    else:
        say("training the char n-gram classifier on DravidianCodeMix (about a minute, once)...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from .eval.train_clf import train_rows
        train = train_rows(args.lang)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                              max_features=300_000)
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(vec.fit_transform([r["text"] for r in train]), [int(r["offensive"]) for r in train])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps((vec, clf)))
    sp = Respeller(args.lang)
    native = sp.to_native(args.text, use_xlit=args.xlit)
    versions = [("as typed", args.text)] + [(f"writer {i + 1}", sp.respell(native, i)) for i in range(5)]
    probs = clf.predict_proba(vec.transform([v for _, v in versions]))[:, 1]
    rule("Offensive-comment classifier on the same comment, spelled six ways")
    for (name, v), p in zip(versions, probs):
        row(name, v, f"{'OFFENSIVE' if p >= 0.5 else 'not offensive':<13} p={p:.2f}")
    flips = len({p >= 0.5 for p in probs}) > 1
    rule("the decision changes with spelling" if flips else "the decision is stable for this comment")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("results")
    p = sub.add_parser("corpus")
    p.add_argument("--lang", choices=LANG, default="ml")
    p.add_argument("--idx", type=int, help="FLORES devtest index (default: random from the evaluated 200)")
    p.add_argument("--seed", type=int, default=int(time.time()))
    p.add_argument("--offline", action="store_true", help="use saved translations, no model")
    for name in ("type", "classify"):
        p = sub.add_parser(name)
        p.add_argument("--lang", choices=LANG, default="ml")
        p.add_argument("--xlit", action="store_true", help="run IndicXlit on words not in the dictionary")
        p.add_argument("text")
    args = ap.parse_args(argv)
    {"results": mode_results, "corpus": mode_corpus, "type": mode_type, "classify": mode_classify}[args.mode](args)


if __name__ == "__main__":
    main()
