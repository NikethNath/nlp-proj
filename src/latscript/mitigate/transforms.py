"""Input-side mitigations: rewrite romanized text before the model sees it.

Two rewrites, both built on IndicXlit so they need no training:

* ``bt`` (back-transliteration) - Latin -> native script. For models that
  know the native script far better than romanized text (NLLB, LaBSE,
  BGE-M3), this is the obvious fix. How well it works depends on how well
  IndicXlit reconstructs the native spelling, and inconsistent romanization
  should make that harder, so L3 may recover less than L2.
* ``rt`` (round-trip) - Latin -> native -> Latin. The result stays in Latin
  script but every word is respelled in IndicXlit's one canonical form. For
  models that prefer romanized input (Qwen, encoders fine-tuned on romanized
  comments), this is spelling normalization without a change of script. A
  perfect round-trip would turn every L3 writer into L2 canonical.

Words already in native script, digits and punctuation pass through. English
words in romanized text are transliterated as well; no language
identification is attempted, because none is reliable on single romanized
words.

    PYTHONPATH=src python -m latscript.mitigate.transforms

writes ``data/derived/mitigated/flores_{lang}_devtest.jsonl`` with ``bt_<cond>``
and ``rt_<cond>`` for every romanized condition, and
``data/derived/mitigated/dcm_{lang}_{split}.jsonl`` with ``bt_text`` (train) or
``rt_text`` (dev, test) for Latin and mixed comments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval.conditions import CONDITIONS, DERIVED, LANGS, load_dcm, texts
from ..translit import indicxlit
from ..translit.tokens import LATIN_WORD, NATIVE_WORD

OUT = DERIVED / "mitigated"
ROMANIZED = tuple(c for c in CONDITIONS if c != "l1")


class Rewriter:
    """Back-transliteration and round-trip for one language, with shared tables."""

    def __init__(self, lang: str):
        self.lang = lang
        self.to_native: dict[str, str] = {}
        self.to_latin: dict[str, str] = {}

    def prepare_native(self, romanized: list[str]) -> None:
        """Run IndicXlit Latin -> native over every word the texts need."""
        vocab = {w.lower() for t in romanized for w in LATIN_WORD.findall(t)}
        self.to_native.update(indicxlit.table(vocab, self.lang, "en2indic"))

    def prepare(self, romanized: list[str]) -> None:
        """Run IndicXlit once over every word the texts need, in both directions."""
        self.prepare_native(romanized)
        natives = {m for t in romanized for m in NATIVE_WORD[self.lang].findall(self.bt(t))}
        self.to_latin.update(indicxlit.table(natives, self.lang, "indic2en"))

    def bt(self, text: str) -> str:
        return indicxlit.nativize(text, self.lang, self.to_native)

    def rt(self, text: str) -> str:
        return indicxlit.romanize(self.bt(text), self.lang, self.to_latin)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_flores(lang: str, splits=("devtest",)) -> None:
    """Rewritten inputs for evaluation. Dev is used for training on romanized
    text directly, so it needs no rewriting."""
    rows = {s: _read(DERIVED / f"flores_{lang}_{s}.jsonl") for s in splits}
    rw = Rewriter(lang)
    rw.prepare([texts(r)[c] for rs in rows.values() for r in rs for c in ROMANIZED])
    for split, rs in rows.items():
        out = []
        for r in rs:
            t = texts(r)
            rec = {"id": r["id"], "idx": r["idx"]}
            for c in ROMANIZED:
                rec[f"bt_{c}"] = rw.bt(t[c])
                rec[f"rt_{c}"] = rw.rt(t[c])
            out.append(rec)
        _write(OUT / f"flores_{lang}_{split}.jsonl", out)
        print(f"  flores_{lang}_{split}: {len(out)} sentences")


def build_dcm(lang: str) -> None:
    """Back-transliteration of train, round-trip of dev and test.

    Dev and test already carry ``l1_backtranslit`` from the corpus build; train
    gets ``bt_text`` here so classifiers can be trained on both scripts. Only
    dev and test get ``rt_text``, the spelling normalization applied at
    inference.
    """
    latin = lambda c: c["script"] in ("latin", "mixed")
    comments = {s: load_dcm(lang, s) for s in ("train", "dev", "test")}
    rw = Rewriter(lang)
    rw.prepare_native([c["text"] for c in comments["train"] if latin(c)])
    rw.prepare([c["text"] for s in ("dev", "test") for c in comments[s] if latin(c)])
    for split, cs in comments.items():
        out = []
        for c in cs:
            rec = {"id": c["id"], "bt_text": None, "rt_text": None}
            if latin(c) and split == "train":
                rec["bt_text"] = rw.bt(c["text"])
            elif latin(c):
                rec["rt_text"] = rw.rt(c["text"])
            out.append(rec)
        _write(OUT / f"dcm_{lang}_{split}.jsonl", out)
        print(f"  dcm_{lang}_{split}: {sum(bool(o['bt_text'] or o['rt_text']) for o in out)} rewritten")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--skip-dcm", action="store_true")
    args = ap.parse_args(argv)
    for lang in args.langs:
        build_flores(lang)
        if not args.skip_dcm:
            build_dcm(lang)


if __name__ == "__main__":
    main()
