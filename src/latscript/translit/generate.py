"""Build the three orthographic conditions from native-script FLORES.

    L1  l1_native     native script, as published
    L2  l2_canonical  IndicXlit romanization - one deterministic form per word
        l2_lexical    most common human spelling (Dakshina), else Aksharantar,
                      else the rule romanizer
    L3  l3_writers    k simulated writers, each internally consistent

Why two L2s. IndicXlit is the canonical form fixed in the study design. But its
conventions are a machine's (it often drops Malayalam's final half-u), while L3
is sampled from human spelling distributions. The L2->L3 gap measured against
IndicXlit therefore mixes spelling variance with a systematic shift in
convention. ``l2_lexical`` is drawn from the same human data as L3, so the
difference between the two decompositions shows how much that shift matters.

L3 words that humans attested are always a spelling some annotator actually
wrote, so the variation is real. It is still somewhat conservative: two
simulated writers disagree on about 80% as many words as two Dakshina
annotators do (see docs/corpus_stats.json). Degradation measured against it is
a mild lower bound on what naturally typed text would cause.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..corpora.flores import Segment, load as load_flores
from . import aksharantar, indicxlit
from .lexicon import Lexicon
from .noise import CharPlausibility, NaturalSpeller, SpellingNoise
from .rules import transliterate
from .tokens import NATIVE_WORD, map_words, words

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "derived"


@dataclass
class Conditions:
    """One FLORES sentence rendered at every orthographic level."""

    id: str
    idx: int
    lang: str
    split: str
    english: str
    l1_native: str
    l2_canonical: str
    l2_lexical: str
    l3_writers: list[str]
    human_coverage: float
    lexical_coverage: float


class Builder:
    """Holds the per-language models so every split is built from the same ones."""

    def __init__(self, lang: str, k: int = 5, seed: int = 0,
                 use_indicxlit: bool = True, use_aksharantar: bool = True):
        self.lang, self.k, self.seed = lang, k, seed
        self.use_indicxlit, self.use_aksharantar = use_indicxlit, use_aksharantar
        self.lexicon = Lexicon.load(lang)
        self.noise = SpellingNoise.learn(self.lexicon)
        self.speller = NaturalSpeller(self.lexicon, self.noise, CharPlausibility(self.lexicon))
        # Writers are fixed for the whole corpus: writer 3 is the same person on
        # every sentence and in every split.
        self.writers = [self.noise.writer(random.Random(seed * 1000 + i), name=f"{seed}-{i}")
                        for i in range(k)]
        self.xlit: dict[str, str] = {}

    def _is_native(self, word: str) -> bool:
        return bool(NATIVE_WORD[self.lang].search(word))

    def prepare(self, segments: list[Segment]) -> None:
        vocab = {w for s in segments for w in words(s.native) if self._is_native(w)}
        if self.use_aksharantar:
            unseen = {w for w in vocab if w not in self.lexicon.entries}
            self.lexicon.backoff.update(aksharantar.lookup(unseen, self.lang))
        if self.use_indicxlit:
            runs = {m for s in segments for m in NATIVE_WORD[self.lang].findall(s.native)}
            self.xlit.update(indicxlit.table(runs, self.lang, "indic2en"))

    def _canonical(self, text: str) -> str:
        if not self.use_indicxlit:
            return ""
        return NATIVE_WORD[self.lang].sub(
            lambda m: self.xlit.get(m.group(0)) or transliterate(m.group(0), self.lang), text)

    def _lexical(self, text: str) -> str:
        return map_words(text, lambda w: self.lexicon.canonical(w) if self._is_native(w) else w)

    def _writer(self, text: str, writer) -> str:
        return map_words(text, lambda w: self.speller.spell(w, writer) if self._is_native(w) else w)

    def build(self, segments: list[Segment]) -> list[Conditions]:
        self.prepare(segments)
        rows = []
        for seg in segments:
            native_words = [w for w in words(seg.native) if self._is_native(w)]
            n = len(native_words) or 1
            rows.append(Conditions(
                id=f"flores-{seg.lang}-{seg.split}-{seg.idx:04d}",
                idx=seg.idx, lang=seg.lang, split=seg.split, english=seg.english,
                l1_native=seg.native,
                l2_canonical=self._canonical(seg.native),
                l2_lexical=self._lexical(seg.native),
                l3_writers=[self._writer(seg.native, w) for w in self.writers],
                human_coverage=round(sum(w in self.lexicon.entries for w in native_words) / n, 4),
                lexical_coverage=round(sum(w in self.lexicon.entries or w in self.lexicon.backoff
                                           for w in native_words) / n, 4),
            ))
        return rows


def write_jsonl(rows, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=["ml", "ta"])
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-indicxlit", action="store_true")
    ap.add_argument("--no-aksharantar", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    for lang in args.langs:
        builder = Builder(lang, k=args.k, seed=args.seed,
                          use_indicxlit=not args.no_indicxlit,
                          use_aksharantar=not args.no_aksharantar)
        segments = {s: load_flores(lang, s, limit=args.limit) for s in args.splits}
        # Prepare on all splits at once so IndicXlit is called a single time.
        builder.prepare([seg for segs in segments.values() for seg in segs])
        for split, segs in segments.items():
            rows = builder.build(segs)
            path = write_jsonl(rows, args.out / f"flores_{lang}_{split}.jsonl")
            human = sum(r.human_coverage for r in rows) / len(rows)
            lexical = sum(r.lexical_coverage for r in rows) / len(rows)
            print(f"  {path.name}: {len(rows)} sentences, k={args.k}, "
                  f"human-attested words {human:.1%}, with Aksharantar {lexical:.1%}")


if __name__ == "__main__":
    main()
