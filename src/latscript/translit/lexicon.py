"""Human romanization distributions harvested from the Dakshina dataset.

Two sources, both human-written:

* ``lexicons/*.translit.sampled.*.tsv`` - native word, one romanization, and
  the number of annotators who produced it. Multiple rows per word, so the
  third column is literally an empirical distribution over spellings.
* ``romanized/*.aligned.cased_nopunct.tsv`` - word-aligned pairs drawn from
  full romanized sentences, separated by ``</s>``. Noisier and more colloquial
  than the lexicon (annotators were transcribing running text, not words in
  isolation), which is exactly why it is worth folding in.

The resulting map drives both conditions: the modal spelling is L2-lexical,
and sampling from the distribution is L3.

Both sources carry annotation errors, and they are not subtle: the Tamil
lexicon lists vanakkam only as "anne", gives chennai the spelling "noolagam"
(a different word), and romanizes haari as "potter" - a worker whose input was
off by one row. Left in, these teach the noise model that any letter may become
any other. `max_distance` drops pairs too far from the canonical romanization
to be a spelling of the same word; 0.6 removes ~2% of Malayalam and ~6% of
Tamil attestations.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .distance import normalized_edit_distance
from .rules import transliterate

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "dakshina_dataset_v1.0"
SENT_BREAK = "</s>"

# Split on runs of non-letters so punctuation survives untouched in its own slot.
# Python's \w excludes combining marks, so a bare \W split would cut every word
# at its vowel signs and viramas; the Tamil and Malayalam blocks (and the joiners
# used to form chillus) are listed explicitly as word characters.
_TOKEN = re.compile(r"([^\w\u0B80-\u0BFF\u0D00-\u0D7F\u200c\u200d]+)")


@dataclass
class Lexicon:
    """native word -> Counter of human romanizations."""

    lang: str
    entries: dict[str, Counter] = field(default_factory=dict)
    max_distance: float | None = 0.6
    rejected: int = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, lang: str, root: Path | None = None,
             include_running_text: bool = True,
             max_distance: float | None = 0.6) -> "Lexicon":
        root = root or DEFAULT_ROOT
        lex = cls(lang=lang, max_distance=max_distance)

        lexdir = root / lang / "lexicons"
        found = sorted(lexdir.glob(f"{lang}.translit.sampled.*.tsv"))
        if not found:
            raise FileNotFoundError(f"no Dakshina lexicon for {lang!r} under {lexdir}")
        for path in found:
            lex._absorb_lexicon(path)

        if include_running_text:
            rom = root / lang / "romanized" / f"{lang}.romanized.rejoined.aligned.cased_nopunct.tsv"
            if rom.exists():
                lex._absorb_aligned(rom)
        return lex

    def _absorb_lexicon(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                native, roman = parts[0].strip(), parts[1].strip().lower()
                count = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 1
                if native and roman and self._plausible(native, roman):
                    self.entries.setdefault(native, Counter())[roman] += count

    def _absorb_aligned(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                native, roman = parts[0].strip(), parts[1].strip().lower()
                if not native or not roman or native == SENT_BREAK:
                    continue
                if self._plausible(native, roman):
                    self.entries.setdefault(native, Counter())[roman] += 1

    def _plausible(self, native: str, roman: str) -> bool:
        """Reject pairs that are too distant to be the same word."""
        if self.max_distance is None:
            return True
        ok = normalized_edit_distance(transliterate(native, self.lang),
                                      roman) <= self.max_distance
        self.rejected += not ok
        return ok

    # -- lookup ------------------------------------------------------------
    def canonical(self, word: str) -> str:
        """Modal human spelling, falling back to the rule romanizer for OOV."""
        counts = self.entries.get(word)
        if not counts:
            return transliterate(word, self.lang)
        # max() over (count, spelling) keeps ties deterministic.
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def sample(self, word: str, rng: random.Random) -> str:
        """Draw a spelling in proportion to how often humans wrote it."""
        counts = self.entries.get(word)
        if not counts:
            return transliterate(word, self.lang)
        spellings, weights = zip(*counts.items())
        return rng.choices(spellings, weights=weights, k=1)[0]

    def n_variants(self, word: str) -> int:
        return len(self.entries.get(word, ()))

    # -- sentence level ----------------------------------------------------
    def _map_sentence(self, text: str, pick) -> str:
        return "".join(
            part if not part or _TOKEN.fullmatch(part) else pick(part)
            for part in _TOKEN.split(text)
        )

    def canonical_sentence(self, text: str) -> str:
        return self._map_sentence(text, self.canonical)

    def sample_sentence(self, text: str, rng: random.Random) -> str:
        return self._map_sentence(text, lambda w: self.sample(w, rng))

    def coverage(self, text: str) -> tuple[int, int]:
        """(tokens found in the lexicon, native-script tokens) for `text`."""
        hit = total = 0
        for part in _TOKEN.split(text):
            if not part or _TOKEN.fullmatch(part):
                continue
            total += 1
            hit += part in self.entries
        return hit, total

    def __len__(self) -> int:
        return len(self.entries)
