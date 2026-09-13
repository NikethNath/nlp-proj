"""A learned spelling-variation model for words Dakshina has never seen.

The lexicon covers only ~68% (ml) / ~77% (ta) of FLORES tokens, because Malayalam and Tamil
are agglutinative and a 60k-type word list cannot enumerate inflected forms.
Falling back to the deterministic romanizer for the rest would be fatal to the
study: L2 and L3 would be *identical* on every unseen token, and the spelling-variance
term the whole decomposition rests on would be biased toward zero.

So instead we learn how humans deviate from the canonical romanization and
transfer that to unseen words. For every (native word, human spelling) pair in
Dakshina we align the rule output against what the human actually typed, and
record what each canonical unit turned into - "aa"->"a", final "u"->"",
Tamil "d"->"t".

Segmentation matters more than it looks. Learning and generation both run over
the same fixed inventory of orthographic units (digraphs like "th"/"zh"/"aa"
plus single characters), so that the probability of a unit surviving unchanged
is estimated at exactly the granularity at which it is later resampled. Count
identity charwise while counting substitutions chunkwise and every digraph
looks maximally unstable.
"""
from __future__ import annotations

import difflib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .lexicon import Lexicon
from .rules import transliterate

# Longest-match wins, so order within a length class is irrelevant.
UNITS = (
    "chh", "aa", "ee", "oo", "ai", "au",
    "kh", "gh", "ch", "jh", "nj", "ng", "th", "dh", "bh", "ph", "sh", "zh",
)
_MAX_UNIT = max(len(u) for u in UNITS)


def segment(roman: str) -> list[str]:
    """Split a canonical romanization into orthographic units."""
    out, i, n = [], 0, len(roman)
    while i < n:
        for size in range(min(_MAX_UNIT, n - i), 1, -1):
            if roman[i:i + size] in UNITS:
                out.append(roman[i:i + size])
                i += size
                break
        else:
            out.append(roman[i])
            i += 1
    return out


def _position_map(src: str, tgt: str) -> list[int]:
    """Monotonic map from each index of `src` (plus the end) into `tgt`."""
    pos = [0] * (len(src) + 1)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, tgt,
                                                       autojunk=False).get_opcodes():
        span_s, span_t = i2 - i1, j2 - j1
        for i in range(i1, i2):
            if tag == "equal":
                pos[i] = j1 + (i - i1)
            else:
                pos[i] = j1 + round((i - i1) / span_s * span_t) if span_s else j1
    pos[len(src)] = len(tgt)
    return pos


@dataclass
class SpellingNoise:
    """P(typed form | canonical unit, word-final?) estimated from human data."""

    lang: str
    table: dict[tuple[str, bool], Counter] = field(default_factory=dict)

    @classmethod
    def learn(cls, lex: Lexicon, min_count: int = 3,
              max_edit_ratio: float = 0.5) -> "SpellingNoise":
        raw: dict[tuple[str, bool], Counter] = defaultdict(Counter)
        for native, counts in lex.entries.items():
            rule = transliterate(native, lex.lang)
            if not rule:
                continue
            units = segment(rule)
            for human, weight in counts.items():
                # A wildly divergent "spelling" is usually a different word
                # (annotator error, or an English gloss). Learning from it
                # injects noise that has nothing to do with orthography.
                if abs(len(human) - len(rule)) > max_edit_ratio * len(rule):
                    continue
                pos = _position_map(rule, human)
                off = 0
                for unit in units:
                    end = off + len(unit)
                    raw[(unit, end == len(rule))][human[pos[off]:pos[end]]] += weight
                    off = end

        table = {}
        for key, counter in raw.items():
            kept = Counter({k: v for k, v in counter.items() if v >= min_count})
            if len(kept) > 1:
                table[key] = kept
        return cls(lang=lex.lang, table=table)

    def perturb(self, roman: str, rng: random.Random) -> str:
        """Resample a canonical romanization, drawing each unit independently.

        Useful as an ablation. Prefer :meth:`writer` for building L3: real
        variation is between people, not between tokens.
        """
        units = segment(roman)
        out, off = [], 0
        for unit in units:
            off += len(unit)
            out.append(self._draw(unit, off == len(roman), rng))
        return "".join(out)

    def _draw(self, unit: str, final: bool, rng: random.Random) -> str:
        counter = self.table.get((unit, final))
        if not counter:
            return unit
        keys, weights = zip(*counter.items())
        return rng.choices(keys, weights=weights, k=1)[0]

    def writer(self, rng: random.Random, consistency: float = 0.85,
               min_habit_prob: float = 0.10) -> "Writer":
        """Draw one person, with habits they mostly stick to.

        Tamil is the clearest case: 67% of annotators render `dh` as "th" and
        23% keep "dh", but almost nobody alternates between them inside a
        single text. Sampling each token independently would manufacture
        chimeras - "thadagom" - that no individual would ever type, and would
        overstate L3's difficulty for reasons that have nothing to do with how
        people actually write.

        A habit is only worth committing to if a real share of annotators
        share it. Letting someone lock in a 1%-frequency rendering and then
        apply it everywhere invents a typist who does not exist - that is where
        "vnkkm" and "ncangalude" came from.
        """
        habits = {}
        for key, counter in self.table.items():
            total = sum(counter.values())
            choice = self._draw(key[0], key[1], rng)
            if counter[choice] / total >= min_habit_prob:
                habits[key] = choice
        return Writer(self, habits, consistency)


@dataclass
class Writer:
    """One simulated typist: a fixed set of spelling habits, loosely held."""

    noise: "SpellingNoise"
    habits: dict[tuple[str, bool], str]
    consistency: float = 0.85

    def spell(self, roman: str, rng: random.Random) -> str:
        out, off = [], 0
        for unit in segment(roman):
            off += len(unit)
            key = (unit, off == len(roman))
            if key in self.habits and rng.random() < self.consistency:
                out.append(self.habits[key])
            else:
                out.append(self.noise._draw(unit, key[1], rng))
        return "".join(out)

    def identity_rate(self, unit: str, final: bool) -> float:
        counter = self.table.get((unit, final))
        if not counter:
            return 1.0
        return counter[unit] / sum(counter.values())

    def __len__(self) -> int:
        return len(self.table)
