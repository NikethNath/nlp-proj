"""Turn the Phase 1 corpora into the exact inputs every model sees.

FLORES (translation, controlled arm), one row per devtest sentence:

    l1                native script (also the gold-native ceiling)
    l2c / l2l         canonical (IndicXlit) and human-modal L2
    l3_0 .. l3_4      the five simulated writers
    bx_0 .. bx_2      mitigation 1: L3 back-transliterated to native (IndicXlit)
    cn_0 .. cn_2      mitigation 2: L3 canonicalized to L2 without leaving Latin

The mitigation inputs exist only for writers 0-2 of the first N_MIT sentences
of the fixed evaluation order (``flores_order``); IndicXlit runs at about two
words per second on a shared laptop CPU, which made all 1,012 x 5 impractical.

DravidianCodeMix (classification, ecological arm), romanized in-language test
comments only:

    l3                the comment as its author typed it
    l1                back-transliterated with IndicXlit (= mitigation 1 here;
                      there is no gold native text for real comments)
    l2                l1 re-romanized with the rule romanizer (deterministic,
                      validated against Dakshina in Phase 1; IndicXlit was too slow
                      for every back-transliterated word)
    s3_0 .. s3_4      l1 re-spelled by the five simulated writers, used for the
                      orthographic consistency (flip rate) metric
    cn                l3 canonicalized to L2

Canonicalization maps each Latin word to its most likely native word - the
native word Dakshina annotators most often meant by that exact spelling, else
IndicXlit - and then to that word's canonical romanization (IndicXlit where
cached, which covers every FLORES word, else the rule romanizer). It is a
lookup, so its cost is near zero once the tables exist.

Tamil DravidianCodeMix is sampled to N_DCM_TA comments (seeded, label ratio
kept) for the same reason.

Writes data/derived/eval/{flores,dcm}_{lang}.jsonl and data/derived/eval/xlit_cost.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..translit import aksharantar, indicxlit
from ..translit.generate import Builder
from ..translit.rules import transliterate
from ..translit.tokens import LATIN_WORD, NATIVE_WORD, map_matches, map_words

ROOT = Path(__file__).resolve().parents[3]
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "eval"
K = 5
N_MIT = 400          # FLORES sentences with mitigation inputs
MIT_WRITERS = 3
N_DCM_TA = 1200      # Tamil test comments kept (Malayalam keeps all 1,023)


def flores_order(n: int) -> list[int]:
    """The fixed evaluation order of FLORES devtest; subsets are its prefixes."""
    return random.Random(2026).sample(range(n), n)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def latin_words(texts) -> set[str]:
    return {m.lower() for t in texts for m in LATIN_WORD.findall(t)}


class Canonicalizer:
    """Latin spelling -> native word -> canonical (IndicXlit) spelling."""

    def __init__(self, lang: str, lexicon):
        self.lang = lang
        votes: dict[str, Counter] = defaultdict(Counter)
        for native, spellings in lexicon.entries.items():
            for roman, n in spellings.items():
                votes[roman.lower()][native] += n
        self.reverse = {r: c.most_common(1)[0][0] for r, c in votes.items()}
        self.to_native: dict[str, str] = dict(self.reverse)
        self.to_roman = indicxlit._read_cache(indicxlit._cache_path("indic2en", lang))

    def prepare(self, texts) -> None:
        unknown = {w for w in latin_words(texts) if w not in self.to_native}
        self.to_native.update(indicxlit.table(unknown, self.lang, "en2indic"))

    def roman(self, native: str) -> str:
        return self.to_roman.get(native) or transliterate(native, self.lang)

    def __call__(self, text: str) -> str:
        def fix(word: str) -> str:
            native = self.to_native.get(word.lower())
            return self.roman(native) if native else word
        return map_matches(text, LATIN_WORD, fix)


def time_xlit(lang: str, words: list[str], direction: str) -> float | None:
    """Cold-cache IndicXlit throughput in words/second, on words not yet cached."""
    path = indicxlit._cache_path(direction, lang)
    cached = indicxlit._read_cache(path)
    probe = [w for w in words if w not in cached][:400]
    if len(probe) < 100:
        return None
    start = time.perf_counter()
    indicxlit.table(probe, lang, direction)
    return len(probe) / (time.perf_counter() - start)


def prepare_flores(lang: str, builder: Builder, canon: Canonicalizer) -> list[dict]:
    rows = read_jsonl(DERIVED / f"flores_{lang}_devtest.jsonl")
    order = flores_order(len(rows))
    mit = {order[i] for i in range(N_MIT)}
    l3 = [w for i in mit for w in rows[i]["l3_writers"][:MIT_WRITERS]]
    back = indicxlit.table(latin_words(l3), lang, "en2indic")
    canon.prepare(l3)
    out = []
    for n, r in enumerate(rows):
        row = {"id": r["id"], "idx": r["idx"], "lang": lang, "ref": r["english"],
               "l1": r["l1_native"], "l2c": r["l2_canonical"], "l2l": r["l2_lexical"],
               "n_words": len(r["l1_native"].split())}
        for i, s in enumerate(r["l3_writers"][:K]):
            row[f"l3_{i}"] = s
            if n in mit and i < MIT_WRITERS:
                row[f"bx_{i}"] = indicxlit.nativize(s, lang, back)
                row[f"cn_{i}"] = canon(s)
        out.append(row)
    return out


def prepare_dcm(lang: str, builder: Builder, canon: Canonicalizer) -> list[dict]:
    rows = [r for r in read_jsonl(DERIVED / f"dcm_{lang}_test.jsonl")
            if r["in_language"] and r["script"] == "latin"]
    if lang == "ta":
        rows = sorted(random.Random(2026).sample(rows, N_DCM_TA), key=lambda r: r["id"])
    natives = {m for r in rows for m in NATIVE_WORD[lang].findall(r["l1_backtranslit"])}
    unseen = {w for w in natives if w not in builder.lexicon.entries}
    builder.lexicon.backoff.update(aksharantar.lookup(unseen, lang))
    canon.prepare([r["text"] for r in rows])
    rule = lambda w: transliterate(w, lang)
    out = []
    for r in rows:
        l1 = r["l1_backtranslit"]
        row = {"id": r["id"], "lang": lang, "label": int(r["offensive"]),
               "label_name": r["label_name"], "l3": r["text"], "l1": l1,
               "l2": map_matches(l1, NATIVE_WORD[lang], rule), "cn": canon(r["text"])}
        for i, w in enumerate(builder.writers):
            row[f"s3_{i}"] = map_words(
                l1, lambda t: builder.speller.spell(t, w) if NATIVE_WORD[lang].search(t) else t)
        out.append(row)
    return out


def canon_coverage(canon: Canonicalizer, rows: list[dict]) -> float:
    """Share of canonicalized FLORES words whose target came from IndicXlit."""
    hits = total = 0
    for r in rows:
        for w in LATIN_WORD.findall(r.get("cn_0", "") and r["l3_0"]):
            native = canon.to_native.get(w.lower())
            if native:
                total += 1
                hits += native in canon.to_roman
    return hits / max(total, 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=["ml", "ta"])
    args = ap.parse_args(argv)
    cost = {}
    for lang in args.langs:
        builder = Builder(lang, k=K)
        canon = Canonicalizer(lang, builder.lexicon)
        # Throughput is measured before the bulk run, on words that are still uncached.
        rows = read_jsonl(DERIVED / f"flores_{lang}_devtest.jsonl")
        sample = [rows[i]["l3_writers"][0] for i in flores_order(len(rows))[:N_MIT]]
        cost[lang] = {"en2indic_words_per_s": time_xlit(lang, sorted(latin_words(sample)), "en2indic")}
        flores = prepare_flores(lang, builder, canon)
        write_jsonl(flores, OUT / f"flores_{lang}.jsonl")
        dcm = prepare_dcm(lang, builder, canon)
        write_jsonl(dcm, OUT / f"dcm_{lang}.jsonl")
        cost[lang]["latin_words_per_sentence"] = round(
            sum(len(LATIN_WORD.findall(r["l3_0"])) for r in flores) / len(flores), 2)
        cost[lang]["canonical_from_indicxlit"] = round(canon_coverage(canon, flores), 3)
        print(f"{lang}: flores {len(flores)}, dcm {len(dcm)} "
              f"({sum(r['label'] for r in dcm)} offensive), cost {cost[lang]}")
    (OUT / "xlit_cost.json").write_text(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
