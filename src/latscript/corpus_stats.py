"""Descriptive statistics for the released corpus, written to docs/corpus_stats.json.

Run after the corpus has been generated::

    PYTHONPATH=src python -m latscript.corpus_stats
"""
from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path

from .translit.distance import normalized_edit_distance
from .translit.lexicon import Lexicon
from .translit.noise import CharPlausibility
from .translit.tokens import NATIVE_WORD, words

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def human_pairwise_disagreement(lex: Lexicon, min_attestations: int = 3) -> float:
    """P(two different annotators spell a word differently), averaged over words."""
    rates = []
    for counts in lex.entries.values():
        n = sum(counts.values())
        if n >= min_attestations:
            same = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
            rates.append(1 - same)
    return sum(rates) / len(rates)


def flores_stats(lang: str) -> dict:
    lex = Lexicon.load(lang)
    plaus = CharPlausibility(lex)
    rows = [r for split in ("dev", "devtest") for r in _jsonl(DERIVED / f"flores_{lang}_{split}.jsonl")]

    words_total = 0
    l3_vs_lexical = l3_vs_canonical = writer_pairs = writer_differs = 0
    implausible = l3_tokens = 0
    xlit_hits = xlit_modal = xlit_scored = 0
    sentence_div = []
    for r in rows:
        native = [w for w in words(r["l1_native"]) if NATIVE_WORD[lang].search(w)]
        words_total += len(native)
        lexical = words(r["l2_lexical"])
        canonical = words(r["l2_canonical"])
        writers = [words(w) for w in r["l3_writers"]]
        aligned = all(len(w) == len(lexical) for w in writers)
        if aligned:
            for w in writers:
                l3_vs_lexical += sum(a != b for a, b in zip(w, lexical))
                l3_tokens += len(w)
                implausible += sum(plaus.unseen(t) > 0 for t in w)
            if len(canonical) == len(lexical):
                l3_vs_canonical += sum(a != b for w in writers for a, b in zip(w, canonical))
            for a, b in itertools.combinations(writers, 2):
                writer_pairs += len(a)
                writer_differs += sum(x != y for x, y in zip(a, b))
        sentence_div += [normalized_edit_distance(a, b)
                         for a, b in itertools.combinations(r["l3_writers"], 2)]
        # How human-like is IndicXlit on words humans attested often enough to judge?
        if len(canonical) == len(native) == len(lexical):
            for nat, xl in zip(native, canonical):
                counts = lex.entries.get(nat)
                if counts and sum(counts.values()) >= 3:
                    xlit_scored += 1
                    xlit_hits += xl in counts
                    xlit_modal += xl == max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    return {
        "sentences": len(rows),
        "native_word_tokens": words_total,
        "human_attested_word_share": round(sum(r["human_coverage"] for r in rows) / len(rows), 4),
        "with_aksharantar_word_share": round(sum(r["lexical_coverage"] for r in rows) / len(rows), 4),
        "l3_word_differs_from_l2_lexical": round(l3_vs_lexical / l3_tokens, 4),
        "l3_word_differs_from_l2_canonical": round(l3_vs_canonical / l3_tokens, 4),
        "l3_writer_pair_word_disagreement": round(writer_differs / writer_pairs, 4),
        "human_annotator_pair_word_disagreement": round(human_pairwise_disagreement(lex), 4),
        "l3_writer_pair_sentence_ned": round(sum(sentence_div) / len(sentence_div), 4),
        "l3_implausible_token_rate": round(implausible / l3_tokens, 4),
        "indicxlit_is_attested_human_spelling": round(xlit_hits / xlit_scored, 4),
        "indicxlit_is_modal_human_spelling": round(xlit_modal / xlit_scored, 4),
        "lexicon_pairs_rejected_as_misaligned": lex.rejected,
    }


def dcm_stats(lang: str) -> dict:
    out = {}
    for split in ("train", "dev", "test"):
        rows = _jsonl(DERIVED / f"dcm_{lang}_{split}.jsonl")
        latin_in_lang = [r for r in rows if r["script"] == "latin" and r["in_language"]]
        out[split] = {
            "comments": len(rows),
            "script": dict(Counter(r["script"] for r in rows)),
            "labels": dict(Counter(r["label_name"] for r in rows)),
            "offensive_share": round(sum(r["offensive"] for r in rows) / len(rows), 4),
            "latin_in_language": len(latin_in_lang),
            "latin_in_language_offensive": sum(r["offensive"] for r in latin_in_lang),
            "collapsed_duplicate_groups": sum(r["copies"] > 1 for r in rows),
            "label_conflicts": sum(r["label_conflict"] for r in rows),
            "with_l1_backtranslit": sum(r["l1_backtranslit"] is not None for r in rows),
        }
    return out


def main() -> None:
    stats = {"flores": {l: flores_stats(l) for l in ("ml", "ta")},
             "dravidiancodemix": {l: dcm_stats(l) for l in ("ml", "ta")}}
    path = ROOT / "docs" / "corpus_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats["flores"], indent=1))
    for l in ("ml", "ta"):
        t = stats["dravidiancodemix"][l]["test"]
        print(l, "dcm test:", {k: t[k] for k in ("comments", "latin_in_language", "latin_in_language_offensive", "with_l1_backtranslit")})


if __name__ == "__main__":
    main()
