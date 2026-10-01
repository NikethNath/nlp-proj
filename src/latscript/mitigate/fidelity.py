"""How faithful the input rewrites are, independent of any downstream model.

* Back-transliteration: the share of words that come back exactly as the
  native-script original, and the character-level distance of the whole
  sentence from it. If inconsistent spelling makes IndicXlit reconstruct
  worse, L3 recovers fewer words than L2.
* Round-trip: whether normalization removes spelling variation. Two measures
  on aligned words, before and after the round-trip:
  - how often two L3 writers spell a word differently (Phase 1 reported
    about 43% before);
  - how often an L3 writer differs from L2 canonical.

    PYTHONPATH=src python -m latscript.mitigate.fidelity
"""
from __future__ import annotations

import itertools
import unicodedata

from ..eval.conditions import LANGS, N_WRITERS, load_rows, texts, write_json, results_dir
from ..translit.distance import normalized_edit_distance
from ..translit.tokens import NATIVE_WORD, words


def _native_words(text: str, lang: str) -> list[str]:
    return [unicodedata.normalize("NFC", w) for w in words(text) if NATIVE_WORD[lang].search(w)]


def _lower_words(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


def _differ(a: list[str], b: list[str]) -> tuple[int, int]:
    """(differing positions, positions) when a and b align word for word, else (0, 0)."""
    if len(a) != len(b):
        return 0, 0
    return sum(x != y for x, y in zip(a, b)), len(a)


def evaluate(lang: str) -> dict:
    rows = load_rows(lang, variant="bt")
    rt_rows = {r["id"]: texts(r, "rt") for r in load_rows(lang, variant="rt")}
    levels = {"l2_canonical": ["l2_canonical"], "l2_lexical": ["l2_lexical"],
              "l3": [f"l3_{i}" for i in range(N_WRITERS)]}

    bt = {lv: {"hit": 0, "words": 0, "ned": []} for lv in levels}
    pair = {"before": [0, 0], "after": [0, 0]}
    to_l2 = {"before": [0, 0], "after": [0, 0]}
    for r in rows:
        native = _native_words(r["l1_native"], lang)
        orig, back, rt = texts(r), texts(r, "bt"), rt_rows[r["id"]]
        for lv, conds in levels.items():
            for c in conds:
                d, n = _differ(_native_words(back[c], lang), native)
                bt[lv]["hit"] += n - d
                bt[lv]["words"] += n
                bt[lv]["ned"].append(normalized_edit_distance(back[c], r["l1_native"]))
        writers = [f"l3_{i}" for i in range(N_WRITERS)]
        for stage, src in (("before", orig), ("after", rt)):
            for a, b in itertools.combinations(writers, 2):
                d, n = _differ(_lower_words(src[a]), _lower_words(src[b]))
                pair[stage][0] += d; pair[stage][1] += n
            for w in writers:
                d, n = _differ(_lower_words(src[w]), _lower_words(src["l2_canonical"]))
                to_l2[stage][0] += d; to_l2[stage][1] += n

    def rate(x):
        return round(x[0] / x[1], 4) if x[1] else None

    return {
        "lang": lang,
        "backtranslit": {lv: {"word_accuracy": round(v["hit"] / v["words"], 4) if v["words"] else None,
                              "aligned_words": v["words"],
                              "sentence_ned_to_l1": round(sum(v["ned"]) / len(v["ned"]), 4)}
                         for lv, v in bt.items()},
        "roundtrip": {"l3_writer_pair_word_disagreement": {k: rate(v) for k, v in pair.items()},
                      "l3_vs_l2_canonical_word_disagreement": {k: rate(v) for k, v in to_l2.items()}},
    }


def main() -> None:
    for lang in LANGS:
        res = evaluate(lang)
        write_json(results_dir("fidelity", True) / f"rewrites_{lang}.json", res)
        bt, rt = res["backtranslit"], res["roundtrip"]
        print(f"{lang} back-transliteration word accuracy  "
              + "  ".join(f"{lv} {v['word_accuracy']:.3f}" for lv, v in bt.items()))
        print(f"{lang} round-trip: writer-pair disagreement "
              f"{rt['l3_writer_pair_word_disagreement']['before']:.3f} -> "
              f"{rt['l3_writer_pair_word_disagreement']['after']:.3f}; L3 vs L2 canonical "
              f"{rt['l3_vs_l2_canonical_word_disagreement']['before']:.3f} -> "
              f"{rt['l3_vs_l2_canonical_word_disagreement']['after']:.3f}")


if __name__ == "__main__":
    main()
