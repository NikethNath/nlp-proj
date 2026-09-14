"""Inter-writer spelling divergence in the human-typed subset.

Reports, per language:

* anchor divergence - mean pairwise normalized edit distance between
  respondents who typed the same sentence;
* word disagreement - for pairs of responses with the same word count as the
  native sentence (so words align by position), the share of word positions
  spelled differently. Comparable to Dakshina's word-level figure (two
  annotators disagree on ~31% of Malayalam and ~37% of Tamil words);
* the same two numbers for the simulated L3 writers on the same sentences,
  which is the direct test of whether L3 is realistic;
* distance from real typing to each L2, i.e. which canonical form real people
  are closer to;
* per-respondent summaries.

Responses flagged native_script, length, translated or no_consent are excluded.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from ..translit.distance import normalized_edit_distance
from ..translit.tokens import NATIVE_WORD, words
from .ingest import DEFAULT_OUT, Response, read

ROOT = Path(__file__).resolve().parents[3]
EXCLUDE = {"native_script", "length", "translated", "no_consent"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _mean(xs):
    return round(statistics.mean(xs), 4) if xs else None


def pairwise(strings: list[str], native: str, lang: str) -> tuple[list[float], list[bool]]:
    """(sentence-level distances, word-level differ? flags) over all pairs."""
    normed = [normalize(s) for s in strings]
    n_words = sum(1 for w in words(native) if NATIVE_WORD[lang].search(w))
    dists, differs = [], []
    for a, b in itertools.combinations(normed, 2):
        dists.append(normalized_edit_distance(a, b))
        ta, tb = a.split(), b.split()
        if len(ta) == len(tb) == n_words:
            differs.extend(x != y for x, y in zip(ta, tb))
    return dists, differs


def analyze(responses: list[Response], flores_rows: dict[int, dict] | None = None) -> dict:
    kept = [r for r in responses if not EXCLUDE & set(r.flags)]
    lang = kept[0].lang if kept else (responses[0].lang if responses else None)
    by_sentence: dict[int, list[Response]] = defaultdict(list)
    for r in kept:
        by_sentence[r.idx].append(r)

    human_d, human_w, sim_d, sim_w = [], [], [], []
    per_sentence = {}
    for idx, rs in sorted(by_sentence.items()):
        writers = {r.respondent: r.typed for r in rs}      # one answer per person
        if len(writers) < 2:
            continue
        d, w = pairwise(list(writers.values()), rs[0].native, lang)
        human_d += d; human_w += w
        entry = {"writers": len(writers), "divergence": _mean(d)}
        if flores_rows and idx in flores_rows:
            sd, sw = pairwise(flores_rows[idx]["l3_writers"], rs[0].native, lang)
            sim_d += sd; sim_w += sw
            entry["simulated_divergence"] = _mean(sd)
        per_sentence[idx] = entry

    to_l2 = defaultdict(list)
    if flores_rows:
        for r in kept:
            row = flores_rows.get(r.idx)
            if row:
                for key in ("l2_canonical", "l2_lexical"):
                    if row.get(key):
                        to_l2[key].append(normalized_edit_distance(normalize(r.typed), normalize(row[key])))

    per_respondent = {}
    for code in sorted({r.respondent for r in responses}):
        mine = [r for r in responses if r.respondent == code]
        typed = " ".join(normalize(r.typed) for r in mine if not EXCLUDE & set(r.flags))
        tokens = typed.split()
        per_respondent[code] = {
            "sentences": len(mine),
            "excluded": sum(bool(EXCLUDE & set(r.flags)) for r in mine),
            "long_vowel_digraph_rate": _mean([bool(re.search(r"aa|ee|ii|oo|uu", t)) for t in tokens]),
            "background": next((r.background for r in mine if r.background), {}),
        }

    return {
        "lang": lang,
        "responses": len(responses),
        "kept": len(kept),
        "respondents": len({r.respondent for r in kept}),
        "sentences_typed": len(by_sentence),
        "sentences_with_2plus_writers": len(per_sentence),
        "human": {"sentence_divergence": _mean(human_d), "word_disagreement": _mean(human_w),
                  "word_pairs": len(human_w)},
        "simulated_l3": {"sentence_divergence": _mean(sim_d), "word_disagreement": _mean(sim_w),
                         "word_pairs": len(sim_w)},
        "distance_to": {k: _mean(v) for k, v in to_l2.items()},
        "per_sentence": per_sentence,
        "per_respondent": per_respondent,
    }


def load_flores_rows(lang: str, split: str = "devtest") -> dict[int, dict]:
    path = ROOT / "data" / "derived" / f"flores_{lang}_{split}.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["idx"]: row for row in map(json.loads, fh)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Analyze human romanization responses.")
    ap.add_argument("lang", choices=["ml", "ta"])
    ap.add_argument("--responses", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    responses = read(args.responses or DEFAULT_OUT / f"responses_{args.lang}.jsonl")
    report = analyze(responses, load_flores_rows(args.lang))
    out = args.out or DEFAULT_OUT / f"analysis_{args.lang}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    h, s = report["human"], report["simulated_l3"]
    print(f"{out}: {report['kept']}/{report['responses']} responses kept, "
          f"{report['respondents']} respondents")
    print(f"  word disagreement  human {h['word_disagreement']}  simulated {s['word_disagreement']}")
    print(f"  sentence divergence human {h['sentence_divergence']}  simulated {s['sentence_divergence']}")
    print(f"  distance to L2: {report['distance_to']}")


if __name__ == "__main__":
    main()
