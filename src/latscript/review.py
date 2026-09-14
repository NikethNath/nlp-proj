"""Export the 50-sentence hand check of the generated corpus.

Writes ``review/sanity_check_{lang}.csv`` (25 random FLORES devtest sentences
per language, fixed seed) for a native speaker to mark up in any spreadsheet.
See review/README.md for what to judge.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["id", "english", "l1_native", "l2_canonical", "l2_lexical",
           "l3_writer_1", "l3_writer_2", "l3_writer_3", "l3_writer_4", "l3_writer_5",
           "l2_canonical_ok", "l2_lexical_ok", "l3_bad_words", "notes"]


def export(lang: str, n: int = 25, seed: int = 50) -> Path:
    src = ROOT / "data" / "derived" / f"flores_{lang}_devtest.jsonl"
    with src.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    picked = sorted(random.Random(seed).sample(rows, n), key=lambda r: r["idx"])
    out = ROOT / "review" / f"sanity_check_{lang}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows Malayalam and Tamil instead of mojibake.
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for r in picked:
            writer.writerow([r["id"], r["english"], r["l1_native"], r["l2_canonical"],
                             r["l2_lexical"], *r["l3_writers"], "", "", "", ""])
    return out


if __name__ == "__main__":
    for lang in ("ml", "ta"):
        print(export(lang))
