"""Aksharantar word pairs, used as a fallback canonical spelling.

97.6% of Aksharantar's Malayalam and Tamil pairs come from IndicCorp and carry
a model score: they are clean, consistent, single romanizations, not a record
of how different people spell a word. So they fill coverage gaps in the
lexical L2 but are never used to learn or sample spelling variation.

The raw release is a 700 MB JSON-lines file per language. scripts/fetch_data.sh
flattens it once into ``data/raw/aksharantar/{lang}.tsv`` (native, roman,
source), and lookups stream that file keeping only the requested words.
"""
from __future__ import annotations

import json
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "aksharantar"
_CODES = {"ml": "mal", "ta": "tam"}


def flatten(lang: str, root: Path | None = None) -> Path:
    """Write the compact TSV from the downloaded zip, once."""
    root = root or ROOT
    out = root / f"{lang}.tsv"
    if out.exists():
        return out
    code = _CODES[lang]
    seen: set[tuple[str, str]] = set()
    tmp = out.with_suffix(".tsv.part")
    with zipfile.ZipFile(root / f"{code}.zip") as zf, tmp.open("w", encoding="utf-8") as fh:
        for split in ("train", "valid", "test"):
            with zf.open(f"{code}_{split}.json") as src:
                for line in src:
                    row = json.loads(line)
                    native = row["native word"].strip()
                    roman = row["english word"].strip().lower()
                    if native and roman and (native, roman) not in seen:
                        seen.add((native, roman))
                        fh.write(f"{native}\t{roman}\t{row['source']}\n")
    tmp.rename(out)
    return out


def lookup(words: Iterable[str], lang: str, root: Path | None = None) -> dict[str, str]:
    """First Aksharantar romanization for each requested native word."""
    wanted = {unicodedata.normalize("NFC", w) for w in words}
    path = (root or ROOT) / f"{lang}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/fetch_data.sh")
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            native, _, rest = line.partition("\t")
            if native in wanted and native not in found:
                found[native] = rest.split("\t", 1)[0]
    return found
