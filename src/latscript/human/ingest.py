"""Turn a Google Forms CSV export into one record per typed sentence.

Download the responses from the linked sheet (File -> Download -> CSV) into
``data/raw/human/``, then::

    PYTHONPATH=src python -m latscript.human.ingest ml data/raw/human/ml.csv

Answers are matched to sentences through the ``[ml-0123]`` prefix in each
column header, never through column position. Nothing is silently dropped:
suspect answers are kept and flagged, and analysis decides what to exclude.

Flags
    native_script   typed mostly in Malayalam/Tamil letters (pasted, or an IME)
    length          far shorter or longer than a romanization of the sentence
    translated      mostly words from the English reference: translated, not typed
    no_consent      consent box not ticked (the form requires it; defensive)
    repeat_code     the respondent code appears on more than one submission
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..corpora import flores
from ..translit.rules import transliterate
from .forms import BACKGROUND, CODE_TITLE, CONSENT_CHOICE, CONSENT_TITLE, LANG_NAME, SET_TITLE
from .select import DEFAULT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "data" / "derived" / "human"

_HEADER = re.compile(r"^\[(ml|ta)-(\d{4})\]")
_NATIVE = {"ml": re.compile("[ഀ-ൿ]"), "ta": re.compile("[஀-௿]")}
_LATIN = re.compile("[A-Za-z]")
_WORD = re.compile("[a-z]+")


@dataclass
class Response:
    lang: str
    idx: int
    role: str                 # "anchor" or "block"
    block: int | None         # 1-based set number for block sentences
    respondent: str
    submission: int           # row number in the export, 1-based
    timestamp: str
    native: str
    typed: str
    background: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _normalize_code(code: str) -> str:
    return re.sub(r"\s+", "", code).upper()


def _flags(typed: str, native: str, english: str, lang: str) -> list[str]:
    flags = []
    native_letters = len(_NATIVE[lang].findall(typed))
    latin_letters = len(_LATIN.findall(typed))
    if native_letters > 0.1 * max(native_letters + latin_letters, 1):
        flags.append("native_script")
    reference = re.sub(r"[^a-z]", "", transliterate(native, lang).lower())
    typed_letters = re.sub(r"[^a-z]", "", typed.lower())
    if reference and not 0.5 <= len(typed_letters) / len(reference) <= 1.8:
        flags.append("length")
    typed_words = [w for w in _WORD.findall(typed.lower()) if len(w) >= 3]
    english_words = {w for w in _WORD.findall(english.lower()) if len(w) >= 3}
    if typed_words and sum(w in english_words for w in typed_words) / len(typed_words) >= 0.5:
        flags.append("translated")
    return flags


def ingest(csv_path: Path, lang: str, manifest: dict | None = None,
           english: dict[int, str] | None = None) -> list[Response]:
    manifest = manifest or load_manifest(DEFAULT_MANIFEST)
    if english is None:
        segs = flores.load(lang, manifest["split"])
        english = {s.idx: s.english for s in segs}
    anchors = set(manifest["anchors"])
    block_of = {idx: b + 1 for b, block in enumerate(manifest["blocks"]) for idx in block}
    background_titles = {key: title.format(lang=LANG_NAME[lang])
                         for key, (title, _) in BACKGROUND.items()}

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []

    sentence_cols = {}
    for col in rows[0].keys():
        m = _HEADER.match(col or "")
        if m and m.group(1) == lang:
            sentence_cols[col] = int(m.group(2))
    if not sentence_cols:
        raise ValueError(f"{csv_path}: no [{lang}-NNNN] columns; wrong language or file?")
    unknown = sorted(i for i in sentence_cols.values() if i not in anchors and i not in block_of)
    if unknown:
        raise ValueError(f"{csv_path}: columns for sentences not in the manifest: {unknown[:5]}")

    codes = Counter(_normalize_code(r.get(CODE_TITLE, "")) for r in rows)
    out: list[Response] = []
    for n, row in enumerate(rows, 1):
        code = _normalize_code(row.get(CODE_TITLE, "")) or f"ANON{n:03d}"
        background = {k: row.get(t, "").strip() for k, t in background_titles.items() if row.get(t)}
        chosen_set = row.get(SET_TITLE, "").strip()
        shared = []
        if row.get(CONSENT_TITLE, "").strip() != CONSENT_CHOICE:
            shared.append("no_consent")
        if codes[_normalize_code(row.get(CODE_TITLE, ""))] > 1:
            shared.append("repeat_code")
        for col, idx in sentence_cols.items():
            typed = (row.get(col) or "").strip()
            if not typed:
                continue
            native = manifest["sentences"][lang][str(idx)]
            role = "anchor" if idx in anchors else "block"
            flags = shared + _flags(typed, native, english.get(idx, ""), lang)
            if role == "block" and chosen_set and chosen_set != str(block_of[idx]):
                flags.append("set_mismatch")
            out.append(Response(
                lang=lang, idx=idx, role=role,
                block=block_of.get(idx), respondent=code, submission=n,
                timestamp=row.get("Timestamp", ""), native=native, typed=typed,
                background=background, flags=flags,
            ))
    return out


def write(responses: list[Response], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in responses:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return path


def read(path: Path) -> list[Response]:
    with path.open(encoding="utf-8") as fh:
        return [Response(**json.loads(line)) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest a Google Forms CSV export.")
    ap.add_argument("lang", choices=["ml", "ta"])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    responses = ingest(args.csv, args.lang, load_manifest(args.manifest))
    path = write(responses, args.out or DEFAULT_OUT / f"responses_{args.lang}.jsonl")
    flags = Counter(f for r in responses for f in r.flags)
    people = len({r.respondent for r in responses})
    print(f"{path}: {len(responses)} typed sentences from {people} respondents; flags {dict(flags)}")


if __name__ == "__main__":
    main()
