"""Aligned FLORES-200 loading.

FLORES-200 ships as one plain-text file per language per split, with line i of
every file being the same sentence. That alignment is the whole reason this is
the controlled arm of the study: L1/L2/L3 differ only in orthography.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FLORES_CODE = {"ml": "mal_Mlym", "ta": "tam_Taml", "en": "eng_Latn"}

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "flores200_dataset"


@dataclass(frozen=True)
class Segment:
    """One FLORES line, addressable across languages and orthography levels."""

    idx: int
    split: str
    lang: str
    native: str
    english: str


def _read(root: Path, split: str, code: str) -> list[str]:
    path = root / split / f"{code}.{split}"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/fetch_data.sh to populate data/raw"
        )
    return path.read_text(encoding="utf-8").splitlines()


def load(lang: str, split: str = "devtest", root: Path | None = None,
         limit: int | None = None) -> list[Segment]:
    """Load `lang` FLORES segments paired with their English references."""
    if lang not in ("ml", "ta"):
        raise ValueError(f"lang must be 'ml' or 'ta', got {lang!r}")
    if split not in ("dev", "devtest"):
        raise ValueError(f"split must be 'dev' or 'devtest', got {split!r}")
    root = root or DEFAULT_ROOT

    src = _read(root, split, FLORES_CODE[lang])
    ref = _read(root, split, FLORES_CODE["en"])
    if len(src) != len(ref):
        raise RuntimeError(f"{lang}/{split} misaligned: {len(src)} vs {len(ref)} lines")

    segs = [
        Segment(idx=i, split=split, lang=lang, native=s.strip(), english=r.strip())
        for i, (s, r) in enumerate(zip(src, ref))
    ]
    return segs[:limit] if limit else segs
