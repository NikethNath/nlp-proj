"""IndicXlit transliteration, run out-of-process with an on-disk cache.

IndicXlit needs fairseq 0.12 and therefore Python 3.10 (see
scripts/setup_indicxlit.sh), so it cannot be imported here. Instead the unique
words of a corpus are sent to scripts/indicxlit_worker.py in that environment
once, and every result is kept in ``data/cache/indicxlit``. Rebuilding the
corpus afterwards never touches the model.

Two directions are used:

* ``indic2en`` - native -> Latin, the canonical L2 of the FLORES arm.
* ``en2indic`` - Latin -> native, the back-transliterated L1 of the
  DravidianCodeMix arm (and later the back-transliteration mitigation).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Iterable

from .tokens import LATIN_WORD, NATIVE_WORD, map_matches

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "cache" / "indicxlit"
WORKER = ROOT / "scripts" / "indicxlit_worker.py"
PYTHON = Path(os.environ.get("INDICXLIT_PYTHON", ROOT / ".venv-xlit" / "bin" / "python"))


def _cache_path(direction: str, lang: str) -> Path:
    return CACHE / f"{direction}_{lang}.tsv"


def _read_cache(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                word, _, out = line.rstrip("\n").partition("\t")
                if word:
                    table[word] = out
    return table


def table(words: Iterable[str], lang: str, direction: str) -> dict[str, str]:
    """Transliterate `words`, calling the model only for uncached ones."""
    if direction not in ("indic2en", "en2indic"):
        raise ValueError(f"unknown direction {direction!r}")
    path = _cache_path(direction, lang)
    cached = _read_cache(path)
    wanted = {unicodedata.normalize("NFC", w) for w in words if w}
    if direction == "en2indic":
        wanted = {w.lower() for w in wanted}
    missing = sorted(wanted - cached.keys())

    if missing:
        if not PYTHON.exists():
            raise RuntimeError(
                f"{len(missing)} words need IndicXlit but {PYTHON} is missing; "
                "run scripts/setup_indicxlit.sh")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as tmp:
            tmp.write("\n".join(missing) + "\n")
        try:
            subprocess.run(
                [str(PYTHON), str(WORKER), "--direction", direction, "--lang", lang,
                 "--input", tmp.name, "--output", str(path)],
                check=True)
        finally:
            os.unlink(tmp.name)
        cached = _read_cache(path)

    return {w: cached[w] for w in wanted if cached.get(w)}


def romanize(text: str, lang: str, lookup: dict[str, str]) -> str:
    """Replace native-script runs with their IndicXlit romanization."""
    text = unicodedata.normalize("NFC", text)
    return map_matches(text, NATIVE_WORD[lang], lambda w: lookup.get(w, w))


def nativize(text: str, lang: str, lookup: dict[str, str]) -> str:
    """Replace Latin-script runs with their IndicXlit native-script form."""
    return map_matches(text, LATIN_WORD, lambda w: lookup.get(w.lower(), w))
