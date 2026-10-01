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
    _merge_shards(path)
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
        _run_sharded(missing, lang, direction, path)
        cached = _read_cache(path)

    return {w: cached[w] for w in wanted if cached.get(w)}


def _run_sharded(missing: list[str], lang: str, direction: str, path: Path) -> None:
    """Run the worker as several processes and fold their output into the cache.

    One beam-search process scales poorly past a few threads, so large jobs
    are split into INDICXLIT_PROCS shards (default 4) sharing the CPU cores.
    Each shard appends to its own file, so an interrupted run loses nothing:
    leftover shard files are merged into the cache on the next call.
    """
    procs = max(1, min(int(os.environ.get("INDICXLIT_PROCS", 4)), len(missing) // 1000 or 1))
    threads = max(1, (os.cpu_count() or 1) // procs)
    outputs = [path.with_name(f"{path.stem}.shard{i}.tsv") for i in range(procs)]
    inputs, running = [], []
    try:
        for i, out in enumerate(outputs):
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                             encoding="utf-8") as tmp:
                tmp.write("\n".join(missing[i::procs]) + "\n")
            inputs.append(tmp.name)
            running.append(subprocess.Popen(
                [str(PYTHON), str(WORKER), "--direction", direction, "--lang", lang,
                 "--input", tmp.name, "--output", str(out), "--threads", str(threads)]))
        codes = [p.wait() for p in running]
    finally:
        for p in running:
            if p.poll() is None:
                p.kill()
        for name in inputs:
            os.unlink(name)
        _merge_shards(path)
    if any(codes):
        raise RuntimeError(f"IndicXlit worker failed with exit codes {codes}")


def _merge_shards(path: Path) -> None:
    """Append leftover per-shard outputs to the cache file and remove them."""
    for shard in sorted(path.parent.glob(f"{path.stem}.shard*.tsv")):
        with shard.open(encoding="utf-8") as src, path.open("a", encoding="utf-8") as dst:
            dst.write(src.read())
        shard.unlink()


def romanize(text: str, lang: str, lookup: dict[str, str]) -> str:
    """Replace native-script runs with their IndicXlit romanization."""
    text = unicodedata.normalize("NFC", text)
    return map_matches(text, NATIVE_WORD[lang], lambda w: lookup.get(w, w))


def nativize(text: str, lang: str, lookup: dict[str, str]) -> str:
    """Replace Latin-script runs with their IndicXlit native-script form."""
    return map_matches(text, LATIN_WORD, lambda w: lookup.get(w.lower(), w))
