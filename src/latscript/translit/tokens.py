"""Word tokenization shared by every orthography transform.

Python's ``\\w`` excludes combining marks, so splitting on ``\\W`` cuts
Malayalam and Tamil words apart at every vowel sign and virama. The two script
blocks, and the zero-width joiners used to form Malayalam chillus, are listed
as word characters explicitly.
"""
from __future__ import annotations

import re
from typing import Callable

_SCRIPT_BLOCKS = "\\u0B80-\\u0BFF\\u0D00-\\u0D7F\\u200c\\u200d"

# Capturing split: odd slots are separators, even slots are words (or empty).
SPLIT = re.compile(f"([^\\w{_SCRIPT_BLOCKS}]+)")

NATIVE_WORD = {
    "ml": re.compile("[\\u0D00-\\u0D7F\\u200c\\u200d]+"),
    "ta": re.compile("[\\u0B80-\\u0BFF]+"),
}
LATIN_WORD = re.compile("[A-Za-z]+")


def is_separator(part: str) -> bool:
    return bool(part) and SPLIT.fullmatch(part) is not None


def words(text: str) -> list[str]:
    return [p for p in SPLIT.split(text) if p and not is_separator(p)]


def map_words(text: str, fn: Callable[[str], str]) -> str:
    """Apply `fn` to every word, leaving whitespace and punctuation untouched."""
    return "".join(p if not p or is_separator(p) else fn(p) for p in SPLIT.split(text))


def map_matches(text: str, pattern: re.Pattern, fn: Callable[[str], str]) -> str:
    """Apply `fn` to every match of `pattern`, e.g. only native-script runs."""
    return pattern.sub(lambda m: fn(m.group(0)), text)
