"""DravidianCodeMix offensive-language data - the ecological arm.

Real YouTube comments, mostly typed in Latin script, labelled for offensive
language (DravidianLangTech-EACL 2021). Unlike FLORES nothing here is
manufactured: the romanization is whatever the commenter typed.

Sources, both on the Hugging Face Hub:

* Tamil  - IsaacRodgz/DravidianCodeMix-Dataset: train/dev/test, test labelled.
* Malayalam - community-datasets/offenseval_dravidian: train/validation only.
  The labelled test set was never published, so validation serves as test and
  a stratified 10% of train becomes dev.

Cleaning, which the raw splits badly need:

* 944 of the 1,999 Malayalam validation comments also occur in train, and 27%
  of Malayalam train is exact duplicates. Anything in an evaluation split is
  removed from the splits used for training, and duplicates within a split are
  collapsed. Without this a model fine-tuned on train is partly tested on its
  own training data.
* When copies of one comment carry different labels, the majority label is
  kept and ``label_conflict`` is set. That is the only annotator-agreement
  signal the dataset offers; there are no per-annotator labels.
* Each comment is tagged ``latin`` / ``native`` / ``mixed`` by the share of
  Latin letters among letters. A quarter of Malayalam comments are written in
  native script, and they are not the L3 condition.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "dravidiancodemix"

# Label order used by offenseval_dravidian. The Isaac parquet files encode the
# same classes in a different order, remapped below.
LABELS = (
    "not_offensive",
    "offensive_untargeted",
    "offensive_targeted_individual",
    "offensive_targeted_group",
    "offensive_targeted_other",
    "not_in_language",
)
_ISAAC_TO_CANONICAL = {0: 0, 1: 5, 2: 4, 3: 3, 4: 1, 5: 2}

_FILES = {
    "ta": {"train": "isaac_train-00000-of-00001.parquet",
           "dev": "isaac_dev-00000-of-00001.parquet",
           "test": "isaac_test-00000-of-00001.parquet"},
    "ml": {"train": "offenseval_malayalam_train.parquet",
           "test": "offenseval_malayalam_validation.parquet"},
}

_NATIVE_LETTER = {"ml": re.compile("[ഀ-ൿ]"), "ta": re.compile("[஀-௿]")}
_LATIN_LETTER = re.compile("[A-Za-z]")
_WS = re.compile(r"\s+")


@dataclass
class Comment:
    id: str
    lang: str
    split: str
    text: str
    label: int
    label_name: str
    offensive: bool
    in_language: bool
    script: str
    latin_ratio: float
    label_conflict: bool
    copies: int


def script_of(text: str, lang: str, threshold: float = 0.9) -> tuple[str, float]:
    native = len(_NATIVE_LETTER[lang].findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if native + latin == 0:
        return "none", 0.0
    ratio = latin / (native + latin)
    if ratio >= threshold:
        return "latin", ratio
    if ratio <= 1 - threshold:
        return "native", ratio
    return "mixed", ratio


def _read(lang: str, split: str, root: Path) -> list[tuple[str, int]]:
    import pandas as pd

    path = root / _FILES[lang][split]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/fetch_data.sh")
    df = pd.read_parquet(path)
    labels = df["label"].astype(int)
    if lang == "ta":
        labels = labels.map(_ISAAC_TO_CANONICAL)
    return list(zip(df["text"].astype(str), labels))


def _key(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _collapse(rows: list[tuple[str, int]]) -> dict[str, tuple[str, int, bool, int]]:
    """key -> (first text, majority label, conflict?, copies)."""
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for text, label in rows:
        if _key(text):
            grouped[_key(text)].append((text, label))
    out = {}
    for key, items in grouped.items():
        counts = Counter(label for _, label in items)
        # Majority label; ties go to the lower id, i.e. toward not_offensive.
        label = min(counts, key=lambda l: (-counts[l], l))
        out[key] = (items[0][0], label, len(counts) > 1, len(items))
    return out


def load(lang: str, root: Path | None = None, seed: int = 0,
         dev_fraction: float = 0.1) -> dict[str, list[Comment]]:
    """Cleaned train/dev/test splits for `lang`."""
    if lang not in _FILES:
        raise ValueError(f"lang must be 'ml' or 'ta', got {lang!r}")
    root = root or ROOT

    raw = {split: _collapse(_read(lang, split, root)) for split in _FILES[lang]}

    if lang == "ml":
        # Carve dev out of train, stratified by label, after removing test copies.
        test_keys = set(raw["test"])
        train = {k: v for k, v in raw["train"].items() if k not in test_keys}
        by_label: dict[int, list[str]] = defaultdict(list)
        for key, (_, label, _, _) in train.items():
            by_label[label].append(key)
        rng = random.Random(seed)
        dev_keys = set()
        for label in sorted(by_label):
            keys = sorted(by_label[label])
            rng.shuffle(keys)
            dev_keys.update(keys[:round(len(keys) * dev_fraction)])
        raw = {"train": {k: v for k, v in train.items() if k not in dev_keys},
               "dev": {k: v for k, v in train.items() if k in dev_keys},
               "test": raw["test"]}
    else:
        test_keys = set(raw["test"])
        raw["dev"] = {k: v for k, v in raw["dev"].items() if k not in test_keys}
        held_out = test_keys | set(raw["dev"])
        raw["train"] = {k: v for k, v in raw["train"].items() if k not in held_out}

    splits: dict[str, list[Comment]] = {}
    for split in ("train", "dev", "test"):
        comments = []
        for i, key in enumerate(sorted(raw[split])):
            text, label, conflict, copies = raw[split][key]
            script, ratio = script_of(text, lang)
            comments.append(Comment(
                id=f"dcm-{lang}-{split}-{i:05d}", lang=lang, split=split,
                text=text, label=label, label_name=LABELS[label],
                offensive=label in (1, 2, 3, 4), in_language=label != 5,
                script=script, latin_ratio=round(ratio, 4),
                label_conflict=conflict, copies=copies,
            ))
        splits[split] = comments
    return splits


def build(langs=("ml", "ta"), backtranslit_splits=("dev", "test"),
          out: Path | None = None) -> None:
    """Write cleaned splits, with IndicXlit back-transliteration as L1.

    Only Latin and mixed comments get an ``l1_backtranslit``: they are the L3
    condition, and back-transliteration is how their native-script counterpart
    is obtained. English words inside a comment are nativized too, which is
    how a native-script typist usually writes loanwords, but whole English or
    Hindi sentences come out as noise (most carry the ``not_in_language``
    label). Train is written without it by default; nothing downstream needs a
    native-script training set.
    """
    import json
    from dataclasses import asdict

    from ..translit import indicxlit
    from ..translit.tokens import LATIN_WORD

    out = out or ROOT.parents[1] / "derived"
    out.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        splits = load(lang)
        needs = [c for s in backtranslit_splits for c in splits[s]
                 if c.script in ("latin", "mixed")]
        vocab = {m.lower() for c in needs for m in LATIN_WORD.findall(c.text)}
        lookup = indicxlit.table(vocab, lang, "en2indic") if vocab else {}
        for split, comments in splits.items():
            path = out / f"dcm_{lang}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for c in comments:
                    row = asdict(c)
                    row["l1_backtranslit"] = (
                        indicxlit.nativize(c.text, lang, lookup)
                        if split in backtranslit_splits and c.script in ("latin", "mixed")
                        else None)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            latin = sum(c.script == "latin" for c in comments)
            print(f"  {path.name}: {len(comments):,} comments, {latin:,} in Latin script")


if __name__ == "__main__":
    build()
