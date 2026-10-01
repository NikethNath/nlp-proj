"""The orthographic conditions every Phase 2 evaluation is run under.

A FLORES row becomes one input per condition, all carrying the same meaning:

    l1            native script
    l2_canonical  IndicXlit romanization
    l2_lexical    most common human spelling
    l3_0..l3_4    the five simulated writers

Scores for L3 are always averaged over the five writers of a sentence before
anything else is done with them, so each sentence contributes one L3 number
and writer-to-writer spread is reported separately.

Evaluation uses FLORES devtest. Dev is left for Phase 3, where mitigations may
need tuning. The *core* subset is the 248 sentences in the human collection
manifest: slower models run only on it, and it is where human-typed L3 will
join the comparison once responses are in.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DERIVED = ROOT / "data" / "derived"
EVAL_OUT = ROOT / "data" / "eval"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "collection" / "manifest.json"

LANGS = ("ml", "ta")
LANG_NAME = {"ml": "Malayalam", "ta": "Tamil"}
N_WRITERS = 5
CONDITIONS = ("l1", "l2_canonical", "l2_lexical") + tuple(f"l3_{i}" for i in range(N_WRITERS))
LEVELS = ("l1", "l2_canonical", "l2_lexical", "l3")


def level(condition: str) -> str:
    return "l3" if condition.startswith("l3_") else condition


def texts(row: dict) -> dict[str, str]:
    """Condition -> input text for one FLORES row."""
    out = {"l1": row["l1_native"], "l2_canonical": row["l2_canonical"],
           "l2_lexical": row["l2_lexical"]}
    for i, t in enumerate(row["l3_writers"][:N_WRITERS]):
        out[f"l3_{i}"] = t
    return out


def core_indices(path: Path = MANIFEST) -> list[int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return sorted(set(manifest["anchors"]) | {i for block in manifest["blocks"] for i in block})


def load_rows(lang: str, split: str = "devtest", subset: str = "all") -> list[dict]:
    """FLORES rows in index order; `subset` is 'all', 'core', or an integer limit."""
    path = DERIVED / f"flores_{lang}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; build the corpus first (see README)")
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if subset == "core":
        keep = set(core_indices())
        rows = [r for r in rows if r["idx"] in keep]
    elif subset != "all":
        rows = rows[:int(subset)]
    return rows


def load_dcm(lang: str, split: str = "test") -> list[dict]:
    path = DERIVED / f"dcm_{lang}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; build the corpus first (see README)")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def slug(model: str) -> str:
    return model.split("/")[-1].replace(":", "-")
