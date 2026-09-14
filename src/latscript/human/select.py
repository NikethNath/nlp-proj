"""Choose the FLORES sentences native speakers will type in Latin script.

The same FLORES indices are used for Malayalam and Tamil, so the two human
subsets are parallel to each other as well as to L1/L2/L3.

Design: every respondent types the same few *anchor* sentences, then one
*block* of sentences that belongs to them alone.

* Anchors are typed by everybody, which gives many writers per sentence - the
  inter-writer divergence measurement.
* Blocks give coverage. They are dealt snake-draft by length so each takes
  about the same effort, and handed out in order: the first N respondents
  cover N blocks, and the (blocks+1)th respondent starts again at block 1,
  adding a second writer. The subset grows with however many people respond.

Sentences are restricted to 6-16 native words in both languages, with no Latin
letters, quotation marks or long numbers, to keep a phone-typed response to a
minute or two.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from ..corpora import flores
from ..translit.tokens import NATIVE_WORD, words

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "collection" / "manifest.json"
LANGS = ("ml", "ta")


def _native_len(text: str, lang: str) -> int:
    return sum(1 for w in words(text) if NATIVE_WORD[lang].search(w))


def eligible(segs: dict[str, flores.Segment], min_words: int, max_words: int) -> bool:
    for lang, seg in segs.items():
        text = seg.native
        if not min_words <= _native_len(text, lang) <= max_words:
            return False
        if re.search("[A-Za-z\"“”]", text) or len(re.findall(r"\d", text)) > 4:
            return False
    return True


def select(split: str = "devtest", anchors: int = 8, blocks: int = 20,
           block_size: int = 12, min_words: int = 6, max_words: int = 16,
           seed: int = 13) -> dict:
    by_lang = {lang: flores.load(lang, split) for lang in LANGS}
    pool = [i for i in range(len(by_lang["ml"]))
            if eligible({l: by_lang[l][i] for l in LANGS}, min_words, max_words)]
    needed = anchors + blocks * block_size
    if len(pool) < needed:
        raise ValueError(f"only {len(pool)} eligible sentences, design needs {needed}")

    rng = random.Random(seed)
    rng.shuffle(pool)
    length = {i: _native_len(by_lang["ml"][i].native, "ml") for i in pool}

    # Anchors span the length range: evenly spaced through the sorted pool.
    candidates = sorted(pool[: needed], key=lambda i: (length[i], i))
    step = len(candidates) / anchors
    anchor_ids = [candidates[int(step * (a + 0.5))] for a in range(anchors)]

    rest = sorted((i for i in pool[: needed + anchors] if i not in anchor_ids),
                  key=lambda i: (-length[i], i))[: blocks * block_size]
    dealt: list[list[int]] = [[] for _ in range(blocks)]
    for n, idx in enumerate(rest):
        rnd, pos = divmod(n, blocks)
        dealt[pos if rnd % 2 == 0 else blocks - 1 - pos].append(idx)
    for block in dealt:
        rng.shuffle(block)

    chosen = anchor_ids + [i for b in dealt for i in b]
    sentences = {lang: {str(i): by_lang[lang][i].native for i in chosen} for lang in LANGS}
    digest = hashlib.sha256(json.dumps(sentences, ensure_ascii=False, sort_keys=True)
                            .encode()).hexdigest()[:16]
    return {
        "version": 1,
        "source": f"FLORES-200 {split}",
        "split": split,
        "seed": seed,
        "criteria": {"min_native_words": min_words, "max_native_words": max_words,
                     "latin_letters": 0, "max_digits": 4},
        "eligible_pool": len(pool),
        "anchors": anchor_ids,
        "blocks": dealt,
        "sentences": sentences,
        "sentences_sha256_16": digest,
    }


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Select sentences for human romanization.")
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)
    manifest = select(anchors=args.anchors, blocks=args.blocks,
                      block_size=args.block_size, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    per_lang = len(manifest["anchors"]) + sum(map(len, manifest["blocks"]))
    print(f"{args.out}: {per_lang} sentences per language "
          f"({len(manifest['anchors'])} anchors + {len(manifest['blocks'])} blocks of "
          f"{args.block_size}), pool {manifest['eligible_pool']}")


if __name__ == "__main__":
    main()
