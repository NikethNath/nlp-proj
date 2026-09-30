"""Where multilingual sentence encoders put each orthography.

For every condition, each sentence is embedded and compared with the English
references of all sentences in the split:

* ``en_retrieval`` - is the sentence's own English reference its nearest
  neighbour (top-1 bitext retrieval accuracy)?
* ``en_cosine`` - cosine similarity to its own English reference.
* ``l1_cosine`` - cosine similarity to the native-script version of itself,
  i.e. whether the encoder recognises the romanized sentence as the same text.

    PYTHONPATH=src python -m latscript.eval.retrieval
"""
from __future__ import annotations

import argparse

import numpy as np

from .conditions import CONDITIONS, LANGS, RESULTS, load_rows, slug, texts, write_json
from .stats import as_mean_stats, decompose, mean_metric

ENCODERS = ["sentence-transformers/LaBSE", "BAAI/bge-m3"]


def evaluate(model, name: str, lang: str, rows: list[dict]) -> dict:
    def embed(items):
        return model.encode(items, batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)

    en = embed([r["english"] for r in rows])
    emb = {cond: embed([texts(r)[cond] for r in rows]) for cond in CONDITIONS}
    idx = np.arange(len(rows))
    retrieval, en_cos, l1_cos = {}, {}, {}
    for cond, e in emb.items():
        sims = e @ en.T
        retrieval[cond] = as_mean_stats(sims.argmax(1) == idx)
        en_cos[cond] = as_mean_stats(sims[idx, idx])
        l1_cos[cond] = float(np.mean(np.sum(e * emb["l1"], axis=1)))
    return {"encoder": name, "lang": lang, "n": len(rows),
            "en_retrieval": decompose(retrieval, mean_metric),
            "en_cosine": decompose(en_cos, mean_metric),
            "l1_cosine": {c: round(v, 4) for c, v in l1_cos.items()}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--subset", default="all")
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer
    for name in args.encoders:
        model = SentenceTransformer(name, device="cuda")
        model.half()
        for lang in args.langs:
            res = evaluate(model, name, lang, load_rows(lang, subset=args.subset))
            write_json(RESULTS / "retrieval" / f"{slug(name)}_{lang}.json", res)
            lv = res["en_retrieval"]["levels"]
            print(f"{lang} {slug(name):12s} top-1 to English  L1 {lv['l1']['score']:.3f}  "
                  f"L2c {lv['l2_canonical']['score']:.3f}  L2l {lv['l2_lexical']['score']:.3f}  "
                  f"L3 {lv['l3']['score']:.3f}")
        del model


if __name__ == "__main__":
    main()
