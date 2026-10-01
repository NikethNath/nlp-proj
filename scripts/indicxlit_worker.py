"""Batch IndicXlit transliteration of a word list. Runs inside .venv-xlit.

Reads one word per line, writes ``word<TAB>top-1 transliteration`` lines,
appending, so an interrupted run resumes where it stopped. Called by
latscript.translit.indicxlit; not meant to be run by hand.

The engine's own batch helper reorders and merges hypotheses, so this calls the
underlying fairseq translator directly and matches outputs to inputs by id.
Beam search top-1 without dictionary rescoring: one deterministic output per
word, which is what the canonical condition needs.
"""
import argparse
import os
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["indic2en", "en2indic"], required=True)
    ap.add_argument("--lang", choices=["ml", "ta"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from ai4bharat.transliteration import XlitEngine

    if args.direction == "indic2en":
        engine = XlitEngine(src_script_type="indic", beam_width=args.beam, rescore=False)
        src, tgt = args.lang, "en"
    else:
        engine = XlitEngine([args.lang], beam_width=args.beam, rescore=False)
        src, tgt = "en", args.lang

    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as fh:
            done = {line.split("\t", 1)[0] for line in fh if "\t" in line}
    with open(args.input, encoding="utf-8") as fh:
        todo = [w for w in (l.strip() for l in fh) if w and w not in done]
    print(f"{len(done)} cached, {len(todo)} to transliterate", file=sys.stderr, flush=True)

    with open(args.output, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.chunk):
            batch = todo[start:start + args.chunk]
            prepared = engine.pre_process(
                [unicodedata.normalize("NFC", w) for w in batch], src, tgt)
            raw = engine.transliterator.translate(prepared, nbest=1)
            best = {}
            for line in raw.split("\n"):
                if line.startswith("H-"):
                    ident, _score, hyp = line.split("\t", 2)
                    best.setdefault(int(ident[2:]), "".join(hyp.split(" ")))
            for i, word in enumerate(batch):
                out.write(f"{word}\t{best.get(i, '')}\n")
            out.flush()
            print(f"  {min(start + args.chunk, len(todo))}/{len(todo)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
