"""Collect every Phase 3 result into docs/phase3_results.md.

    PYTHONPATH=src python -m latscript.mitigate.report

Each mitigation is compared with the Phase 2 baseline on the same model and
sentences. *Recovered* is the share of the baseline's L1 -> L3 loss that the
mitigation wins back: (L3 mitigated - L3 baseline) / (L1 baseline - L3 baseline).
For fine-tuned models, L3 is split into the writers seen in training (0-2)
and the unseen ones (3-4).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..eval.conditions import RESULTS, ROOT

OUT = ROOT / "docs" / "phase3_results.md"
P3 = RESULTS / "mitigate"
SEEN, UNSEEN = (0, 1, 2), (3, 4)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _levels(dec: dict) -> dict:
    lv, w = dec["levels"], dec["l3_writers"]
    return {"l1": lv["l1"]["score"], "l2c": lv["l2_canonical"]["score"],
            "l2l": lv["l2_lexical"]["score"], "l3": lv["l3"]["score"],
            "seen": sum(w[i] for i in SEEN) / len(SEEN), "unseen": sum(w[i] for i in UNSEEN) / len(UNSEEN)}


def _recovered(base: dict, mit: dict, key: str = "l3") -> str:
    loss = base["l1"] - base["l3"]
    return f"{(mit[key] - base['l3']) / loss:+.0%}" if abs(loss) > 1e-9 else "-"


def comparison(title: str, unit: str, d: int, rows: list[tuple[str, str, str, dict, dict]],
               writers: bool = False) -> list[str]:
    """rows: (model, lang, mitigation, baseline levels, mitigated levels)."""
    head = ["Model", "Lang", "Mitigation", "L1", "L2 canonical", "L2 lexical", "L3"]
    head += ["L3 seen writers", "L3 unseen writers"] if writers else []
    head += ["Recovered (L3)"] + (["Recovered (unseen)"] if writers else [])
    lines = [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    shown = set()
    for model, lang, name, base, mit in rows:
        if (model, lang) not in shown:
            shown.add((model, lang))
            cells = [model, lang, "none (Phase 2)"] + [f"{base[k]:.{d}f}" for k in ("l1", "l2c", "l2l", "l3")]
            cells += [f"{base['seen']:.{d}f}", f"{base['unseen']:.{d}f}"] if writers else []
            cells += ["-"] + (["-"] if writers else [])
            lines.append("| " + " | ".join(cells) + " |")
        cells = ["", "", name] + [f"{mit[k]:.{d}f}" for k in ("l1", "l2c", "l2l", "l3")]
        cells += [f"{mit['seen']:.{d}f}", f"{mit['unseen']:.{d}f}"] if writers else []
        cells += [_recovered(base, mit)] + ([_recovered(base, mit, "unseen")] if writers else [])
        lines.append("| " + " | ".join(cells) + " |")
    return lines + ["", f"Unit: {unit}.", ""]


def fidelity_section() -> list[str]:
    res = [r for r in (_load(P3 / "fidelity" / f"rewrites_{l}.json") for l in ("ml", "ta")) if r]
    if not res:
        return []
    lines = ["## How faithful the rewrites are", "",
             "| Lang | Back-transliteration word accuracy: L2 canonical | L2 lexical | L3 | "
             "Writer-pair word disagreement: before | after round-trip | "
             "L3 vs L2 canonical: before | after round-trip |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        bt, rt = r["backtranslit"], r["roundtrip"]
        pair, to_l2 = rt["l3_writer_pair_word_disagreement"], rt["l3_vs_l2_canonical_word_disagreement"]
        lines.append(f"| {r['lang']} | {bt['l2_canonical']['word_accuracy']:.1%} | "
                     f"{bt['l2_lexical']['word_accuracy']:.1%} | {bt['l3']['word_accuracy']:.1%} | "
                     f"{pair['before']:.1%} | {pair['after']:.1%} | {to_l2['before']:.1%} | {to_l2['after']:.1%} |")
    return lines + ["", "Word accuracy: back-transliterated words identical to the native original, "
                        "over sentences whose word counts align.", ""]


def mt_section() -> list[str]:
    runs = [("nllb-200-distilled-600M", "all", [("back-transliterated", "nllb-200-distilled-600M", "_bt"),
                                                ("fine-tuned on romanized dev", "nllb-200-distilled-600M-romanized", "")]),
            ("Qwen2.5-7B-Instruct", "core", [("back-transliterated", "Qwen2.5-7B-Instruct", "_bt"),
                                             ("round-tripped", "Qwen2.5-7B-Instruct", "_rt")])]
    rows = []
    for model, subset, mits in runs:
        for lang in ("ml", "ta"):
            base = _load(RESULTS / "mt" / f"{model}_{lang}_{subset}.json")
            for name, mslug, suffix in mits:
                mit = _load(P3 / "mt" / f"{mslug}_{lang}_{subset}{suffix}.json")
                if base and mit:
                    rows.append((model, lang, name, _levels(base), _levels(mit)))
    return comparison("Translation into English, chrF++", "chrF++; NLLB on the full devtest, "
                      "Qwen on the 248-sentence core subset", 1, rows, writers=True) if rows else []


def retrieval_section() -> list[str]:
    rows = []
    for enc in ("LaBSE", "bge-m3"):
        for lang in ("ml", "ta"):
            base = _load(RESULTS / "retrieval" / f"{enc}_{lang}.json")
            for name, path in (("back-transliterated", P3 / "retrieval" / f"{enc}_{lang}_bt.json"),
                               ("round-tripped", P3 / "retrieval" / f"{enc}_{lang}_rt.json"),
                               ("fine-tuned on romanized dev",
                                P3 / "retrieval" / f"{enc}-romanized_{lang}.json")):
                mit = _load(path)
                if base and mit:
                    rows.append((enc, lang, name, _levels(base["en_retrieval"]), _levels(mit["en_retrieval"])))
    return comparison("Bitext retrieval to English, top-1 accuracy", "top-1 accuracy over the "
                      "1,012 devtest English references", 3, rows, writers=True) if rows else []


def classify_section() -> list[str]:
    lines = []
    zs = [(l, _load(RESULTS / "classify" / f"zeroshot_Qwen2.5-7B-Instruct_{l}.json"),
           _load(P3 / "classify" / f"zeroshot_Qwen2.5-7B-Instruct_{l}_rt.json")) for l in ("ml", "ta")]
    ft = [(l, tag, _load(P3 / "classify" / f"finetune_muril-base-cased_{l}{tag}.json"))
          for l in ("ml", "ta") for tag in ("_rt", "_aug_rt")]
    if not any(m for _, _, m in zs) and not any(r for _, _, r in ft):
        return []
    lines += ["## Offensive-language detection on DravidianCodeMix", "",
              "Latin-script, in-language test comments. *Typed* is the comment as posted, *native* its "
              "IndicXlit back-transliteration, *round-tripped* its spelling normalized in Latin script.", "",
              "| Classifier | Lang | Training | Metric | Typed | Native | Round-tripped |",
              "|---|---|---|---|---|---|---|"]
    for lang, base, mit in zs:
        if base and mit:
            for metric in ("auroc", "macro_f1"):
                lines.append(f"| Qwen2.5-7B zero-shot | {lang} | none | {metric} | "
                             f"{base[metric]['a']['score']:.3f} | {base[metric]['b']['score']:.3f} | "
                             f"{mit['rt'][metric]['b']['score']:.3f} |")
    for lang, tag, r in ft:
        if not r:
            continue
        seeds = r["seeds"]
        mean = lambda f: sum(f(s) for s in seeds) / len(seeds)
        training = "romanized + back-transliterated copies" if r["augmented"] else "as released (mostly romanized)"
        for metric in ("macro_f1", "auroc"):
            lines.append(f"| MuRIL fine-tuned | {lang} | {training} | {metric} | "
                         f"{mean(lambda s: s[metric]['a']['score']):.3f} | "
                         f"{mean(lambda s: s[metric]['b']['score']):.3f} | "
                         f"{mean(lambda s: s['rt'][metric]['b']['score']):.3f} |")
    return lines + ["", f"Fine-tuned rows average {len(ft[0][2]['seeds']) if ft and ft[0][2] else 3} seeds.", ""]


def main() -> None:
    parts = ["# Phase 3 results", "",
             "Generated by `python -m latscript.mitigate.report` from `results/`; do not edit by hand. "
             "Methods and interpretation are in [mitigation.md](mitigation.md).", "",
             "*Recovered* is the share of the Phase 2 L1 -> L3 loss won back by the mitigation "
             "(negative means it made L3 worse). Fine-tuned models were trained on L2 and on "
             "writers 0-2 of FLORES dev; writers 3-4 are unseen.", ""]
    for section in (fidelity_section, retrieval_section, mt_section, classify_section):
        parts += section()
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
