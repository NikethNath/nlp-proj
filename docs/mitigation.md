# Phase 3: mitigations

Phase 2 found that the change of script causes most of the loss on romanized
Malayalam and Tamil, and that how much depends on the model's familiarity with
the script. Inconsistent spelling adds a smaller penalty. Phase 3 tests the
fixes those findings suggest, on the same sentences, models and metrics. Full
tables are in [phase3_results.md](phase3_results.md), generated from
`results/mitigate/`.

## Mitigations

| Mitigation | Idea | Applied to |
|---|---|---|
| **Back-transliteration** (bt) | Convert romanized input to native script with IndicXlit before the model sees it | Models that know native script better: NLLB, LaBSE, BGE-M3; also Qwen |
| **Round-trip normalization** (rt) | Latin → native → Latin: stay in Latin script, but respell every word in IndicXlit's one canonical form | Models that prefer romanized input: Qwen, MuRIL fine-tuned on romanized comments; also LaBSE, BGE-M3 |
| **Fine-tuning on romanized text** | Train on FLORES dev (held out in Phase 2) with romanized sources | LaBSE (contrastive, in-batch negatives), NLLB-600M (translation into English) |
| **Script-augmented training** | Add each Latin-script training comment a second time, back-transliterated, with the same label | MuRIL on DravidianCodeMix |

Rewrites need no training and no extra data. They cost one IndicXlit pass per
input word, and results are cached. The training mitigations need parallel
romanized data, which the Phase 1 corpus provides for FLORES dev.

### Design choices

- **Same reference.** L1 always stays the original native text. *Recovered*
  is the share of the Phase 2 L1 → L3 loss that a mitigation wins back.
- **Seen and unseen writers.** Fine-tuning uses dev's native script, both L2s
  and simulated writers 0-2. Each writer keeps the same habits in dev and
  devtest, so writers 0-2 test spelling habits the model has seen, and
  writers 3-4 test habits it has not.
- **Forgetting.** Native script stays in the training data, and L1 is reported
  for every fine-tuned model, to check that fixing romanized input does not
  cost native-script quality.
- **Frozen embeddings.** LaBSE and NLLB keep most of their parameters in the
  word embeddings. Freezing those (and using 8-bit Adam for NLLB) fits
  training on an 8 GB GPU and limits how far the model can drift.
- **Rewrite fidelity is measured separately** (`latscript.mitigate.fidelity`),
  so a downstream result can be traced to how well IndicXlit did its part.

## Findings

**Fix the script, not the spelling.** Phase 2 found that the change of script
causes most of the loss, and the mitigations line up with that.
Back-transliteration recovers most of the loss, and so does training on
romanized text. Normalizing spelling within Latin script recovers almost
nothing.

| Mitigation | Model | Recovered L3 loss (ml / ta) | Cost |
|---|---|---|---|
| Back-transliteration | LaBSE, BGE-M3 (retrieval) | 100% / 100% | none measured |
| | Qwen2.5-7B (translation) | 81% / 82% | none measured |
| | NLLB-600M (translation) | 89% / 88% | none measured |
| Fine-tuning on romanized dev | LaBSE (retrieval) | 56% / 47% | none on native script |
| | NLLB-600M (translation) | 34% / 26% | −3.3 chrF++ on native script |
| Round-trip normalization | LaBSE, BGE-M3 (retrieval) | 8-10% / 0-1% | none |
| | Qwen2.5-7B (translation) | 4% / 3% | none |

### Rewriting the input

- **Back-transliteration works even though IndicXlit is imperfect.** It
  restores 83-91% of L2 words and 71-79% of L3 words exactly. Inconsistent
  spelling does survive the rewrite: L3 comes back about 10 points less
  accurately than L2. Yet the retrieval encoders score 0.998-1.000 top-1 on
  every back-transliterated condition. Matching one sentence among 1,012 is
  easy once most of its words are right, so the metric is at its ceiling.
  Translation is more demanding, and there Qwen's back-transliterated L3
  still trails native script by 2.7 points: the reconstruction errors show,
  but they are small.
- **NLLB goes from copying to translating.** Back-transliterated L3 reaches
  54.1 / 50.6 chrF++, against 13.7 / 12.6 romanized and 59.1 / 55.8 native.
- **Once the script is fixed, spelling is half of what remains.** After
  back-transliteration, NLLB's remaining loss for Malayalam is 5.0 points.
  Of that, 2.3 points fall between the back-transliterated L2 lexical and L3.
  That is 46% of the remainder, against 3% of the loss before
  back-transliteration. Tamil (1.9 of 5.2) and Qwen (1.5 of 2.7 for Malayalam)
  show the same pattern.
  Inconsistent spelling matters little while the script is wrong. Once the
  script is fixed, it is what limits the rewrite, through IndicXlit's weaker
  reconstruction of L3.
- **Back-transliteration helps Qwen too.** Its tokenizer is three times less
  efficient on native script, yet on FLORES it translates back-transliterated
  input (35.2 / 31.1 chrF++ for L3) far better than romanized input
  (23.7 / 19.0). Tokenizer cost is not the same as competence.
- **Round-trip normalization fixes spelling but not script, so it does little.**
  - It halves writer-to-writer disagreement (43% → 24% for Malayalam, 18% for
    Tamil) and brings L3 much closer to L2 canonical (51% → 22% and 17% of
    words differ).
  - The spelling gap was small to begin with, so the gain is small: 8-10% of
    the Malayalam retrieval loss, almost none for Tamil, 3-4% for Qwen
    translation.
  - It can also hurt. It respells human romanization into IndicXlit's machine
    convention, and the data card notes that convention matches the most
    common human spelling only about 62% of the time.

### Training on romanized text

- **Fine-tuning on 997 dev sentences closes a large part of the gap.** One
  epoch of contrastive training (about 2 minutes) roughly doubles LaBSE's
  romanized retrieval: L3 0.37 → 0.72 for Malayalam, 0.28 → 0.62 for Tamil.
  Two epochs of NLLB fine-tuning double its romanized chrF++ (L3 13.7 → 29.0,
  12.6 → 23.8). NLLB stops copying romanized input.
- **It generalizes to new spelling habits.** L3 writers 3-4, never seen in
  training, score the same as writers 0-2 for every model, to within 0.005
  top-1 and 0.1 chrF++. What the models learn is romanized Malayalam and Tamil
  in general, not the simulated writers' quirks.
- **NLLB pays for it on native script.** Native-script chrF++ falls 3.3 points
  in both languages, even with native sentences in the training mix and
  frozen embeddings. LaBSE shows no such cost. Back-transliteration, which
  leaves the model untouched, has no such cost either.

### Real comments: script augmentation fixes the classifier

On DravidianCodeMix, the fine-tuned classifier was the case where
back-transliteration *hurt* (Phase 2). Training on both scripts removes that
weakness without costing anything on the comments as typed.

| MuRIL fine-tuned, Tamil (3 seeds) | Typed | Back-transliterated | Round-tripped |
|---|---|---|---|
| Macro-F1, trained as released | 0.787 | 0.723 | 0.696 |
| Macro-F1, + back-transliterated copies | 0.791 | 0.803 | 0.729 |
| AUROC, trained as released | 0.892 | 0.833 | 0.833 |
| AUROC, + back-transliterated copies | 0.899 | 0.903 | 0.850 |

- **The augmented model is script-robust.** It is now slightly *better* on
  native script than on typed text, and no worse on typed text than before.
  A deployment can then accept comments in either script, or
  back-transliterate everything, without retraining.
- **Round-trip normalization hurts the classifier** (macro-F1 0.787 → 0.696
  on Tamil). The model learned offensive vocabulary as people actually spell
  it, and normalization rewrites those spellings into IndicXlit's convention.
  This is the clearest case against spelling normalization.
- **Malayalam is less clear.** The same pattern holds for native script with
  augmentation (macro-F1 0.619 → 0.663; AUROC 0.800 → 0.836), but with only
  32 offensive test comments the differences are within seed-to-seed noise.
- **Zero-shot Qwen gains nothing from normalization.** Malayalam AUROC is
  0.725 typed against 0.701 round-tripped. On Tamil, round-tripping (0.567)
  hurts as much as back-transliteration (0.580), against 0.677 typed. On
  noisy social-media text, both rewrites move the comment away from the
  spellings the model knows.

### What this means in practice

1. **For a model trained on native script** (NLLB, LaBSE, BGE-M3, most
   multilingual models), back-transliterate romanized input. It needs no
   training, has no measured cost on native-script input, and recovers most
   of the loss on clean text (82-100% on FLORES). Test it on your own data
   first. On noisy, code-mixed comments it hurt both zero-shot Qwen and the
   classifier trained on romanized comments: English words get nativized and
   slang is misreconstructed.
2. **Where parallel romanized data exists, fine-tuning helps** and generalizes
   to unseen spelling habits. Watch for loss on native script (NLLB: −3.3
   chrF++).
3. **For a classifier trained on real romanized data, augment training with
   back-transliterated copies.** The model then works in both scripts.
4. **Don't bother normalizing spelling.** It recovers at most a tenth of the
   loss, and on a model trained on human spellings it makes things worse.

## Reproducing

```bash
scripts/run_phase3.sh      # after the Phase 2 setup; ~6 hours on an 8 GB laptop GPU
```

IndicXlit runs as four parallel CPU processes (`INDICXLIT_PROCS`). The first
build of the rewritten inputs takes about 20 minutes, and later builds read
the cache. Fine-tuned models are saved under `data/models/`.
