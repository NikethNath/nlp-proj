# Phase 2: model evaluation

How much do models lose when Malayalam and Tamil are written in Latin script,
and how much of that is the change of script as opposed to inconsistent
spelling? The full tables are in [phase2_results.md](phase2_results.md),
generated from `results/`.

## Design

### Decomposition

Every FLORES devtest sentence is scored in eight inputs with the same meaning:
native script (L1), the two fixed romanizations (L2 canonical from IndicXlit,
L2 lexical from the most common human spelling), and the five simulated
writers (L3). For any metric:

- **script gap** = L1 → L2: the same words, in Latin script with one fixed
  spelling each;
- **spelling gap** = L2 → L3: the same script, with spelling that varies
  between writers the way real typists' does;
- **spelling share** = spelling gap / total L1 → L3 gap.

Each gap is reported against both L2s. The lexical L2 comes from the same
human spelling data as L3, so its spelling gap isolates variance. The canonical
L2 also carries IndicXlit's machine conventions (see the data card).

L3 is the mean of the five writers' corpus-level scores. Intervals are 95%
paired bootstrap intervals over sentences (1,000 resamples): each resample
draws the same sentences for every condition and writer, so condition
differences are tested within sentence.

FLORES dev is left untouched for Phase 3, where mitigations may need tuning.

### Tasks

| Task | What it measures | Models | Data |
|---|---|---|---|
| Tokenizer fertility | Subword tokens per native word | NLLB, XLM-R, MuRIL, IndicBERT v2, LaBSE, Qwen2.5, Sarvam-1, o200k (GPT-4o) | FLORES devtest, 1,012 sentences |
| Bitext retrieval | Is the sentence's own English reference its nearest neighbour among all 1,012 references? | LaBSE, BGE-M3 | FLORES devtest |
| LM surprisal | Bits to encode the whole sentence, per native word | Qwen2.5-1.5B, Sarvam-1 (2B) | FLORES devtest |
| Translation to English | chrF++ against the FLORES English reference | NLLB-200 600M and 1.3B (distilled) | FLORES devtest |
| | | Qwen2.5-7B-Instruct (4-bit) | Core subset: the 248 sentences of the human collection |
| Offensive-language detection | Macro-F1 and AUROC, as typed vs. back-transliterated | MuRIL and XLM-R fine-tuned (3 seeds); Qwen2.5-7B-Instruct zero-shot | DravidianCodeMix test, Latin-script in-language comments |

**Surprisal.** The content is held fixed, so the total negative
log-likelihood of a sentence is comparable across conditions even though they
tokenize differently. Per-token perplexity would not be comparable.

**Translation.** NLLB has no code for romanized Malayalam or Tamil, so every
condition is tagged `mal_Mlym`/`tam_Taml`, as a user of the model would tag
it. The LLM gets one prompt for every condition. The prompt names the language
and says the text may be in native script or Latin letters, but never says
which. Outputs that just echo the input are counted separately as the copy rate.

**Classification.** The FLORES arm holds meaning fixed by construction. Here
it is held fixed by pairing: each real Latin-script comment is scored as typed
and after IndicXlit back-transliteration into native script. The two
classifier types answer different questions:

- *Fine-tuned* encoders are trained on the cleaned train split, which is
  mostly romanized. The typed-vs-native gap shows what back-transliterating at
  inference would do to a model trained this way. It does not measure how hard
  either script is.
- The *zero-shot* LLM has no task training that favours either script, so its
  gap is its own orthographic handicap. AUROC is used because it does not
  depend on the decision threshold.

### Models not included

IndicTrans2, Llama 3, Gemma 2 and Aya are gated on Hugging Face and need an
access token. Every script takes model names as arguments, so they can be
added by running with `HF_TOKEN` set. COMET was left out because chrF++ needs
no model and is the FLORES-200 standard.

## Findings

**The change of script causes most of the loss; inconsistent spelling adds a
smaller, reliable penalty.** On every task that measures understanding, the
L1 → L2 gap is several times the L2 → L3 gap. Against the lexical L2, the one
that isolates spelling variance, the spelling share of the total loss is
about 1-12%.

| Task | Model | Lang | L1 | L2 lexical | L3 | Spelling share |
|---|---|---|---|---|---|---|
| Retrieval to English (top-1) | LaBSE | ml / ta | 1.00 / 1.00 | 0.44 / 0.34 | 0.37 / 0.28 | 11% / 8% |
| | BGE-M3 | ml / ta | 1.00 / 1.00 | 0.44 / 0.31 | 0.36 / 0.23 | 12% / 11% |
| Translation (chrF++) | NLLB-600M | ml / ta | 59.1 / 55.8 | 14.9 / 13.1 | 13.7 / 12.6 | 3% / 1% |
| Surprisal (bits/word, lower is better) | Qwen2.5-1.5B | ml / ta | 23.1 / 20.0 | 35.4 / 32.7 | 36.1 / 33.1 | 5% / 3% |

- **Retrieval.** Both encoders map native-script sentences onto their English
  translations almost perfectly. Romanized, only a third to a half still find
  their translation. The romanized sentence is also far from its own
  native-script version: LaBSE cosine to L1 is about 0.3. The encoders do not
  treat the two as the same text. Spelling variation costs another 6-10
  points of top-1 accuracy, the largest and most consistent spelling effect
  measured.
- **Translation: NLLB stops translating.** Given romanized input, NLLB-600M
  echoes it back 70-80% of the time instead of translating it. The copy rate
  is 0% for native script. chrF++ falls by about 43 points. Once the model has
  given up, spelling barely matters (0.5-1.6 points), so a small spelling gap
  here does not mean the model tolerates variation. Scale does not help. On
  the core subset NLLB-1.3B scores 2-3 points higher than 600M in every
  condition, but its script gap is the same 42-43 points.
- **Surprisal.** Romanized text costs Qwen2.5-1.5B about 12-13 extra bits per
  word. That holds even though its tokenizer handles romanized text far more
  cheaply (next point), so the cost is unfamiliarity, not tokenization.
  Inconsistent spelling adds 0.4-0.7 bits.
- **Tokenization.** The script tax runs both ways.
  - General-purpose tokenizers barely know the scripts. Qwen2.5 spends 13.6
    tokens per Malayalam word (10.3 for Tamil) in native script against about
    4.3 romanized, a threefold cost, context and latency penalty for
    native-script users.
  - Indic-trained tokenizers (MuRIL, IndicBERT v2, Sarvam-1) are the
    opposite: romanized text costs them 40-120% more tokens.
  - GPT-4o's o200k is nearly neutral (3.6 vs 3.9 for Malayalam).
  - Spelling variation changes token counts by at most 0.15 per word.
- **Two L2s.** Against IndicXlit's canonical L2, the Tamil spelling gap
  sometimes comes out slightly negative on tokenization and surprisal: L3's
  human-like spellings are *easier* than IndicXlit's conventions. This is the
  convention shift the data card warns about, and why the lexical L2 is the
  clean reference for spelling variance.

**Real comments: converting to native script is not a free fix.** Encoders
fine-tuned on the (mostly romanized) DravidianCodeMix training data do worse
when the same Latin-script test comments are back-transliterated into native
script.

| Classifier | Lang | Macro-F1 typed | Macro-F1 native | AUROC typed | AUROC native |
|---|---|---|---|---|---|
| MuRIL | ta | 0.783 | 0.716 | 0.891 | 0.830 |
| XLM-R | ta | 0.774 | 0.694 | 0.892 | 0.810 |
| MuRIL | ml | 0.648 | 0.637 | 0.850 | 0.797 |
| XLM-R | ml | 0.591 | 0.500 | 0.728 | 0.601 |

The Tamil drop of 6-10 macro-F1 points holds in every seed of both models. The model has
learned romanized offensive vocabulary, and back-transliteration moves inputs
away from it: a Latin-trained model should see Latin text. The Malayalam test
set has 32 offensive comments, so its macro-F1 is unstable. One XLM-R seed
collapsed to predicting the majority class. The AUROC drop is consistent in
direction but should be read with that caveat.

Taken together, the model's familiarity with a script matters more than the
script itself. Models trained on native script (NLLB, LaBSE, BGE-M3) lose most
of their competence on romanized input. A model trained on romanized input
loses competence the other way. Within romanized text, inconsistent spelling
is a second-order but measurable cost.

### Not yet run

The Qwen2.5-7B-Instruct runs (zero-shot offensive classification and
translation on the core subset) and Sarvam-1 surprisal are implemented but not
run. Their weights could not be downloaded in the time available (the network
fell to under 100 KB/s mid-run). Run them with the corresponding lines of
`scripts/run_phase2.sh` and regenerate the report. The zero-shot classifier
matters most of these: it is the one classification measurement not biased
toward either script by training data.

The human-typed L3 is also pending. Once responses are collected, the core
subset lets its gaps be compared directly with the simulated writers'. Since
simulated writers disagree about 20% less than real annotators, the spelling
gaps above are likely a lower bound.

## Reproducing

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements-eval.txt
scripts/run_phase2.sh      # ~8 hours on an 8 GB laptop GPU; resumable
```

Model outputs are cached under `data/eval/` (hypotheses, one JSON line per
sentence and condition). Summaries go to `results/`, and
`python -m latscript.eval.report` rebuilds `docs/phase2_results.md` from them.
