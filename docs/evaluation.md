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
about 1-17%.

| Task | Model | Lang | L1 | L2 lexical | L3 | Spelling share |
|---|---|---|---|---|---|---|
| Retrieval to English (top-1) | LaBSE | ml / ta | 1.00 / 1.00 | 0.44 / 0.34 | 0.37 / 0.28 | 11% / 8% |
| | BGE-M3 | ml / ta | 1.00 / 1.00 | 0.44 / 0.31 | 0.36 / 0.23 | 12% / 11% |
| Translation (chrF++) | NLLB-600M | ml / ta | 59.1 / 55.8 | 14.9 / 13.1 | 13.7 / 12.6 | 3% / 1% |
| | Qwen2.5-7B (core subset) | ml / ta | 37.9 / 33.8 | 24.8 / 19.8 | 23.7 / 19.0 | 8% / 6% |
| Surprisal (bits/word, lower is better) | Qwen2.5-1.5B | ml / ta | 23.1 / 20.0 | 35.4 / 32.7 | 36.1 / 33.1 | 5% / 3% |
| | Sarvam-1 | ml / ta | 13.5 / 11.9 | 21.9 / 20.6 | 23.7 / 22.3 | 17% / 17% |

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
- **Translation: an LLM trades native-script quality for robustness.**
  Qwen2.5-7B translates native script much worse than NLLB (chrF++ 38/34
  against 58/55 on the same core sentences). It does translate romanized
  input: its copy rate is under 3%, and romanized chrF++ is 19-25, well above
  NLLB's 12-14. So its script gap is about 14 points, a third of NLLB's.
  Spelling costs it 1 point (lexical L2), similar to NLLB.
- **Surprisal.**
  - Romanized text costs Qwen2.5-1.5B about 12-13 extra bits per word, even
    though its tokenizer handles romanized text far more cheaply (next
    point). The cost is unfamiliarity, not tokenization. Inconsistent
    spelling adds 0.4-0.7 bits.
  - Sarvam-1, an Indic language model, models native script far better
    (13.5 and 11.9 bits per word, against Qwen's 23.1 and 20.0), and pays a
    smaller script penalty (8.4-8.7 bits).
  - It is the model most sensitive to spelling. L3 costs it 1.7 bits per word
    more than the lexical L2 in both languages, 17% of its total loss. That
    fits a model that learned romanized text in one convention.
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

**Real comments: converting to native script is not a free fix.** Every
classifier did worse when the same Latin-script test comments were
back-transliterated into native script. That includes the zero-shot LLM, which
had no task training in either script.

| Classifier | Lang | Macro-F1 typed | Macro-F1 native | AUROC typed | AUROC native |
|---|---|---|---|---|---|
| MuRIL, fine-tuned | ta | 0.783 | 0.716 | 0.891 | 0.830 |
| XLM-R, fine-tuned | ta | 0.774 | 0.694 | 0.892 | 0.810 |
| Qwen2.5-7B, zero-shot | ta | 0.516 | 0.453 | 0.677 | 0.580 |
| MuRIL, fine-tuned | ml | 0.648 | 0.637 | 0.850 | 0.797 |
| XLM-R, fine-tuned | ml | 0.591 | 0.500 | 0.728 | 0.601 |
| Qwen2.5-7B, zero-shot | ml | 0.575 | 0.543 | 0.725 | 0.720 |

- **Fine-tuned encoders.** The Tamil drop of 6-10 macro-F1 points holds in
  every seed of both models. They have learned romanized offensive
  vocabulary, and back-transliteration moves inputs away from it: a
  Latin-trained model should see Latin text.
- **Zero-shot LLM.** Qwen2.5-7B also prefers the comments as typed. On Tamil,
  AUROC falls from 0.677 to 0.580 after back-transliteration (difference
  0.097, 95% interval 0.073-0.119). This matches its tokenizer, which spends
  about 10 tokens on a native Tamil word. It is also a weak offensive-language
  detector in either script: AUROC 0.68 against 0.89 for the fine-tuned
  encoders on the same Tamil comments.
- **Malayalam is too small to be conclusive.** The test set has 32 offensive
  comments, so its macro-F1 is unstable. One XLM-R seed collapsed to
  predicting the majority class, and Qwen's Malayalam differences have
  intervals that include zero. The fine-tuned AUROC drops point the same way
  as Tamil but should be read with that caveat.

Taken together, a model's familiarity with a script matters more than the
script itself.

- Models trained mainly on native script (NLLB, LaBSE, BGE-M3) lose most of
  their competence on romanized input; NLLB gives up and copies it.
- Models that have seen more romanized text lose less. Qwen2.5 translates
  romanized input at a third of NLLB's script gap, and it and the encoders
  fine-tuned on romanized comments do *worse* on native script.
- Within romanized text, inconsistent spelling is a second-order but
  measurable cost. It is largest for Sarvam-1, whose romanized training is
  likely the most uniform.

### Still pending

The human-typed L3 is pending. Once responses are collected, the core
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
