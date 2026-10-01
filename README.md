# Latin-script degradation in Malayalam and Tamil

Measuring how much language models lose when Malayalam and Tamil are typed in
Latin script instead of native script, and how much of that loss comes from
the change of script versus the inconsistency of informal spelling.

## Corpus

**Controlled arm.** Every FLORES-200 dev and devtest sentence is rendered at
three orthographic levels with the meaning held fixed:

| Level | Field | How it is made |
|---|---|---|
| L1 | `l1_native` | Native script, as published |
| L2 | `l2_canonical` | IndicXlit romanization, one deterministic form per word |
| L2 (alt.) | `l2_lexical` | Most common human spelling from Dakshina, else Aksharantar, else a rule-based romanizer |
| L3 | `l3_writers` | Five simulated writers who sample real human spellings from Dakshina, each with consistent personal habits |

**Ecological arm.** DravidianCodeMix offensive-language comments (Tamil 43.5k,
Malayalam 12.7k after deduplication), tagged by script. Latin-script comments
in dev and test also carry an IndicXlit back-transliteration as their L1.

**Human-written subset.** 248 FLORES sentences per language, typed in Latin
script by native speakers through Google Forms. The collection kit is ready;
responses have not been collected yet.

[docs/datacard.md](docs/datacard.md) has the schema, construction details,
statistics, limitations and licenses.

## Status

**Phase 1 (evaluation set).** Built:
- the FLORES L1/L2/L3 corpus;
- the cleaned DravidianCodeMix splits with back-transliteration;
- the collection kit;
- the release bundle.

Two steps need people:
- the hand check of 50 sentences ([review/README.md](review/README.md));
- collecting the human-written subset ([collection/README.md](collection/README.md)).

**Phase 2 (model evaluation).** Run on FLORES devtest and the DravidianCodeMix
test set, covering tokenization, bitext retrieval, LM surprisal, translation
into English and offensive-language detection. Methods and findings are in
[docs/evaluation.md](docs/evaluation.md); the full tables are in
[docs/phase2_results.md](docs/phase2_results.md). The human-typed L3
comparison waits on the collection.

**Phase 3 (mitigations).** Back-transliteration, spelling normalization by
round-trip, fine-tuning LaBSE and NLLB on romanized FLORES dev, and
script-augmented classifier training, each measured against the Phase 2
baseline. Methods and findings are in [docs/mitigation.md](docs/mitigation.md);
the full tables are in [docs/phase3_results.md](docs/phase3_results.md).

**Phase 4.** The write-up has not started.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/fetch_data.sh          # ~2.3 GB of source data into data/raw
scripts/setup_indicxlit.sh     # separate Python 3.10 env for IndicXlit (CPU is fine)
```

IndicXlit needs fairseq 0.12, which does not install on current Python. It runs
as a subprocess from `.venv-xlit`, and every result is cached under
`data/cache/indicxlit`, so the model runs only the first time.

## Building

```bash
export PYTHONPATH=src
.venv/bin/python -m latscript.translit.generate           # FLORES arm      -> data/derived/flores_*.jsonl
.venv/bin/python -m latscript.corpora.dravidiancodemix    # DravidianCodeMix -> data/derived/dcm_*.jsonl
.venv/bin/python -m latscript.corpus_stats                # docs/corpus_stats.json
.venv/bin/python -m latscript.review                      # review/sanity_check_*.csv
.venv/bin/python scripts/build_release.py v0.1            # dist/orthography-corpus-v0.1.tar.gz
.venv/bin/python -m pytest tests/
```

## Evaluating (Phase 2)

Needs a CUDA GPU (8 GB is enough) and the built corpus.

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements-eval.txt
scripts/run_phase2.sh                         # every evaluation, then the report
.venv/bin/python -m latscript.eval.report     # docs/phase2_results.md from results/
```

Each step can also be run on its own, for example
`python -m latscript.eval.translate nllb facebook/nllb-200-distilled-600M`.
Model outputs are cached in `data/eval`, so interrupted runs resume.

## Mitigating (Phase 3)

```bash
scripts/run_phase3.sh                         # rewrites, fine-tuning, re-evaluation, report
.venv/bin/python -m latscript.mitigate.report # docs/phase3_results.md from results/mitigate/
```

The evaluation scripts take `--variant bt|rt` to run on back-transliterated or
round-tripped inputs, and `--phase3` to write a fine-tuned model's results to
`results/mitigate/`.

The first build runs IndicXlit over about 30k FLORES words and 23k comment
words, roughly 10 minutes on CPU. Later builds take about a minute.

## Layout

```
src/latscript/
  corpora/flores.py            aligned FLORES-200 loading
  corpora/dravidiancodemix.py  DravidianCodeMix cleaning, splits, back-transliteration
  translit/rules.py            deterministic ml/ta -> ASCII romanizer
  translit/lexicon.py          human spelling distributions from Dakshina, with misalignment filter
  translit/aksharantar.py      fallback spellings from Aksharantar
  translit/indicxlit.py        IndicXlit via subprocess, with cache
  translit/noise.py            spelling-variation model, simulated writers, plausibility check
  translit/generate.py         builds the FLORES L1/L2/L3 corpus
  translit/tokens.py           word tokenization shared by all transforms
  human/select.py              chooses sentences for human collection
  human/forms.py               renders the Google Forms Apps Script
  human/ingest.py              Forms CSV -> flagged per-sentence records
  human/analyze.py             inter-writer divergence, human vs simulated L3
  corpus_stats.py              statistics for the data card
  review.py                    hand-check sheet export
  eval/conditions.py           the eight inputs per sentence, data loading
  eval/stats.py                script/spelling decomposition, paired bootstrap
  eval/tokenization.py         tokens per word under each orthography
  eval/retrieval.py            bitext retrieval with sentence encoders
  eval/surprisal.py            sentence negative log-likelihood under causal LMs
  eval/translate.py            translation into English, chrF++, copy rate
  eval/classify.py             DravidianCodeMix: fine-tuned and zero-shot, typed vs native
  eval/report.py               results/ -> docs/phase2_results.md
  mitigate/transforms.py       back-transliteration and round-trip rewrites via IndicXlit
  mitigate/fidelity.py         how faithfully the rewrites recover native words and remove variation
  mitigate/train.py            fine-tuning LaBSE and NLLB on romanized FLORES dev
  mitigate/report.py           results/mitigate/ -> docs/phase3_results.md
results/                       Phase 2 summaries, one JSON per task, model and language
results/mitigate/              Phase 3 summaries
collection/                    manifest, form script, collection protocol
review/                        hand-check sheets and instructions
docs/                          data card and corpus statistics
scripts/                       data download, IndicXlit setup and worker, release bundling
tests/                         romanizer regressions, collection kit
```

## Findings so far

**Mitigations**
- **Back-transliteration recovers most of the loss on clean text.** It wins
  back 88-89% of NLLB's L1 → L3 chrF++ loss, 81-82% of Qwen2.5-7B's, and all
  of the retrieval loss, with no model change.
- **Training on romanized text also works and generalizes.** One epoch on
  FLORES dev recovers about half of LaBSE's retrieval loss. Two epochs
  recover 26-34% of NLLB's, at a cost of 3.3 chrF++ on native script. Unseen
  writers do as well as the ones trained on.
- **Spelling normalization barely helps** (0-10% recovered), and it hurts a
  classifier trained on real romanized comments.
- **Script-augmented training makes the classifier work in both scripts.**
  Adding back-transliterated copies of the training comments lifts Tamil
  native-script macro-F1 from 0.72 to 0.80, with no loss on typed comments.

**Model evaluation**
- **The script causes most of the loss.** Romanized, top-1 retrieval to
  English falls from ~100% to 31-47%, and NLLB-600M chrF++ from ~57 to ~14.
  NLLB echoes romanized input back untranslated 70-80% of the time.
- **How much depends on the model's familiarity with the script.** Qwen2.5-7B
  translates native script worse than NLLB but actually translates romanized
  input, so its script gap is about 14 chrF++ points, a third of NLLB's.
- **Inconsistent spelling adds a smaller, reliable penalty.** Against the
  lexical L2 it is 1-17% of the total loss: 6-10 points of retrieval
  accuracy, 0.4-1.7 bits per word of surprisal. The Indic LM Sarvam-1 is the
  most spelling-sensitive.
- **The script tax runs both ways.** Qwen2.5's tokenizer spends about 3x more
  tokens on native-script Malayalam than on romanized text; Indic tokenizers
  spend 40-120% more on romanized text.
- **Back-transliteration is not a free fix.** Classifiers fine-tuned on real
  romanized comments lose 6-10 macro-F1 points on Tamil when the same comments
  are converted to native script. Zero-shot Qwen2.5-7B, with no task
  training, also drops from AUROC 0.68 to 0.58.

**Spelling in the source data**
- **Real spelling is inconsistent.** Two Dakshina annotators spell the same
  word differently 52% of the time for Malayalam and 58% for Tamil. The most
  common spelling of a word matches a given annotator only 69.5% and 64.6% of
  the time.
- **IndicXlit spells like a person, but not like most people.** Its output is a
  spelling some annotator used about 90% of the time, but the most common one
  only about 62% of the time.
- **Dakshina has misaligned rows.** Some lexicon entries are off by one
  (ஹாரி romanized as "potter"). Filtering drops 1.8% of Malayalam and 5.8% of
  Tamil attestations.

**The simulated L3**
- **It is close to human variation, and a little tamer.** Two simulated writers
  disagree on about 80% as many words as two annotators do, so degradation
  measured on it is a mild lower bound.

**DravidianCodeMix**
- **The raw Malayalam splits leak.** 944 of 1,999 validation comments also
  appear in train, and 27% of train is duplicates.
- **The Malayalam data is imbalanced.** Only 32 comments in the Latin-script
  test set are offensive.
