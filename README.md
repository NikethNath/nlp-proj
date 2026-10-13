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

Two steps still need people:
- the hand check of 50 sentences ([review/README.md](review/README.md));
- collecting the human-written subset ([collection/README.md](collection/README.md)).

**Phases 2-3 (quantify, mitigate).** Done on CPU with stand-in models for each planned role:
- NLLB-200 600M for translation;
- Gemma-3 4B, zero-shot;
- MuRIL, fine-tuned;
- a character n-gram classifier.

The harness runs any OpenAI-compatible endpoint, so the planned 8B and frontier-API models only need a GPU or API key. Results are in [results/README.md](results/README.md).

**Literature survey.** [docs/related_work.md](docs/related_work.md).

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

The first build runs IndicXlit over about 30k FLORES words and 23k comment
words, roughly 10 minutes on CPU. Later builds take about a minute.

## Evaluation (Phases 2-3)

```bash
scripts/setup_eval.sh          # .venv-eval (Python 3.12, CPU torch), llama.cpp server, Gemma/Qwen GGUFs
scripts/run_all.sh             # prepare inputs, run every model, train classifiers and LoRA, analyze
```

`run_all.sh` resumes if interrupted and documents the sample sizes used for the
reported results (set for an 8-core CPU; raise `N_NLLB`, `N_LLM`, `N_POS` on a GPU).
To evaluate a hosted model, point `run.py` at its endpoint:

```bash
PYTHONPATH=src .venv-eval/bin/python -m latscript.eval.run --task mt --model gpt-4o --chat gpt-4o \
    --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --lang ml --n 200 --conds l1 l2c l3 bx cn
```

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
  eval/prepare.py              every model input: L1/L2/L3, back-transliterated and canonicalized L3
  eval/backends.py             NLLB (transformers) and any OpenAI-compatible chat endpoint
  eval/run.py                  one model x task x language over a list of conditions (resumable)
  eval/train_clf.py            MuRIL fine-tuning and the char n-gram baseline
  eval/train_lora.py           LoRA adapter for NLLB on romanized FLORES dev
  eval/fertility.py            tokenizer fertility per level
  eval/analyze.py              chrF++/F1, paired bootstrap, decomposition, consistency, costs
collection/                    manifest, form script, collection protocol
review/                        hand-check sheets and instructions
docs/                          data card, corpus statistics, related work
results/                       model outputs, summary.json, results tables
scripts/                       data download, IndicXlit setup and worker, release bundling
tests/                         romanizer regressions, collection kit, evaluation helpers
```

## Findings

Full tables with 95% bootstrap intervals: [results/README.md](results/README.md).

**Where the loss comes from (Phase 2)**

The test translates FLORES devtest into English and scores chrF++. NLLB-200 600M was run on 200 sentences, Gemma-3 4B on 30.
- **The script term dominates.** NLLB loses about 42 points going from native script to canonical romanization. Natural spelling variation adds only 1.3 (Malayalam) and 0.3 (Tamil) more.
- **Prior exposure shrinks the script term.** Gemma has seen romanized text, and its script term is 27–31 points. Its spelling term is not significant for Malayalam and 2.4 points for Tamil, 7% of the drop. The prediction that spelling would dominate does not hold for these models.
- **Spelling makes outputs unstable.** Two spellings of the same sentence get translations that agree at only 24–28 chrF++ (Gemma) or 52–54 (NLLB).
- **Romanization is not cheaper for modern tokenizers.** For Llama-3.1 and Qwen3, romanizing cuts tokens about fourfold. For Gemma-3, Sarvam-1, MuRIL, NLLB and GPT-4o, it costs more tokens than native script.

**Mitigations (Phase 3)**
- **Back-transliterating** L3 to native script with IndicXlit recovers 22–37 points. It ends within 4–5 points of the native-script ceiling.
- **Canonicalizing spelling** gives no significant accuracy gain, but it raises output agreement to 71–80.
- **Four-shot prompting** costs about 4x the prompt tokens and gains at most +2.2 points, not significant.
- **LoRA on NLLB** (rank 16, 1.9 CPU-hours) adds +8.2 and +6.5 points on L3 and leaves native script unchanged.

**Real romanized comments**
- Re-spelling a comment while keeping its words flips the char n-gram classifier's decision for 25% of Tamil comments.
- It flips zero-shot Gemma's decision for 38–45% of comments, and a MuRIL linear probe's for 14–32%.

## Phase 1 findings

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
