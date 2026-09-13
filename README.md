# Latin-script degradation in Malayalam and Tamil

Measuring how much language models lose when Malayalam and Tamil are typed in
Latin script rather than native script, and how much of that loss comes from
the change of script versus the inconsistency of informal spelling.

Every FLORES-200 sentence is rendered at three orthographic levels, with the
meaning held fixed:

| Level | Condition | Source |
|---|---|---|
| L1 | Native script | FLORES-200 as published |
| L2 | Canonical romanization, one spelling per word | Most frequent human spelling in Dakshina; rule-based romanizer for unseen words |
| L3 | Naturalistic romanization, k=5 writers per sentence | Spelling-variation model learned from Dakshina, applied per simulated writer |

## Status

Corpus construction (L1-L3 for FLORES dev and devtest) is implemented. Model
evaluation, metrics, mitigations, the DravidianCodeMix arm and the
human-written subset are not yet started.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/fetch_data.sh                       # FLORES-200 + Dakshina (~2 GB download)
PYTHONPATH=src .venv/bin/python -m latscript.translit.generate
.venv/bin/python -m pytest tests/
```

The generator writes `data/derived/flores_{ml,ta}_{dev,devtest}.jsonl`, one
sentence per line with `english`, `l1_native`, `l2_canonical`, `l3_writers`
and `lexicon_coverage`.

## Layout

```
src/latscript/
  corpora/flores.py      aligned FLORES-200 loading
  translit/rules.py      deterministic ml/ta -> ASCII romanizer (L2 fallback)
  translit/lexicon.py    human spelling distributions from Dakshina, with noise filter
  translit/noise.py      learned spelling-variation model and simulated writers (L3)
  translit/generate.py   builds the L1/L2/L3 corpus
  translit/distance.py   edit distance
tests/                   romanizer regression cases
scripts/fetch_data.sh    downloads source data
```

## Notes on the data

- **Human spelling agreement.** Always choosing the most common human spelling
  of a word matches another annotator only 69% of the time for Malayalam and
  63% for Tamil.
- **Dakshina annotation errors.** Some lexicon rows are off by one (ஹாரி
  romanized as "potter"). Pairs with normalized edit distance above 0.6 from
  the canonical romanization are dropped: about 2% of Malayalam and 6% of Tamil
  attestations.
- **Tamil voicing.** Tamil typists split on voicing (*pogum* vs *pokum*).
  Voicing intervocalic plosives matches human spellings more often (36.0% vs
  31.0% exact match), so L2 applies it.
- **L3 is conservative.** Simulated writers vary less than real ones
  (normalized edit distance about 0.07 vs 0.25 between humans), so degradation
  measured on L3 is a lower bound.

## Data sources

- [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200), CC BY-SA 4.0
- [Dakshina](https://github.com/google-research-datasets/dakshina), CC BY-SA 4.0
