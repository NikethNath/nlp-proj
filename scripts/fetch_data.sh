#!/usr/bin/env bash
# Download FLORES-200 and the Malayalam/Tamil parts of Dakshina into data/raw.
# Both archives are skipped if already present. Dakshina is ~2 GB; only the
# lexicons and word-aligned romanized text are extracted from it.
set -euo pipefail

RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$RAW"
cd "$RAW"

FLORES_URL=https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
DAKSHINA_URL=https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar

if [ ! -d flores200_dataset ]; then
  [ -f flores200.tar.gz ] || curl -fL -o flores200.tar.gz "$FLORES_URL"
  tar xzf flores200.tar.gz
fi
echo "FLORES-200: $RAW/flores200_dataset"

if [ ! -f dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv ] ||
   [ ! -f dakshina_dataset_v1.0/ml/lexicons/ml.translit.sampled.train.tsv ]; then
  [ -f dakshina.tar ] || curl -fL -o dakshina.tar "$DAKSHINA_URL"
  tar xf dakshina.tar --wildcards \
    'dakshina_dataset_v1.0/ml/lexicons/*.translit.sampled.*.tsv' \
    'dakshina_dataset_v1.0/ta/lexicons/*.translit.sampled.*.tsv' \
    'dakshina_dataset_v1.0/ml/romanized/*.rejoined.aligned.cased_nopunct.tsv' \
    'dakshina_dataset_v1.0/ta/romanized/*.rejoined.aligned.cased_nopunct.tsv'
fi
echo "Dakshina:   $RAW/dakshina_dataset_v1.0"
