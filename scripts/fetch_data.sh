#!/usr/bin/env bash
# Download every source dataset into data/raw. Anything already present is
# skipped. Total download is about 2.3 GB (Dakshina is 2 GB, Aksharantar 230 MB);
# only the Malayalam and Tamil parts are kept.
set -euo pipefail

RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$RAW"
cd "$RAW"

HF=https://huggingface.co/datasets
FLORES_URL=https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
DAKSHINA_URL=https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar

fetch() { [ -f "$2" ] || curl -fL --create-dirs -o "$2" "$1"; }

# FLORES-200
if [ ! -d flores200_dataset ]; then
  fetch "$FLORES_URL" flores200.tar.gz
  tar xzf flores200.tar.gz
fi
echo "FLORES-200:       $RAW/flores200_dataset"

# Dakshina: lexicons and word-aligned romanized text only
if [ ! -f dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv ] ||
   [ ! -f dakshina_dataset_v1.0/ml/lexicons/ml.translit.sampled.train.tsv ]; then
  fetch "$DAKSHINA_URL" dakshina.tar
  tar xf dakshina.tar --wildcards \
    'dakshina_dataset_v1.0/ml/lexicons/*.translit.sampled.*.tsv' \
    'dakshina_dataset_v1.0/ta/lexicons/*.translit.sampled.*.tsv' \
    'dakshina_dataset_v1.0/ml/romanized/*.rejoined.aligned.cased_nopunct.tsv' \
    'dakshina_dataset_v1.0/ta/romanized/*.rejoined.aligned.cased_nopunct.tsv'
fi
echo "Dakshina:         $RAW/dakshina_dataset_v1.0"

# DravidianCodeMix offensive-language data
D=dravidiancodemix
for split in train dev test; do
  fetch "$HF/IsaacRodgz/DravidianCodeMix-Dataset/resolve/main/data/$split-00000-of-00001.parquet" \
        "$D/isaac_$split-00000-of-00001.parquet"
done
for split in train validation; do
  fetch "$HF/community-datasets/offenseval_dravidian/resolve/main/malayalam/$split-00000-of-00001.parquet" \
        "$D/offenseval_malayalam_$split.parquet"
done
echo "DravidianCodeMix: $RAW/$D"

# Aksharantar, flattened to a compact TSV per language
for pair in ml:mal ta:tam; do
  lang=${pair%%:*}; code=${pair##*:}
  if [ ! -f "aksharantar/$lang.tsv" ]; then
    fetch "$HF/ai4bharat/Aksharantar/resolve/main/$code.zip" "aksharantar/$code.zip"
    PYTHONPATH="$RAW/../../src" python3 -c \
      "from latscript.translit.aksharantar import flatten; flatten('$lang')"
  fi
done
echo "Aksharantar:      $RAW/aksharantar"
