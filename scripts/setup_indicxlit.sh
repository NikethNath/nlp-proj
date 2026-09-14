#!/usr/bin/env bash
# IndicXlit (ai4bharat-transliteration) depends on fairseq 0.12, which does not
# install on current Python. It gets its own Python 3.10 environment, separate
# from the main one, and is called as a subprocess by latscript.translit.indicxlit.
# Set XLIT_VENV to install somewhere other than .venv-xlit (then point
# INDICXLIT_PYTHON at its interpreter).
set -euo pipefail
cd "$(dirname "$0")/.."

UV="${UV:-uv}"
command -v "$UV" >/dev/null || { python3 -m pip install --user uv; UV=uv; }

VENV="${XLIT_VENV:-.venv-xlit}"
"$UV" venv --clear --python 3.10 "$VENV"
PY="$VENV/bin/python"
"$UV" pip install --python "$PY" "torch==2.1.2" --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$PY" "numpy<2" "setuptools<70" cython
"$UV" pip install --python "$PY" ai4bharat-transliteration

# The engine imports urduhack at module load purely for Shahmukhi (Urdu)
# normalization, and urduhack drags in a TensorFlow stack that no longer
# imports. Neither language here needs it, so replace it with a stub.
"$UV" pip uninstall --python "$PY" urduhack
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p "$SITE/urduhack"
cat > "$SITE/urduhack/__init__.py" <<'STUB'
"""Stub: IndicXlit imports this only for Urdu, which this project never uses."""
def normalize(text):
    raise NotImplementedError("urduhack stub: Urdu normalization is not installed")
STUB

"$PY" -c "from ai4bharat.transliteration import XlitEngine; print('IndicXlit OK')"
