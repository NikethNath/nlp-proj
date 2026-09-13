"""Regression cases for the canonical romanizer.

Each entry is a spelling a literate Malayalam/Tamil typist would actually
produce. They encode the phonological rules the naive abugida walk gets wrong:
Malayalam's word-final half-u, nasal-cluster voicing, geminate-nasal collapse,
and Tamil's position-dependent value of ca.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
from latscript.translit.rules import transliterate

MALAYALAM = [
    ("നമസ്കാരം", "namaskaaram"),
    ("കേരളം", "keralam"),
    ("മലയാളം", "malayaalam"),
    ("പുസ്തകം", "pusthakam"),      # intervocalic ka stays voiceless
    ("സന്തോഷം", "santhosham"),
    ("ഞാൻ", "njaan"),               # chillu carries no inherent vowel
    ("അവൾ", "aval"),
    ("കുട്ടി", "kutti"),            # geminate blocks voicing
    ("എന്താണ്", "enthaanu"),        # nta is written nth, plus final half-u
    ("വീട്", "veedu"),              # final half-u licenses voicing
    ("ഉണ്ട്", "undu"),              # nasal cluster + final half-u
    ("ഞങ്ങളുടെ", "njangalude"),     # geminate nasal collapses; ta voices
]

TAMIL = [
    ("வணக்கம்", "vanakkam"),
    ("தமிழ்", "thamizh"),
    ("புத்தகம்", "puththagam"),     # intervocalic ka voices
    ("நான்", "naan"),
    ("போகும்", "pogum"),
    ("சென்னை", "chennai"),          # ca is an affricate word-initially
    ("சிறு", "chiru"),
    ("இசை", "isai"),                # ...but a sibilant between vowels
    ("எங்கள்", "engal"),            # nasal already voices the cluster
]


@pytest.mark.parametrize("native,expected", MALAYALAM)
def test_malayalam(native, expected):
    assert transliterate(native, "ml") == expected


@pytest.mark.parametrize("native,expected", TAMIL)
def test_tamil(native, expected):
    assert transliterate(native, "ta") == expected


def test_passthrough_preserves_latin_and_punctuation():
    assert transliterate("കേരളം, 2024 (India)!", "ml") == "keralam, 2024 (India)!"


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        transliterate("test", "kn")


def test_sentence_tokenizer_keeps_vowel_signs_inside_words():
    from latscript.translit.lexicon import _TOKEN
    words = [p for p in _TOKEN.split("பிற வல்லுனர்கள், ഞങ്ങളുടെ വീട്.") if p and not _TOKEN.fullmatch(p)]
    assert words == ["பிற", "வல்லுனர்கள்", "ഞങ്ങളുടെ", "വീട്"]
