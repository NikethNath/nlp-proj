"""Phase 3 plumbing: input rewrites, variant loading, recovery arithmetic.

Toy lookup tables stand in for IndicXlit, so no model is needed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from latscript.eval import conditions
from latscript.mitigate import fidelity, report
from latscript.mitigate.transforms import Rewriter


def _rewriter():
    rw = Rewriter("ml")
    # Two spellings of one word map to the same native form, which romanizes
    # back to a single canonical spelling.
    rw.to_native = {"veedu": "വീട്", "vidu": "വീട്", "ente": "എന്റെ"}
    rw.to_latin = {"വീട്": "veedu", "എന്റെ": "ente"}
    return rw


def test_backtransliteration_keeps_punctuation_and_unknown_words():
    rw = _rewriter()
    assert rw.bt("Ente veedu, 2024!") == "എന്റെ വീട്, 2024!"
    assert rw.bt("ente xyz") == "എന്റെ xyz"


def test_roundtrip_normalizes_spelling_variants():
    rw = _rewriter()
    assert rw.rt("ente vidu") == rw.rt("ente veedu") == "ente veedu"


def test_texts_variant_replaces_only_romanized_conditions():
    row = {"l1_native": "N", "l2_canonical": "a", "l2_lexical": "b",
           "l3_writers": ["c0", "c1", "c2", "c3", "c4"],
           **{f"bt_{c}": f"BT-{c}" for c in conditions.CONDITIONS if c != "l1"}}
    t = conditions.texts(row, "bt")
    assert t["l1"] == "N"
    assert t["l3_3"] == "BT-l3_3"
    assert conditions.texts(row)["l3_3"] == "c3"


def test_word_alignment_skips_unequal_lengths():
    assert fidelity._differ(["a", "b"], ["a", "c"]) == (1, 2)
    assert fidelity._differ(["a"], ["a", "b"]) == (0, 0)


def test_recovered_share():
    base = {"l1": 0.9, "l3": 0.3}
    assert report._recovered(base, {"l3": 0.6}) == "+50%"
    assert report._recovered(base, {"l3": 0.24}) == "-10%"


def test_results_dir_routes_phase3():
    assert conditions.results_dir("mt").name == "mt"
    assert conditions.results_dir("mt", True).parent.name == "mitigate"
