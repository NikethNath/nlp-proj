"""Phase 2 evaluation plumbing: decomposition, bootstrap and output cleaning.

Synthetic scores only, so no model or corpus is needed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from latscript.eval import conditions, stats
from latscript.eval.translate import clean


def _stats(means, n=200, seed=0):
    """Per-condition (value, 1) stats with the given condition means plus noise."""
    rng = np.random.default_rng(seed)
    return {c: stats.as_mean_stats(m + rng.normal(0, 0.01, n)) for c, m in means.items()}


MEANS = {"l1": 0.9, "l2_canonical": 0.6, "l2_lexical": 0.7,
         **{f"l3_{i}": 0.5 + 0.02 * i for i in range(5)}}


def test_conditions_cover_all_writers():
    row = {"l1_native": "a", "l2_canonical": "b", "l2_lexical": "c",
           "l3_writers": ["d0", "d1", "d2", "d3", "d4"]}
    t = conditions.texts(row)
    assert tuple(t) == conditions.CONDITIONS
    assert {conditions.level(c) for c in t} == set(conditions.LEVELS)


def test_decomposition_adds_up():
    res = stats.decompose(_stats(MEANS), stats.mean_metric, n_boot=200)
    lv, gp = res["levels"], res["gaps"]
    assert lv["l3"]["score"] == pytest.approx(0.54, abs=0.005)       # mean of the five writers
    assert gp["script_canonical"]["loss"] == pytest.approx(0.3, abs=0.005)
    assert gp["spelling_lexical"]["loss"] == pytest.approx(0.16, abs=0.005)
    for tag in ("canonical", "lexical"):
        assert gp[f"script_{tag}"]["loss"] + gp[f"spelling_{tag}"]["loss"] == \
            pytest.approx(gp["total"]["loss"], abs=1e-3)
    lo, hi = gp["total"]["ci"]
    assert lo < gp["total"]["loss"] < hi


def test_lower_is_better_flips_the_sign():
    res = stats.decompose(_stats(MEANS), stats.mean_metric, higher_is_better=False, n_boot=50)
    assert res["gaps"]["total"]["loss"] < 0


def test_decompose_rejects_misaligned_conditions():
    s = _stats(MEANS)
    s["l3_4"] = s["l3_4"][:-1]
    with pytest.raises(ValueError):
        stats.decompose(s, stats.mean_metric, n_boot=10)


def test_paired_difference_interval():
    a = np.r_[np.ones(80), np.zeros(20)]
    b = np.r_[np.ones(60), np.zeros(40)]
    res = stats.paired(lambda i: a[i].mean(), lambda i: b[i].mean(), 100, n_boot=500)
    assert res["a_minus_b"]["diff"] == pytest.approx(0.2)
    lo, hi = res["a_minus_b"]["ci"]
    assert 0 < lo < 0.2 < hi


@pytest.mark.parametrize("raw,expected", [
    ("The cat sat.", "The cat sat."),
    ("Translation: The cat sat.\n\nNote: literal.", "The cat sat."),
    ('"The cat sat."', "The cat sat."),
    ("\n  English translation: Hello  \n", "Hello"),
])
def test_clean_llm_output(raw, expected):
    assert clean(raw) == expected


def test_chrf_statistics_sum_to_corpus_score():
    sacrebleu = pytest.importorskip("sacrebleu")
    from latscript.eval.translate import chrf_metric, chrf_stats
    hyps = ["the cat sat on the mat", "a dog barked", "hello world"]
    refs = ["the cat is on the mat", "the dog barked loudly", "hello there world"]
    corpus = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    assert chrf_metric(chrf_stats(hyps, refs).sum(0)) == pytest.approx(corpus)
