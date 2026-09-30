"""Degradation decomposition with paired bootstrap intervals.

Every metric here is a function of summed per-sentence statistics: a mean is
``sum(value) / sum(1)``, corpus chrF++ is computed from summed n-gram counts,
retrieval accuracy is a mean of hits. That lets one bootstrap serve them all:
resample sentence indices, sum each condition's statistics over the sample,
and apply the metric.

Levels and gaps (loss is positive when the model does worse):

    script gap    L1 -> L2    same words, Latin instead of native script
    spelling gap  L2 -> L3    same script, inconsistent instead of fixed spelling
    total gap     L1 -> L3

L3 is the mean of the five writers' corpus scores, so a resample draws the
same sentences for every writer. Gaps are reported against both L2s; see the
data card for why they differ.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .conditions import CONDITIONS, N_WRITERS

Metric = Callable[[np.ndarray], float]


def mean_metric(summed: np.ndarray) -> float:
    """Metric over summed (value, 1) statistics: the plain mean."""
    return float(summed[0] / summed[1])


def as_mean_stats(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.stack([v, np.ones_like(v)], axis=-1)


def level_scores(summed: Mapping[str, np.ndarray], metric: Metric) -> dict[str, float]:
    """Scores per level from per-condition *summed* statistics."""
    l3 = [metric(summed[f"l3_{i}"]) for i in range(N_WRITERS)]
    return {"l1": metric(summed["l1"]),
            "l2_canonical": metric(summed["l2_canonical"]),
            "l2_lexical": metric(summed["l2_lexical"]),
            "l3": float(np.mean(l3)),
            "l3_writers": l3}


def gaps(levels: Mapping[str, float], higher_is_better: bool = True) -> dict[str, float]:
    sign = 1.0 if higher_is_better else -1.0
    out = {"total": sign * (levels["l1"] - levels["l3"])}
    for l2 in ("l2_canonical", "l2_lexical"):
        tag = l2.split("_")[1]
        out[f"script_{tag}"] = sign * (levels["l1"] - levels[l2])
        out[f"spelling_{tag}"] = sign * (levels[l2] - levels["l3"])
    return out


def decompose(stats: Mapping[str, np.ndarray], metric: Metric, higher_is_better: bool = True,
              n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimates and percentile bootstrap intervals for levels and gaps.

    `stats` maps every condition to an array shaped (n_sentences, k) whose
    rows are aligned across conditions.
    """
    missing = [c for c in CONDITIONS if c not in stats]
    if missing:
        raise ValueError(f"missing conditions {missing}")
    n = len(stats["l1"])
    if any(len(v) != n for v in stats.values()):
        raise ValueError("conditions are not aligned")

    point_levels = level_scores({c: v.sum(0) for c, v in stats.items()}, metric)
    point_gaps = gaps(point_levels, higher_is_better)

    rng = np.random.default_rng(seed)
    boot_levels = {k: [] for k in ("l1", "l2_canonical", "l2_lexical", "l3")}
    boot_gaps = {k: [] for k in point_gaps}
    for _ in range(n_boot):
        # Multinomial weights == resampling indices with replacement, but one
        # matrix product per condition instead of fancy indexing.
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        lv = level_scores({c: w @ v for c, v in stats.items()}, metric)
        for k in boot_levels:
            boot_levels[k].append(lv[k])
        for k, g in gaps(lv, higher_is_better).items():
            boot_gaps[k].append(g)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def interval(xs):
        return [round(float(np.percentile(xs, lo)), 4), round(float(np.percentile(xs, hi)), 4)]

    return {
        "n": n,
        "higher_is_better": higher_is_better,
        "levels": {k: {"score": round(point_levels[k], 4), "ci": interval(boot_levels[k])}
                   for k in boot_levels},
        "l3_writers": [round(x, 4) for x in point_levels["l3_writers"]],
        "gaps": {k: {"loss": round(point_gaps[k], 4), "ci": interval(boot_gaps[k])}
                 for k in point_gaps},
    }


def paired(fa: Callable[[np.ndarray], float], fb: Callable[[np.ndarray], float], n: int,
           n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Two scores on the same n items and their difference a - b, with intervals.

    `fa` and `fb` take an array of item indices (a bootstrap resample) and
    return a score, so any metric works, including non-additive ones like AUROC.
    """
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    a, b = fa(full), fb(full)
    ba, bb = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ba.append(fa(idx))
        bb.append(fb(idx))
    ba, bb = np.array(ba), np.array(bb)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def interval(xs):
        return [round(float(np.percentile(xs, lo)), 4), round(float(np.percentile(xs, hi)), 4)]

    return {"n": n, "a": {"score": round(a, 4), "ci": interval(ba)},
            "b": {"score": round(b, 4), "ci": interval(bb)},
            "a_minus_b": {"diff": round(a - b, 4), "ci": interval(ba - bb)}}
