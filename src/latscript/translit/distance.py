"""Edit distance, shared by the lexicon filter and the evaluation metrics."""
from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    return edit_distance(a, b) / max(len(a), len(b))
