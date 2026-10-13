import numpy as np
import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("requests")

from latscript.eval.analyze import bootstrap, f1, group_of  # noqa: E402
from latscript.eval.run import expand, parse_yes_no, source_field  # noqa: E402


def test_condition_names():
    assert expand("mt", ["l1", "l3", "fs"], 2) == ["l1", "l3_0", "l3_1", "fs_0", "fs_1"]
    assert expand("clf", ["l3", "s3"], 2) == ["l3", "s3_0", "s3_1"]
    assert source_field("fs_3") == "l3_3"
    assert source_field("lora_l3_1") == "l3_1"
    assert source_field("l2c") == "l2c"
    assert group_of("lora_l3_4") == "lora_l3"
    assert group_of("l2c") == "l2c"


def test_parse_yes_no():
    assert parse_yes_no("Yes.") == 1
    assert parse_yes_no("no") == 0
    assert parse_yes_no("I cannot tell") is None


def test_f1():
    y = np.array([0, 0, 1, 1])
    assert f1(y, y) == 100
    assert f1(y, np.array([0, 0, 0, 0])) == pytest.approx(100 * (2 / 3) / 2)


def test_bootstrap_sign():
    a = np.array([10.0, 12, 11, 13, 12])
    d = bootstrap(a, a - 2)
    assert d["diff"] == 2 and d["p"] < 0.05 and d["ci"][0] == pytest.approx(2)
