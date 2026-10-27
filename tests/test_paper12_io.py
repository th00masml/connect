import json

from artifact_signals.paper12 import stable_json


def test_stable_json_absorbs_last_ulp_float_drift():
    # CPython 3.12 changed float summation; the same mean can differ in the 16th digit.
    a = stable_json({"x": -0.2409023043237984, "y": [0.4075757575757575]})
    b = stable_json({"x": -0.24090230432379842, "y": [0.4075757575757576]})
    assert a == b


def test_stable_json_keeps_non_floats():
    out = json.loads(stable_json({"n": 3, "none": None, "flag": True, "s": "0.1", "nested": {"k": [1, 2.5]}}))
    assert out == {"n": 3, "none": None, "flag": True, "s": "0.1", "nested": {"k": [1, 2.5]}}
    assert isinstance(out["n"], int)
