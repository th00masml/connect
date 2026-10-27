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


def test_stable_json_keeps_tiny_p_values():
    # 12 decimal places would write 8.27e-25 as 0.0; 12 significant digits keep it.
    out = json.loads(stable_json({"p": 8.271806125530277e-25, "q": 5.960464477539063e-08}))
    assert out == {"p": 8.27180612553e-25, "q": 5.96046447754e-08}


def test_probe_sentence_follows_the_data():
    from artifact_signals.paper12 import probe_sentence
    empty = probe_sentence({"n": 30, "answered": 0, "exact_code": 0.0, "verdict": "INCONCLUSIVE: no answers"})
    assert "30 of 30 probe calls returned empty" in empty and "not a score" in empty
    some = probe_sentence({"n": 30, "answered": 12, "exact_code": 0.25, "verdict": "OK: answered"})
    assert "12 of 30" in some and "0.25" in some and "zero answered" not in some and "not a score" not in some
