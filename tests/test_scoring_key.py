"""D-006 scoring_key_validity: is the key the benchmark scores against itself trustworthy?"""

from artifact_signals.benchmarks import cod
from artifact_signals.data import Benchmark, Item
from artifact_signals.signals import scoring_key_validity
from artifact_signals.synth import checker_key, generate


def test_scoring_key_validity_is_clean_on_synthmodal_control():
    bench = generate(200, seed=3, ood_frac=0.1)
    s = scoring_key_validity(bench, independent_key=checker_key)
    assert s.value == 0.0 and s.level == "dataset"
    assert s.details["rule_null_rate"] == 0.0 and s.details["n_keyed"] == 200


def test_scoring_key_validity_counts_corrupted_labels():
    bench = generate(200, seed=3, ood_frac=0.1)
    flip = {"TRUE": "FALSE", "FALSE": "TRUE"}
    bad = [Item.from_dict({**i.to_dict(), "label": flip[i.label]}) if k < 7 and not i.ood else i
           for k, i in enumerate(bench.items)]
    n_flipped = sum(1 for a, b in zip(bad, bench.items) if a.label != b.label)
    s = scoring_key_validity(Benchmark("corrupt", bad), independent_key=checker_key)
    assert n_flipped > 0 and s.value == n_flipped / 200
    assert sorted(s.details["disagreeing_ids"]) == sorted(a.id for a, b in zip(bad, bench.items) if a.label != b.label)


def test_rule_null_rate_counts_items_the_rule_cannot_key():
    items = [Item(id=str(k), question="q", label="A", options=["A", "B"], meta={"code": c})
             for k, c in enumerate(["A1", "", "B2", "?"])]
    s = scoring_key_validity(Benchmark("t", items), rule=lambda i: "A" if i.meta["code"][:1].isalpha() else None)
    assert s.value is None and s.details["rule_null_rate"] == 0.5


def test_cod_label_space_asymmetry_is_the_injury_artifact():
    # COD v2: the dictionary can never emit Injury (or Special); the models can and did (Sonnet 15, Qwen 31).
    prof = cod.profile("M_comm_sonnet5", retrieval_methods=("char3",))
    s = prof.signals["scoring_key_validity"]
    assert s.value is None  # no independent key offline: labels come from the codes by the scoring rule itself
    d = s.details
    assert d["asymmetric_labels"] == ["Injury", "Special"] and abs(d["label_space_asymmetry"] - 2 / 22) < 1e-12
    assert d["predictions_in_asymmetric_labels"] == {"claude-sonnet-5": 15, "qwen2.5:14b": 31, "fuzzy-char3-dictionary": 0}
    assert d["rule_null_rate"] == 0.0


def test_compute_profile_reports_scoring_key_validity_only_with_a_key():
    from artifact_signals.models import OracleModel
    from artifact_signals.signals import compute_profile
    bench = generate(60, seed=5, ood_frac=0.1)
    gold = {i.id: i.label for i in bench}
    with_key = compute_profile(OracleModel(gold, 0.8), bench, n_boot=20, independent_key=checker_key)
    assert with_key.signals["scoring_key_validity"].value == 0.0
    assert "scoring_key_validity" not in compute_profile(OracleModel(gold, 0.8), bench, n_boot=20).signals


def test_chapter_rule_returns_none_for_malformed_codes():
    assert cod.chapter_from_code("K55.902") == "Digestive" and cod.chapter_from_code("D50.1") == "Blood"
    for bad in ("Dxx", "C7A", "I-1", "D5", "", "9A0"):
        assert cod.chapter_from_code(bad) is None, bad


def test_folded_scoring_removes_injury_from_the_model_emit_space():
    prof = cod.profile("M_comm_sonnet5", retrieval_methods=("char3",), fold_injury=True)
    d = prof.signals["scoring_key_validity"].details
    assert d["asymmetric_labels"] == ["Special"] and abs(d["label_space_asymmetry"] - 1 / 22) < 1e-12
    assert set(d["predictions_in_asymmetric_labels"].values()) == {0}
