import json
from collections import Counter

import pytest

from artifact_signals.data import ABSTAIN, Benchmark, Item
from artifact_signals.evaluate import run_model
from artifact_signals.interventions import (ABSTENTION, LEAKAGE, REBALANCE, inject_ood, invariance_check, leakage_resplit,
                                            make_unanswerable, novel_terms_only, rebalance_and_filter, rerender_score)
from artifact_signals.ledger import Ledger, Prediction, lookup, score_ledger
from artifact_signals.models import ConstantModel, OracleModel
from artifact_signals.render import baseline_rendering, partial_input_renderings, standard_renderings
from artifact_signals.signals import abstention_failure, partial_input_accuracy, retrieval_dominance
from artifact_signals.synth import generate
from tests.test_signals import leaky_pair


def test_leakage_resplit_drops_near_duplicates_and_passes_check():
    tr, te = leaky_pair()
    kept, dropped = leakage_resplit(tr, te, max_similarity_=0.8)
    assert len(dropped) == len(te) and len(kept) == 0
    novel = novel_terms_only(tr, te, min_novel=1)
    assert len(novel) == len(te)  # "Instance" is a novel token in every test item
    ctrl = generate(40, seed=3)
    a, b = ctrl.split(0.5, seed=0)
    kept2, dropped2 = leakage_resplit(a, b, max_similarity_=0.95)
    gold = {i.id: i.label for i in kept2}
    rd = retrieval_dominance(a, kept2, run_model(OracleModel(gold, 0.9), kept2, baseline_rendering()))
    assert len(kept2) > 0 and LEAKAGE.manipulation_check(rd)


def test_rebalance_and_filter_balances_and_removes_solved():
    items = [Item(id=f"i{k}", premises=["x"], question="?", options=["yes", "no"], label="yes" if k < 30 else "no") for k in range(40)]
    b = Benchmark("skew", items)
    partial = [run_model(ConstantModel("yes"), b, r) for r in partial_input_renderings("choice")]
    after = rebalance_and_filter(b, partial)
    assert len(after) == 0  # every 'yes' item is partial-input-solvable, so nothing balances against 'no'
    after2 = rebalance_and_filter(b, None)
    assert Counter(i.label for i in after2) == {"yes": 10, "no": 10}
    gold = {i.id: i.label for i in after2}
    pia = partial_input_accuracy(ConstantModel("yes"), after2)
    assert REBALANCE.manipulation_check(pia)


def test_inject_ood_and_abstention_check():
    base = generate(40, seed=2)
    probes = generate(40, seed=8, ood_frac=0.5)
    src = Benchmark("probes", probes.ood_items())
    mixed = inject_ood(base, src, fraction=0.2)
    assert len(mixed.ood_items()) == 10 and len(mixed.in_distribution()) == 40
    assert all(i.label == ABSTAIN for i in mixed.ood_items())
    gold = {i.id: i.label for i in mixed}
    assert ABSTENTION.manipulation_check(abstention_failure(OracleModel(gold, 1.0), mixed))
    assert not ABSTENTION.manipulation_check(abstention_failure(ConstantModel("TRUE"), mixed))
    un = make_unanswerable(base, fraction=0.1)
    assert len(un.ood_items()) == 4 and all(i.premises == [] for i in un.ood_items())


def test_rerender_and_invariance():
    b = generate(30, seed=12)
    gold = {i.id: i.label for i in b}
    m1, m2 = OracleModel(gold, 0.7, seed=1, name="m1"), OracleModel(gold, 0.7, seed=2, name="m2")
    rs = standard_renderings(0)
    r1 = rerender_score(m1, b, renderings=rs)
    assert abs(r1["accuracy_mean_over_renderings"] - r1["per_rendering"]["baseline"]) < 1e-9
    runs = {"m1": r1["runs"], "m2": rerender_score(m2, b, renderings=rs)["runs"]}
    ids = [i.id for i in b]
    chk = invariance_check(runs, ids, ids[:15], min_rho=0.6)
    assert set(chk) >= {"agreement_before", "agreement_after", "passed"}
    same = {"m1": r1["runs"], "m1b": r1["runs"]}
    assert invariance_check(same, ids, ids)["agreement_after"] == 1.0


def test_prediction_semantics():
    assert Prediction("a", "b", "i", "x.delta", "down", 0.05, 0.7).holds(-0.06)
    assert not Prediction("a", "b", "i", "x.delta", "down", 0.05, 0.7).holds(-0.04)
    assert Prediction("a", "b", "i", "x.delta", "flat", 0.03, 0.7).holds(0.02)
    assert Prediction("a", "b", "i", "x", "above", 0.6, 0.7).holds(0.61)
    assert Prediction("a", "b", "i", "x", "below", 0.15, 0.7).holds(0.1)
    with pytest.raises(ValueError):
        Prediction("a", "b", "i", "x", "sideways", 0.1, 0.5)
    with pytest.raises(ValueError):
        Prediction("a", "b", "i", "x", "up", 0.1, 1.5)


def test_ledger_hash_is_stable_and_detects_tampering(tmp_path):
    led = Ledger("t", [Prediction("P1", "b", "i", "acc.delta", "down", 0.05, 0.7),
                       Prediction("P2", "b", "i", "gap.delta", "up", 0.05, 0.6, "metric_specific")], created="2026-01-01T00:00:00+00:00")
    p = tmp_path / "l.json"
    h = led.save(p)
    assert Ledger.load(p).sha256() == h
    d = json.loads(p.read_text())
    d["predictions"][0]["confidence"] = 0.99
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        Ledger.load(p)


def test_ledger_scoring_and_naive_comparison():
    led = Ledger("t", [
        Prediction("P1", "b", "i", "acc.delta", "down", 0.05, 0.8),
        Prediction("P2", "b", "i", "gap.delta", "up", 0.05, 0.6, "metric_specific"),
        Prediction("P3", "b", "i", "sem.delta", "flat", 0.03, 0.7, "metric_specific"),
        Prediction("P4", "b", "i", "corr", "below", 0.15, 0.9, "manipulation_check"),
        Prediction("P5", "b", "j", "acc.delta", "down", 0.05, 0.5),
        Prediction("P6", "c", "i", "acc.delta", "flat", 0.03, 0.85, "specificity"),
    ])
    results = {"b": {"i": {"acc": {"delta": -0.10}, "gap": {"delta": 0.07}, "sem": {"delta": -0.10}, "corr": 0.05}},
               "c": {"i": {"acc": {"delta": -0.01}}}}
    s = score_ledger(led, results, voided={"P6"})
    assert s["n_scored"] == 4 and s["n_voided"] == 1 and s["n_missing"] == 1
    assert s["hits"] == 3 and abs(s["hit_rate"] - 0.75) < 1e-9
    assert s["by_family"]["metric_specific"]["hit_rate"] == 0.5
    c = s["vs_naive_on_change_predictions"]
    assert c["n"] == 3 and c["model_hit_rate"] == 2 / 3 and c["naive_hit_rate"] == 2 / 3
    assert lookup(results, "b", "i", "acc.delta") == -0.10 and lookup(results, "b", "i", "nope") is None
    assert 0 <= s["brier"] <= 1


def test_shipped_ledger_loads_and_hashes():
    led = Ledger.load("ledger/paper3_ledger.json", verify=False)
    assert len(led.predictions) == 14 and len({p.id for p in led.predictions}) == 14
    assert len(led.sha256()) == 64
