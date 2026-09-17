from artifact_signals.data import ABSTAIN, Benchmark, Item
from artifact_signals.evaluate import run_model
from artifact_signals.judge import exact_match, lenient_choice
from artifact_signals.models import ConstantModel, FirstChoiceModel, NoisyFormatModel, OracleModel, RetrievalModel
from artifact_signals.render import baseline_rendering, partial_input_renderings
from artifact_signals.signals import (abstention_failure, compute_profile, constraint_gain, format_sensitivity,
                                      label_prior_skew, partial_input_accuracy, retrieval_dominance)
from artifact_signals.synth import generate


def leaky_pair(n=60):
    """Train/test with near-duplicate texts: the retrieval shortcut on purpose."""
    tr, te = [], []
    for k in range(n):
        lab = ["TRUE", "FALSE"][k % 2]
        prem = [f"Worlds: w0, w1. Accessibility: w0 can see w1. At w0: p is {'true' if k % 2 == 0 else 'false'}. At w1: q is true. Case {k}."]
        tr.append(Item(id=f"tr{k}", premises=prem, question="Is p true at w0?", options=["TRUE", "FALSE"], label=lab))
        te.append(Item(id=f"te{k}", premises=[prem[0].replace("Case", "Instance")], question="Is p true at w0?",
                       options=["TRUE", "FALSE"], label=lab))
    return Benchmark("leaky-train", tr), Benchmark("leaky-test", te)


def test_format_sensitivity_zero_for_oracle_and_positive_for_position_bias():
    b = generate(40, seed=5)
    gold = {i.id: i.label for i in b}
    fs, runs = format_sensitivity(OracleModel(gold, 0.8), b, n_boot=50)
    assert fs.value == 0.0 and len(runs) == 5 and fs.details["item_agreement"] == 1.0
    fs2, _ = format_sensitivity(FirstChoiceModel(), b, n_boot=50)
    assert fs2.value > 0.0


def test_partial_input_flags_constant_model_on_skewed_benchmark():
    items = [Item(id=f"i{k}", premises=["x"], question="?", options=["yes", "no"], label="yes" if k < 30 else "no") for k in range(40)]
    b = Benchmark("skew", items)
    pia = partial_input_accuracy(ConstantModel("yes"), b)
    assert abs(pia.value - 0.25) < 1e-9 and pia.details["chance"] == 0.5
    run = run_model(ConstantModel("yes"), b, baseline_rendering())
    lps = label_prior_skew(b, run)
    assert abs(lps.value - 0.25) < 1e-9 and abs(lps.details["tv_label_vs_uniform"] - 0.25) < 1e-9
    gold = {i.id: i.label for i in b}
    pia_oracle = partial_input_accuracy(OracleModel(gold, 1.0), b)   # oracle ignores input, so it also "solves" partial input
    assert pia_oracle.value == 0.5


def test_retrieval_dominance_high_on_leaky_low_on_control():
    tr, te = leaky_pair()
    gold = {i.id: i.label for i in te}
    llm = run_model(OracleModel(gold, 0.9, seed=3), te, baseline_rendering())
    rd = retrieval_dominance(tr, te, llm)
    assert rd.details["per_method"]["fuzzy"]["accuracy"] == 1.0 and rd.value >= 1.0
    ctrl = generate(40, seed=9)
    ctrl_tr, ctrl_te = ctrl.split(0.5, seed=1)
    gold2 = {i.id: i.label for i in ctrl_te}
    llm2 = run_model(OracleModel(gold2, 0.9, seed=3), ctrl_te, baseline_rendering())
    rd2 = retrieval_dominance(ctrl_tr, ctrl_te, llm2)
    assert rd2.value < rd.value and rd2.details["per_method"]["fuzzy"]["accuracy"] < 0.8


def test_retrieval_model_is_dominated_by_retrieval():
    tr, te = leaky_pair()
    llm = run_model(RetrievalModel(tr.items), te, baseline_rendering())
    rd = retrieval_dominance(tr, te, llm)
    assert llm.accuracy == 1.0 and rd.details["per_method"]["fuzzy"]["agreement_with_llm"] == 1.0


def test_constraint_gain_positive_svg_for_noisy_format_model():
    b = generate(200, seed=11)
    gold = {i.id: i.label for i in b}
    sig, runs = constraint_gain(NoisyFormatModel(gold, 0.6, 0.5, seed=2), b, em_judge=exact_match, sem_judge=lenient_choice)
    d = sig.details
    assert d["constraint_gain"] > 0.15 and abs(d["semantic_gain"]) < 0.05 and sig.value > 0.1
    assert d["dissociation_rate"] > 0.0
    sig2, _ = constraint_gain(OracleModel(gold, 0.7), b, em_judge=exact_match, sem_judge=lenient_choice)
    assert abs(sig2.value) < 1e-9


def test_abstention_failure_detects_never_abstaining_model():
    b = generate(60, seed=4, ood_frac=0.2)
    gold = {i.id: i.label for i in b}
    af = abstention_failure(OracleModel(gold, 1.0), b)          # oracle knows ABSTAIN labels: perfect abstention
    assert af.value == 0.0 and af.details["n_ood"] == 12
    af2 = abstention_failure(ConstantModel("TRUE"), b)
    assert af2.value == 1.0
    assert abstention_failure(ConstantModel("TRUE"), generate(10, seed=4)).value is None


def test_profile_gate_marks_other_signals():
    b = generate(40, seed=6, ood_frac=0.1)
    tr, te = b.split(0.4, seed=0)
    gold = {i.id: i.label for i in te}
    prof = compute_profile(OracleModel(gold, 0.8), te, train=tr, sem_judge=lenient_choice, n_boot=30)
    assert not prof.gated and all(not s.gated for s in prof.signals.values())
    assert set(prof.vector()) == {"format_sensitivity", "partial_input_accuracy", "label_prior_skew",
                                  "retrieval_dominance", "semantic_validity_gap", "abstention_failure"}
    prof2 = compute_profile(FirstChoiceModel(), te, train=tr, n_boot=30, fs_gate=-1.0)  # force the gate
    assert prof2.gated and prof2.signals["label_prior_skew"].gated and not prof2.signals["format_sensitivity"].gated
    assert "Artifact Risk Profile" in prof2.summary()
