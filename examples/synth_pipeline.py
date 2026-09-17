"""End-to-end dry run of the protocol on SynthModal-Control with dummy models.

Phase 1: diagnose a leaky benchmark and a clean control, write a ledger, hash it.
Phase 2: intervene, build the results dict, score the ledger.

Runs in a few seconds on a laptop:  python examples/synth_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifact_signals.data import Benchmark, Item  # noqa: E402
from artifact_signals.evaluate import run_model  # noqa: E402
from artifact_signals.interventions import LEAKAGE, REBALANCE, leakage_resplit, rebalance_and_filter, rerender_score  # noqa: E402
from artifact_signals.judge import exact_match, lenient_choice  # noqa: E402
from artifact_signals.ledger import Ledger, Prediction, format_score, score_ledger  # noqa: E402
from artifact_signals.models import NoisyFormatModel, OracleModel, RetrievalModel  # noqa: E402
from artifact_signals.render import baseline_rendering, partial_input_renderings, standard_renderings  # noqa: E402
from artifact_signals.signals import compute_profile, partial_input_accuracy, retrieval_dominance  # noqa: E402
from artifact_signals.synth import generate  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def leaky_benchmark(n: int = 240) -> tuple[Benchmark, Benchmark]:
    """A template-heavy benchmark where test items are near-copies of train items and TRUE is over-represented."""
    tr, te = [], []
    for k in range(n):
        lab = "TRUE" if k % 4 else "FALSE"
        prem = [f"Worlds: w0, w1. Accessibility: w0 can see w1. At w0: p is {'true' if lab == 'TRUE' else 'false'}. At w1: q is true. Record {k}."]
        tr.append(Item(id=f"tr{k}", premises=prem, question="Is p true at w0?", options=["TRUE", "FALSE"], label=lab))
        te.append(Item(id=f"te{k}", premises=[prem[0].replace("Record", "Entry")], question="Is p true at w0?", options=["TRUE", "FALSE"], label=lab))
    return Benchmark("leaky-train", tr), Benchmark("leaky", te)


def main() -> None:
    # ---------------- Phase 1: diagnosis ----------------
    ctrl = generate(240, seed=0, ood_frac=0.1, name="synthmodal_control")
    ctrl_tr, ctrl_te = ctrl.split(0.4, seed=0, names=("train", "eval"))
    lk_tr, lk_te = leaky_benchmark()

    profiles = {}
    for name, tr, te in (("synthmodal_control", ctrl_tr, ctrl_te), ("leaky", lk_tr, lk_te)):
        gold = {i.id: i.label for i in te}
        model = NoisyFormatModel(gold, accuracy=0.75, noise=0.4, seed=1)   # one subject model, same on both benchmarks
        prof = compute_profile(model, te, exact_match, train=tr, sem_judge=lenient_choice, n_boot=200)
        prof.to_json(OUT / f"profile_{name}.json")
        profiles[name] = prof
        print(prof.summary(), "\n")

    ledger = Ledger("synth dry-run ledger", [
        Prediction("D01", "leaky", "leakage_resplit", "retrieval.accuracy.delta", "down", 0.20, 0.8, "magnitude",
                   "retrieval baseline collapses once near-duplicates are removed"),
        Prediction("D02", "leaky", "leakage_resplit", "retrieval_dominance.sim_correctness_corr", "below", 0.15, 0.7, "manipulation_check"),
        Prediction("D03", "leaky", "rebalance_and_filter", "label_marginal.tv_label_vs_uniform", "below", 0.05, 0.9, "manipulation_check"),
        Prediction("D04", "leaky", "constraint_ablation", "em.constrained_minus_free", "up", 0.10, 0.8, "metric_specific"),
        Prediction("D05", "leaky", "constraint_ablation", "sem.constrained_minus_free", "flat", 0.03, 0.8, "metric_specific"),
        Prediction("D06", "synthmodal_control", "leakage_resplit", "llm.accuracy.delta", "flat", 0.03, 0.85, "specificity"),
        Prediction("D07", "synthmodal_control", "rebalance_and_filter", "llm.accuracy.delta", "flat", 0.03, 0.85, "specificity"),
        Prediction("D08", "synthmodal_control", "surface_rerender", "llm.accuracy.delta", "flat", 0.03, 0.85, "specificity"),
        Prediction("D09", "leaky", "surface_rerender", "llm.accuracy.delta", "flat", 0.03, 0.6, "specificity",
                   "the leaky benchmark is not format-sensitive, only retrieval-driven"),
    ])
    h = ledger.save(OUT / "ledger.json")
    print(f"ledger stamped: sha256 {h}\n")

    # ---------------- Phase 2: interventions ----------------
    results: dict = {}
    for name, tr, te in (("synthmodal_control", ctrl_tr, ctrl_te), ("leaky", lk_tr, lk_te)):
        gold = {i.id: i.label for i in te}
        model = NoisyFormatModel(gold, accuracy=0.75, noise=0.4, seed=1)
        base = baseline_rendering()
        before = run_model(model, te, base, exact_match)
        rd_before = retrieval_dominance(tr, te, before)
        r: dict = {}

        # leakage-aware resplit
        kept, dropped = leakage_resplit(tr, te, max_similarity_=0.85)
        if len(kept):
            gold_k = {i.id: i.label for i in kept}
            after = run_model(NoisyFormatModel(gold_k, 0.75, 0.4, seed=1), kept, base, exact_match)
            rd_after = retrieval_dominance(tr, kept, after)
            r["leakage_resplit"] = {
                "n_dropped": len(dropped), "n_kept": len(kept),
                "llm": {"accuracy": {"before": before.accuracy, "after": after.accuracy, "delta": after.accuracy - before.accuracy}},
                "retrieval": {"accuracy": {"before": rd_before.details["per_method"]["fuzzy"]["accuracy"],
                                           "after": rd_after.details["per_method"]["fuzzy"]["accuracy"],
                                           "delta": rd_after.details["per_method"]["fuzzy"]["accuracy"] - rd_before.details["per_method"]["fuzzy"]["accuracy"]}},
                "retrieval_dominance": {"sim_correctness_corr": rd_after.details["sim_correctness_corr"]},
                "manipulation_check": LEAKAGE.manipulation_check(rd_after),
            }
        else:
            # everything leaked: the resplit empties the benchmark, which is itself the finding
            r["leakage_resplit"] = {"n_dropped": len(dropped), "n_kept": 0, "note": "all test items within similarity cap of train",
                                    "retrieval": {"accuracy": {"before": rd_before.details["per_method"]["fuzzy"]["accuracy"], "after": 0.0,
                                                               "delta": -rd_before.details["per_method"]["fuzzy"]["accuracy"]}},
                                    "retrieval_dominance": {"sim_correctness_corr": 0.0}, "manipulation_check": True}

        # rebalance + partial-input filter
        partial = [run_model(model, te, pr, exact_match) for pr in partial_input_renderings("choice")]
        bal = rebalance_and_filter(te, None, seed=0)
        gold_b = {i.id: i.label for i in bal}
        after_b = run_model(NoisyFormatModel(gold_b, 0.75, 0.4, seed=1), bal, base, exact_match)
        labels = [i.label for i in bal.in_distribution()]
        tv = abs(labels.count("TRUE") / len(labels) - 0.5) if labels else 0.0
        r["rebalance_and_filter"] = {
            "n_after": len(bal),
            "llm": {"accuracy": {"before": before.accuracy, "after": after_b.accuracy, "delta": after_b.accuracy - before.accuracy}},
            "label_marginal": {"tv_label_vs_uniform": tv},
            "manipulation_check": REBALANCE.manipulation_check(partial_input_accuracy(NoisyFormatModel(gold_b, 0.75, 0.4, seed=1), bal)),
        }

        # surface re-rendering
        rr = rerender_score(model, te, exact_match, standard_renderings(0))
        r["surface_rerender"] = {"llm": {"accuracy": {"before": before.accuracy, "after": rr["accuracy_mean_over_renderings"],
                                                      "delta": rr["accuracy_mean_over_renderings"] - before.accuracy}}}

        # constraint ablation (from the profile's signal)
        svg = profiles[name].signals["semantic_validity_gap"].details
        r["constraint_ablation"] = {"em": {"constrained_minus_free": svg["constraint_gain"]},
                                    "sem": {"constrained_minus_free": svg["semantic_gain"]}, "svg": svg["constraint_gain"] - svg["semantic_gain"]}
        results[name] = r

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    score = score_ledger(Ledger.load(OUT / "ledger.json"), results)
    (OUT / "ledger_score.json").write_text(json.dumps(score, indent=2))
    print(format_score(score))


if __name__ == "__main__":
    main()
