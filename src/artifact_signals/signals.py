"""The five diagnostics and the Artifact Risk Profile.

Each diagnostic returns a Signal: a scalar value on a stated scale, a bootstrap CI where
one is meaningful, and a details dict with everything needed to audit it. The profile is
reported as a vector and never collapsed to a scalar; the format-sensitivity gate marks
the other signals as uninterpretable when surface variance exceeds the threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .data import ABSTAIN, Benchmark
from .evaluate import Cache, Run, run_model
from .judge import Judge, exact_match
from .render import Rendering, abstention_rendering, baseline_rendering, partial_input_renderings, standard_renderings
from .retrieval import METHODS, nearest_neighbor
from .stats import (binom_test, cluster_bootstrap, mcnemar, mean, point_biserial, pstdev,
                    risk_coverage_auc, tv_distance)

FS_GATE_DEFAULT = 0.03

# What a diagnostic is a property of. A profile mixes levels on purpose and labels each one,
# because "the benchmark's artifact profile" is too broad a phrase for a vector that also
# contains model behaviour.
LEVELS = ("dataset", "protocol", "model", "model_x_benchmark")
SIGNAL_LEVEL = {
    "format_sensitivity": "model_x_benchmark",   # a model's response to equivalent surface forms of these items
    "partial_input_accuracy": "dataset",         # solvability without the input; estimated with a model, owned by the data
    "label_prior_skew": "model_x_benchmark",     # auxiliary, not in the profile: its dataset half is reported inside partial_input_accuracy
    "retrieval_dominance": "dataset",            # a non-parametric baseline vs the split; the ratio to an LLM is model x benchmark
    "semantic_validity_gap": "protocol",         # a decoding protocol's effect on two metrics; forced_rate needs no model at all
    "abstention_failure": "model_x_benchmark",
}


@dataclass
class Signal:
    name: str
    value: float | None
    ci: tuple[float, float] | None = None
    details: dict = field(default_factory=dict)
    gated: bool = False
    note: str = ""
    level: str = ""

    def __post_init__(self):
        if not self.level:
            self.level = SIGNAL_LEVEL.get(self.name, "model_x_benchmark")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")

    def to_dict(self) -> dict:
        return {"name": self.name, "level": self.level, "value": self.value, "ci": list(self.ci) if self.ci else None,
                "gated": self.gated, "note": self.note, "details": self.details}


def _ids(bench: Benchmark, in_dist: bool = True) -> list[str]:
    return [i.id for i in (bench.in_distribution() if in_dist else bench.items)]


def _acc_ci(run: Run, bench: Benchmark, n_boot: int, seed: int) -> tuple[float, float, float]:
    items = bench.in_distribution()
    vals = [1.0 if run.correct[i.id] else 0.0 for i in items]
    return cluster_bootstrap(vals, [i.cluster for i in items], n_boot=n_boot, seed=seed)


# --- 1. format sensitivity ------------------------------------------------------------


def format_sensitivity_from_runs(runs: list[Run], bench: Benchmark, *, n_boot: int = 500, seed: int = 0) -> Signal:
    """SD of in-distribution accuracy across runs that differ only in surface form."""
    ids = _ids(bench)
    accs = [r.accuracy_on(ids) for r in runs]
    sd = pstdev(accs)
    items = bench.in_distribution()
    clusters = [i.cluster for i in items]
    mats = [[1.0 if r.correct[i.id] else 0.0 for i in items] for r in runs]
    import random as _r
    rng = _r.Random(seed)
    groups: dict[str, list[int]] = {}
    for k, c in enumerate(clusters):
        groups.setdefault(c, []).append(k)
    keys = list(groups)
    draws = []
    for _ in range(n_boot):
        idx = [j for _ in keys for j in groups[rng.choice(keys)]]
        draws.append(pstdev([mean([m[j] for j in idx]) for m in mats]))
    draws.sort()
    ci = (draws[int(0.025 * (n_boot - 1))], draws[int(0.975 * (n_boot - 1))]) if draws else None
    base = runs[0]
    agree = mean([1.0 if len({r.correct[i] for r in runs}) == 1 else 0.0 for i in ids])
    details = {
        "per_rendering": {r.rendering: {"accuracy": a, "invalid_rate": r.invalid_rate()} for r, a in zip(runs, accs)},
        "range": max(accs) - min(accs),
        "item_agreement": agree,
        "mcnemar_vs_baseline": {r.rendering: mcnemar([base.correct[i] for i in ids], [r.correct[i] for i in ids])
                                for r in runs[1:]},
    }
    return Signal("format_sensitivity", sd, ci, details)


def format_sensitivity(model, bench: Benchmark, judge: Judge = exact_match, renderings: list[Rendering] | None = None,
                       *, n_boot: int = 500, seed: int = 0, cache: Cache | None = None) -> tuple[Signal, list[Run]]:
    renderings = renderings or standard_renderings(seed)
    runs = [run_model(model, bench, r, judge, cache=cache) for r in renderings]
    return format_sensitivity_from_runs(runs, bench, n_boot=n_boot, seed=seed), runs


# --- 2. partial-input accuracy ----------------------------------------------------------


def partial_input_from_runs(runs: list[Run], bench: Benchmark) -> Signal:
    ids = _ids(bench)
    items = bench.in_distribution()
    chance = mean([1.0 / len(i.options) if i.options else 0.0 for i in items])
    per, best = {}, None
    for run in runs:
        acc = run.accuracy_on(ids)
        k = sum(1 for i in ids if run.correct[i])
        per[run.rendering] = {"accuracy": acc, "excess_over_chance": acc - chance,
                              "p_vs_chance": binom_test(k, len(ids), chance, "greater") if chance > 0 else None}
        if best is None or acc - chance > best[1]:
            best = (run.rendering, acc - chance)
    note = "" if bench.task == "choice" else "free-form task: chance is 0, interpret excess as raw accuracy"
    return Signal("partial_input_accuracy", best[1] if best else None,
                  details={"chance": chance, "per_mode": per, "worst_mode": best[0] if best else None,
                           "label_marginal": label_marginal(bench)}, note=note)


def label_marginal(bench: Benchmark) -> dict:
    """Dataset-level label prior: the marginal and its TV distance from uniform. Reported inside
    partial_input_accuracy, since a prior only matters if a model can exploit it without the input."""
    items = bench.in_distribution()
    space = bench.label_space()
    n = max(1, len(items))
    m = {c: sum(1 for i in items if i.label == c) / n for c in space}
    return {"marginal": m, "tv_vs_uniform": tv_distance(m, {c: 1.0 / len(space) for c in space}) if space else 0.0}


def partial_input_accuracy(model, bench: Benchmark, judge: Judge = exact_match, renderings: list[Rendering] | None = None,
                           *, cache: Cache | None = None) -> Signal:
    renderings = renderings or partial_input_renderings(bench.task)
    runs = [run_model(model, bench, r, judge, cache=cache) for r in renderings]
    return partial_input_from_runs(runs, bench)


# --- auxiliary: label-prior skew (library only; not one of the five profile signals) -----------


def label_prior_skew(bench: Benchmark, run: Run) -> Signal:
    items = bench.in_distribution()
    space = bench.label_space()
    n = len(items)
    label_m = {c: sum(1 for i in items if i.label == c) / n for c in space}
    valid = [run.answers[i.id] for i in items if run.answers[i.id]]
    model_m = {c: (sum(1 for a in valid if a == c) / len(valid) if valid else 0.0) for c in space}
    uniform = {c: 1.0 / len(space) for c in space}
    return Signal("label_prior_skew", tv_distance(model_m, label_m), details={
        "label_marginal": label_m, "model_marginal": model_m,
        "tv_label_vs_uniform": tv_distance(label_m, uniform),
        "tv_model_vs_uniform": tv_distance(model_m, uniform),
        "invalid_rate": 1 - len(valid) / n if n else None,
    })


# --- 4. retrieval dominance -----------------------------------------------------------------


def retrieval_dominance(train: Benchmark, test: Benchmark, llm_run: Run, methods: tuple[str, ...] = METHODS) -> Signal:
    items = test.in_distribution()
    ids = [i.id for i in items]
    tr_texts, tr_labels = [i.text() for i in train], [i.label for i in train]
    te_texts = [i.text() for i in items]
    llm_correct = [llm_run.correct[i] for i in ids]
    llm_acc = mean([1.0 if c else 0.0 for c in llm_correct])
    per, best = {}, None
    for m in methods:
        preds, sims = nearest_neighbor(tr_texts, tr_labels, te_texts, m)
        correct = [p == i.label for p, i in zip(preds, items)]
        acc = mean([1.0 if c else 0.0 for c in correct])
        per[m] = {"accuracy": acc, "ratio": acc / max(llm_acc, 1e-9), "gap": llm_acc - acc,
                  "sim_correctness_corr": point_biserial(sims, llm_correct),
                  "mean_max_similarity": mean(sims),
                  "agreement_with_llm": mean([1.0 if (c == l) else 0.0 for c, l in zip(correct, llm_correct)])}
        if best is None or acc > per[best]["accuracy"]:
            best = m
    return Signal("retrieval_dominance", per[best]["ratio"] if best else None, details={
        "llm_accuracy": llm_acc, "best_method": best, "per_method": per,
        "sim_correctness_corr": per[best]["sim_correctness_corr"] if best else None,
        "gap": per[best]["gap"] if best else None,
    })


# --- 5. constraint gain and semantic-validity gap ----------------------------------------------


def constraint_gain_from_runs(free: Run, con: Run, bench: Benchmark, *, sem_judge: Judge,
                              forced_ids: set[str] | None = None) -> Signal:
    """EM gain minus semantic gain between an unconstrained and a constrained run judged strictly.
    `forced_ids` marks items on which the constraint admits only one answer (constraint leakage);
    the gain is then reported on forced and free subsets separately."""
    ids = _ids(bench)
    sem_free, sem_con = free.rejudge(bench, sem_judge), con.rejudge(bench, sem_judge)
    em_f, em_c = free.accuracy_on(ids), con.accuracy_on(ids)
    s_f, s_c = sem_free.accuracy_on(ids), sem_con.accuracy_on(ids)
    cg, sg = em_c - em_f, s_c - s_f
    dissoc = mean([1.0 if (free.correct[i] != sem_free.correct[i] or con.correct[i] != sem_con.correct[i]) else 0.0 for i in ids])
    details = {
        "em_free": em_f, "em_constrained": em_c, "sem_free": s_f, "sem_constrained": s_c,
        "constraint_gain": cg, "semantic_gain": sg,
        "mcnemar_em": mcnemar([free.correct[i] for i in ids], [con.correct[i] for i in ids]),
        "mcnemar_sem": mcnemar([sem_free.correct[i] for i in ids], [sem_con.correct[i] for i in ids]),
        "dissociation_rate": dissoc,
        "invalid_free": free.invalid_rate(), "invalid_constrained": con.invalid_rate(),
    }
    if forced_ids is not None:
        forced = [i for i in ids if i in forced_ids]
        free_ids = [i for i in ids if i not in forced_ids]
        details["forced_rate"] = len(forced) / len(ids) if ids else 0.0
        for name, sub in (("forced", forced), ("free", free_ids)):
            if sub:
                details[f"{name}_subset"] = {
                    "n": len(sub), "em_free": free.accuracy_on(sub), "em_constrained": con.accuracy_on(sub),
                    "sem_free": sem_free.accuracy_on(sub), "sem_constrained": sem_con.accuracy_on(sub),
                    "mcnemar_sem": mcnemar([sem_free.correct[i] for i in sub], [sem_con.correct[i] for i in sub]),
                }
    return Signal("semantic_validity_gap", cg - sg, details=details)


def constraint_gain(model, bench: Benchmark, rendering: Rendering | None = None, *, em_judge: Judge = exact_match,
                    sem_judge: Judge, cache: Cache | None = None) -> tuple[Signal, dict[str, Run]]:
    rendering = rendering or baseline_rendering()
    free = run_model(model, bench, rendering, em_judge, constrained=False, cache=cache)
    con = run_model(model, bench, rendering, em_judge, constrained=True, cache=cache)
    sig = constraint_gain_from_runs(free, con, bench, sem_judge=sem_judge)
    return sig, {"em_free": free, "em_constrained": con, "sem_free": free.rejudge(bench, sem_judge),
                 "sem_constrained": con.rejudge(bench, sem_judge)}


# --- 6. abstention failure ---------------------------------------------------------------------


def abstention_failure_from_run(run: Run, bench: Benchmark) -> Signal:
    ood = bench.ood_items()
    if not ood:
        return Signal("abstention_failure", None, note="benchmark has no OOD probes; use interventions.inject_ood first")
    ind = bench.in_distribution()
    answered = mean([0.0 if run.answers[i.id] == ABSTAIN else 1.0 for i in ood])
    false_abstain = mean([1.0 if run.answers[i.id] == ABSTAIN else 0.0 for i in ind]) if ind else None
    confs = [run.confidence[i.id] for i in ind]
    rc = None
    if ind and all(c is not None for c in confs):
        rc = risk_coverage_auc([float(c) for c in confs], [run.correct[i.id] for i in ind])
    k = sum(1 for i in ood if run.answers[i.id] != ABSTAIN)
    return Signal("abstention_failure", answered, details={
        "n_ood": len(ood), "ood_answered_rate": answered, "false_abstention_rate": false_abstain,
        "ood_invalid_rate": mean([1.0 if not run.answers[i.id] else 0.0 for i in ood]),
        "in_dist_accuracy": run.accuracy_on([i.id for i in ind]) if ind else None,
        "risk_coverage_auc": rc, "p_vs_half": binom_test(k, len(ood), 0.5, "greater"),
    })


def abstention_failure(model, bench: Benchmark, judge: Judge = exact_match, rendering: Rendering | None = None,
                       *, cache: Cache | None = None) -> Signal:
    if not bench.ood_items():
        return Signal("abstention_failure", None, note="benchmark has no OOD probes; use interventions.inject_ood first")
    rendering = rendering or abstention_rendering()
    return abstention_failure_from_run(run_model(model, bench, rendering, judge, cache=cache), bench)


# --- profile ---------------------------------------------------------------------------------


@dataclass
class ArtifactRiskProfile:
    benchmark: str
    model: str
    signals: dict[str, Signal]
    fs_gate: float = FS_GATE_DEFAULT

    def apply_gate(self) -> "ArtifactRiskProfile":
        fs = self.signals.get("format_sensitivity")
        if fs and fs.value is not None and fs.value > self.fs_gate:
            for k, s in self.signals.items():
                if k != "format_sensitivity":
                    s.gated = True
                    s.note = (s.note + "; " if s.note else "") + f"gated: format sensitivity {fs.value:.3f} > {self.fs_gate}"
        return self

    @property
    def gated(self) -> bool:
        fs = self.signals.get("format_sensitivity")
        return bool(fs and fs.value is not None and fs.value > self.fs_gate)

    def vector(self) -> dict[str, float | None]:
        return {k: s.value for k, s in self.signals.items()}

    def to_dict(self) -> dict:
        return {"benchmark": self.benchmark, "model": self.model, "fs_gate": self.fs_gate, "gated": self.gated,
                "vector": self.vector(), "signals": {k: s.to_dict() for k, s in self.signals.items()}}

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        lines = [f"Artifact Risk Profile: {self.benchmark} / {self.model}  (gate={self.fs_gate}, gated={self.gated})",
                 "  levels: dataset | protocol | model | model_x_benchmark"]
        for k, s in self.signals.items():
            v = "n/a" if s.value is None else f"{s.value:.3f}"
            ci = f"  [{s.ci[0]:.3f}, {s.ci[1]:.3f}]" if s.ci else ""
            flag = "  (GATED)" if s.gated else ""
            extra = ""
            if k == "retrieval_dominance" and s.details.get("sim_correctness_corr") is not None:
                extra = f"  corr={s.details['sim_correctness_corr']:+.3f} gap={s.details['gap']:+.3f}"
            if k == "semantic_validity_gap" and "constraint_gain" in s.details:
                extra = f"  cg={s.details['constraint_gain']:+.3f} sg={s.details['semantic_gain']:+.3f}"
            lines.append(f"  {k:<26}{v:>8}{ci}  [{s.level}]{extra}{flag}")
        return "\n".join(lines)


def compute_profile(
    model,
    bench: Benchmark,
    judge: Judge = exact_match,
    *,
    train: Benchmark | None = None,
    sem_judge: Judge | None = None,
    renderings: list[Rendering] | None = None,
    fs_gate: float = FS_GATE_DEFAULT,
    n_boot: int = 500,
    seed: int = 0,
    cache: Cache | None = None,
) -> ArtifactRiskProfile:
    signals: dict[str, Signal] = {}
    fs, runs = format_sensitivity(model, bench, judge, renderings, n_boot=n_boot, seed=seed, cache=cache)
    signals["format_sensitivity"] = fs
    base = runs[0]
    est, lo, hi = _acc_ci(base, bench, n_boot, seed)
    fs.details["baseline_accuracy"] = {"value": est, "ci": [lo, hi]}
    signals["partial_input_accuracy"] = partial_input_accuracy(model, bench, judge, cache=cache)
    if train is not None:
        signals["retrieval_dominance"] = retrieval_dominance(train, bench, base)
    else:
        signals["retrieval_dominance"] = Signal("retrieval_dominance", None, note="no training split supplied")
    if sem_judge is not None:
        signals["semantic_validity_gap"], _ = constraint_gain(model, bench, renderings[0] if renderings else None,
                                                             em_judge=judge, sem_judge=sem_judge, cache=cache)
    else:
        signals["semantic_validity_gap"] = Signal("semantic_validity_gap", None, note="no semantic judge supplied")
    signals["abstention_failure"] = abstention_failure(model, bench, judge, cache=cache)
    return ArtifactRiskProfile(bench.name, model.name, signals, fs_gate).apply_gate()
