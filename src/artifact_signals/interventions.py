"""De-artifacting interventions, each paired with the signal it targets, a manipulation
check (did the signal move?) and, where the intervention filters items, an invariance
check (did the construct survive?).

Minimal-edit principle: an intervention may remove an artifact, it may not change the task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .data import ABSTAIN, Benchmark, Item
from .evaluate import Cache, Run, run_model
from .judge import Judge, exact_match
from .render import Rendering, standard_renderings
from .retrieval import max_similarity, tokens
from .signals import Signal
from .stats import mean, spearman


@dataclass
class InterventionSpec:
    name: str
    targets: tuple[str, ...]
    manipulation_check: Callable[[Signal], bool]
    description: str


# --- 1. leakage-aware resplit ---------------------------------------------------------------


def leakage_resplit(train: Benchmark, test: Benchmark, *, max_similarity_: float = 0.8, method: str = "fuzzy",
                    name: str | None = None) -> tuple[Benchmark, list[str]]:
    """Drop test items whose nearest training item is more similar than the cap."""
    sims = max_similarity([i.text() for i in train], [i.text() for i in test], method)
    keep, dropped = [], []
    for it, s in zip(test, sims):
        (keep if s <= max_similarity_ else dropped).append(it)
    return Benchmark(name or f"{test.name}-resplit", keep), [i.id for i in dropped]


def novel_terms_only(train: Benchmark, test: Benchmark, *, min_novel: int = 1, name: str | None = None) -> Benchmark:
    """Keep test items containing at least `min_novel` content tokens absent from the training vocabulary."""
    vocab = {t for i in train for t in tokens(i.text())}
    keep = [i for i in test if sum(1 for t in set(tokens(i.text())) if t not in vocab) >= min_novel]
    return Benchmark(name or f"{test.name}-novel", keep)


LEAKAGE = InterventionSpec(
    "leakage_resplit", ("retrieval_dominance",),
    lambda after: (after.details.get("sim_correctness_corr") or 0.0) < 0.15,
    "cap train/test lexical similarity; the similarity-correctness correlation must fall below 0.15",
)


# --- 2. label rebalancing + partial-input filtering -------------------------------------------


def rebalance_and_filter(bench: Benchmark, partial_runs: list[Run] | None = None, *, seed: int = 0,
                         name: str | None = None) -> Benchmark:
    """Remove items any partial-input run solved, then downsample to equal label counts."""
    rng = random.Random(seed)
    solved: set[str] = set()
    for r in partial_runs or []:
        solved |= {i for i, c in r.correct.items() if c}
    pool = [i for i in bench.in_distribution() if i.id not in solved]
    by_label: dict[str, list[Item]] = {}
    for i in pool:
        by_label.setdefault(i.label, []).append(i)
    if len(by_label) < 2:  # a single surviving class cannot be balanced; the filter ate the benchmark
        return Benchmark(name or f"{bench.name}-balanced", [])
    k = min(len(v) for v in by_label.values())
    keep: list[Item] = []
    for lab, v in by_label.items():
        rng.shuffle(v)
        keep.extend(v[:k])
    keep.extend(bench.ood_items())
    rng.shuffle(keep)
    return Benchmark(name or f"{bench.name}-balanced", keep)


REBALANCE = InterventionSpec(
    "rebalance_and_filter", ("partial_input_accuracy",),
    lambda after: (after.value or 0.0) < 0.05,
    "balance labels and drop partial-input-solvable items; partial-input excess over chance must fall below 0.05",
)


# --- 3. surface re-rendering (a scoring change, not a data change) ----------------------------------


def rerender_score(model, bench: Benchmark, judge: Judge = exact_match, renderings: list[Rendering] | None = None,
                   *, seed: int = 0, cache: Cache | None = None) -> dict:
    renderings = renderings or standard_renderings(seed)
    runs = [run_model(model, bench, r, judge, cache=cache) for r in renderings]
    ids = [i.id for i in bench.in_distribution()]
    per_item = {i: mean([1.0 if r.correct[i] else 0.0 for r in runs]) for i in ids}
    return {"accuracy_mean_over_renderings": mean(list(per_item.values())),
            "per_rendering": {r.rendering: r.accuracy_on(ids) for r in runs}, "per_item": per_item, "runs": runs}


RERENDER = InterventionSpec(
    "surface_rerender", ("format_sensitivity",),
    lambda after: (after.value or 0.0) <= 0.03,
    "score as the mean over equivalent renderings; between-rendering SD must fall to the gate or below",
)


# --- 4. constraint ablation (computed by signals.constraint_gain) -------------------------------------

CONSTRAINT = InterventionSpec(
    "constraint_ablation", ("semantic_validity_gap",),
    lambda after: (after.details.get("dissociation_rate") or 0.0) > 0.0,
    "free vs constrained decoding with the semantic judge fixed; EM and semantic judgments must be separable",
)


# --- 5. abstention-enabled scoring ----------------------------------------------------------------------


def inject_ood(bench: Benchmark, ood_source: Benchmark | list[Item], *, fraction: float = 0.1, seed: int = 0,
               name: str | None = None) -> Benchmark:
    """Mix OOD probes (labelled ABSTAIN) into the benchmark at the requested fraction of the final set."""
    rng = random.Random(seed)
    pool = [i for i in (ood_source.items if isinstance(ood_source, Benchmark) else ood_source)]
    n_in = len(bench.in_distribution())
    k = min(len(pool), int(round(fraction * n_in / max(1e-9, 1 - fraction))))
    picked = rng.sample(pool, k) if k else []
    probes = [Item(id=f"ood-{i.id}", question=i.question, label=ABSTAIN, premises=list(i.premises),
                   options=list(i.options) if i.options else None, ood=True, source_id=i.cluster,
                   meta={**i.meta, "ood": True}) for i in picked]
    items = list(bench.in_distribution()) + probes
    rng.shuffle(items)
    return Benchmark(name or f"{bench.name}-abstain", items)


def make_unanswerable(bench: Benchmark, *, fraction: float = 0.1, seed: int = 0, name: str | None = None) -> Benchmark:
    """OOD probes built from the benchmark itself by deleting all premises (answer becomes undetermined).
    Use when no external OOD source exists; weaker than principled probes."""
    rng = random.Random(seed)
    ind = bench.in_distribution()
    k = int(round(fraction * len(ind) / max(1e-9, 1 - fraction)))
    picked = rng.sample(ind, min(k, len(ind)))
    probes = [Item(id=f"unans-{i.id}", question=i.question, label=ABSTAIN, premises=[], options=list(i.options) if i.options else None,
                   ood=True, source_id=i.cluster, meta={**i.meta, "ood": "premises_removed"}) for i in picked]
    items = list(ind) + probes
    rng.shuffle(items)
    return Benchmark(name or f"{bench.name}-unans", items)


ABSTENTION = InterventionSpec(
    "abstention_enabled", ("abstention_failure",),
    lambda after: (after.value if after.value is not None else 1.0) < 0.5,
    "add an explicit abstain option and OOD probes; abstention on probes must exceed 50%",
)


# --- invariance check ----------------------------------------------------------------------------


def cross_model_difficulty_agreement(runs_by_model: dict[str, list[Run]], ids: list[str]) -> float:
    """Mean pairwise Spearman between models' per-item difficulty (mean correctness over renderings)."""
    names = list(runs_by_model)
    diff = {m: [mean([1.0 if r.correct[i] else 0.0 for r in runs_by_model[m]]) for i in ids] for m in names}
    rhos = [spearman(diff[a], diff[b]) for k, a in enumerate(names) for b in names[k + 1:]]
    return mean(rhos) if rhos else float("nan")


def invariance_check(runs_by_model: dict[str, list[Run]], before_ids: list[str], after_ids: list[str],
                     *, min_rho: float = 0.6) -> dict:
    """The residual set must still be ordered by something models agree on. If cross-model agreement
    on survivors collapses, the intervention removed signal, not artifact, and its predictions are voided."""
    b = cross_model_difficulty_agreement(runs_by_model, before_ids)
    a = cross_model_difficulty_agreement(runs_by_model, after_ids)
    return {"agreement_before": b, "agreement_after": a, "passed": (a != a) or a >= min_rho, "min_rho": min_rho,
            "n_before": len(before_ids), "n_after": len(after_ids)}


SPECS = {s.name: s for s in (LEAKAGE, REBALANCE, RERENDER, CONSTRAINT, ABSTENTION)}
