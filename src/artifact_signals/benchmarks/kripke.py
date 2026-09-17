"""Paper 2: Kripke modal accessibility (th00masml/kripke-modal-accessibility).

160 fixtures: 80 in-language (gold judgement "w_i |= F iff TRUE/FALSE.") and 80 out-of-language
(correct output is empty). Two Qwen2.5 models x five arms, two runs: `gold` (constraint set built
from gold judgements, forcing the answer on 25/80) and `product` (leak-free world x formula x truth).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from ..data import ABSTAIN, Benchmark, Item
from ..evaluate import Run
from ..retrieval import char3_sim
from ..signals import (ArtifactRiskProfile, Signal, abstention_failure_from_run, constraint_gain_from_runs,
                       format_sensitivity_from_runs, label_prior_skew)
from ..stats import mean, point_biserial

ROOT = Path(os.environ.get("ARTIFACT_SIGNALS_DATA", Path(__file__).resolve().parents[3] / "data" / "external")) / "kripke"

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
SHORT = {MODELS[0]: "0.5B", MODELS[1]: "1.5B"}
ARMS = ["naive", "prompted", "prompt_tuned", "constrained", "constrained_tuned"]
UNCONSTRAINED = ["naive", "prompted", "prompt_tuned"]

CANONICAL_RE = re.compile(r"(w\d+)\s*\|=\s*([A-Za-z0-9_(),]+)\s*iff\s*(TRUE|FALSE)\.?", re.IGNORECASE)
LENIENT_RE = re.compile(r"(w)_?(\d+)\s*\|=?\s*([A-Za-z0-9_(), ]+?)\s*iff\s*(TRUE|FALSE)\.?", re.IGNORECASE)


def strict_parse(out: str) -> str | None:
    m = CANONICAL_RE.search(out or "")
    if not m:
        return None
    w, f, t = m.groups()
    return f"{w.lower()} |= {f} iff {t.upper()}."


def lenient_parse(out: str) -> str | None:
    m = LENIENT_RE.search(out or "")
    if not m:
        return None
    _, d, f, t = m.groups()
    f = f.replace(" ", "").replace("DIAMONDS", "DIA").replace("DIAMOND", "DIA")
    return f"w{d} |= {f} iff {t.upper()}."


def truth_of(judgement: str | None) -> str | None:
    if not judgement or " iff " not in judgement:
        return None
    return judgement.split(" iff ")[-1].rstrip(".").upper()


def load(root: Path = ROOT) -> Benchmark:
    items = []
    for line in (root / "fixtures.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        f = json.loads(line)
        items.append(Item(id=f["id"], premises=[l for l in f["source"].split("\n") if l.strip()], question=f["query"],
                          options=None, label=f["gold"] if f["present"] else ABSTAIN, ood=not f["present"],
                          source_id=f["id"], meta={"world": f["world"], "formula": f["formula"], "present": f["present"]}))
    return Benchmark("kripke", items)


def allowed_set(bench: Benchmark, mode: str) -> set[str]:
    present = [i for i in bench if not i.ood]
    if mode == "gold":
        return {i.label for i in present}
    worlds = sorted({i.meta["world"] for i in present})
    formulas = sorted({i.meta["formula"] for i in present})
    return {f"{w} |= {f} iff {t}." for w in worlds for f in formulas for t in ("TRUE", "FALSE")}


def _flip(s: str) -> str:
    return s.replace("iff TRUE.", "iff XX.").replace("iff FALSE.", "iff TRUE.").replace("iff XX.", "iff FALSE.")


def forced_ids(bench: Benchmark, allowed: set[str]) -> set[str]:
    """Fixtures on which the admissible set contains only one truth value: the constraint answers them."""
    return {i.id for i in bench if not i.ood and _flip(i.label) not in allowed}


# --- judges ----------------------------------------------------------------------------------


def strict_judge(item: Item, answer: str, raw: str = "") -> bool:
    if item.ood:
        return answer == ABSTAIN
    return strict_parse(raw) == item.label


def lenient_judge(item: Item, answer: str, raw: str = "") -> bool:
    if item.ood:
        return answer == ABSTAIN
    return lenient_parse(raw) == item.label


def truth_judge(item: Item, answer: str, raw: str = "") -> bool:
    """Semantic judge: the truth value the model committed to, read through format noise."""
    if item.ood:
        return answer == ABSTAIN
    t = truth_of(lenient_parse(raw))
    return t is not None and t == truth_of(item.label)


def decode(out: str) -> str:
    out = (out or "").strip()
    if not out:
        return ABSTAIN
    return strict_parse(out) or out


def cached_runs(bench: Benchmark, root: Path = ROOT, run: str = "gold") -> dict[tuple[str, str], Run]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for line in (root / f"{run}_raw.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows.setdefault((r["model"], r["arm"]), {})[r["id"]] = r["output"]
    out = {}
    for (model, arm), outputs in rows.items():
        out[(model, arm)] = Run.from_answers(f"{SHORT.get(model, model)}", bench, {i: decode(o) for i, o in outputs.items()},
                                             strict_judge, raw=outputs, rendering=arm, constrained="constrained" in arm)
    return out


def run_meta(run: str = "gold", root: Path = ROOT) -> dict:
    return json.loads((root / f"{run}_run_meta.json").read_text())


# --- truth-value projection: the 80 in-language fixtures as a binary choice task ---------------------


def truth_projection(bench: Benchmark, run: Run) -> tuple[Benchmark, Run]:
    items = [Item(id=i.id, premises=i.premises, question=f"Is {i.meta['formula']} true at {i.meta['world']}?",
                  options=["TRUE", "FALSE"], label=truth_of(i.label) or "", source_id=i.id, meta=i.meta)
             for i in bench if not i.ood]
    tb = Benchmark("kripke-truth", items)
    answers = {i.id: (truth_of(lenient_parse(run.raw[i.id])) or "") for i in items}
    return tb, Run.from_answers(run.model, tb, answers, rendering=run.rendering, constrained=run.constrained)


def loo_retrieval_dominance(tb: Benchmark, tr: Run, method: str = "char3") -> Signal:
    """Leave-one-out nearest neighbour over the fixtures' own text -> its truth value."""
    texts = {i.id: i.text() for i in tb}
    preds, sims = {}, {}
    for i in tb:
        best = max((j for j in tb if j.id != i.id), key=lambda j: char3_sim(texts[i.id], texts[j.id]))
        preds[i.id], sims[i.id] = best.label, char3_sim(texts[i.id], texts[best.id])
    ids = [i.id for i in tb]
    by = tb.by_id()
    nn_correct = [preds[i] == by[i].label for i in ids]
    acc = mean([1.0 if c else 0.0 for c in nn_correct])
    llm_acc = tr.accuracy_on(ids)
    return Signal("retrieval_dominance", acc / max(llm_acc, 1e-9), details={
        "llm_accuracy": llm_acc, "best_method": f"loo-{method}",
        "per_method": {f"loo-{method}": {"accuracy": acc, "ratio": acc / max(llm_acc, 1e-9), "gap": llm_acc - acc,
                                         "sim_correctness_corr": point_biserial([sims[i] for i in ids], [tr.correct[i] for i in ids]),
                                         "mean_max_similarity": mean(list(sims.values())),
                                         "agreement_with_llm": mean([1.0 if c == tr.correct[i] else 0.0 for c, i in zip(nn_correct, ids)])}},
        "sim_correctness_corr": point_biserial([sims[i] for i in ids], [tr.correct[i] for i in ids]),
        "gap": llm_acc - acc, "note": "leave-one-out over the 80 in-language fixtures; no separate training split exists"})


# --- profile and interventions -------------------------------------------------------------------


def profile(model: str = MODELS[1], run: str = "gold", root: Path = ROOT) -> ArtifactRiskProfile:
    bench = load(root)
    runs = cached_runs(bench, root, run)
    allowed = allowed_set(bench, run_meta(run, root).get("allowed_set", "gold"))
    forced = forced_ids(bench, allowed)
    unc = [runs[(model, a)] for a in UNCONSTRAINED]
    prompted, con = runs[(model, "prompted")], runs[(model, "constrained")]
    signals: dict[str, Signal] = {}
    fs = format_sensitivity_from_runs(unc, bench, n_boot=300)
    fs_len = format_sensitivity_from_runs([r.rejudge(bench, lenient_judge) for r in unc], bench, n_boot=300)
    fs_truth = format_sensitivity_from_runs([r.rejudge(bench, truth_judge) for r in unc], bench, n_boot=300)
    fs.details["lenient"] = {"value": fs_len.value, "per_rendering": fs_len.details["per_rendering"]}
    fs.details["truth_value"] = {"value": fs_truth.value, "per_rendering": fs_truth.details["per_rendering"]}
    fs.note = "SD over the three unconstrained prompt arms (naive / prompted / prompt_tuned), strict parse"
    signals["format_sensitivity"] = fs
    tb, tr = truth_projection(bench, prompted)
    gold_t = Counter(i.label for i in tb)
    maj = max(gold_t.values()) / len(tb)
    model_t = Counter(a for a in tr.answers.values())
    signals["partial_input_accuracy"] = Signal(
        "partial_input_accuracy", maj - 0.5,
        details={"proxy": "majority truth value over the 80 in-language fixtures", "majority_accuracy": maj, "chance": 0.5,
                 "gold_truth_dist": dict(gold_t), "model_truth_dist_prompted": dict(model_t)},
        note="proxy from the label prior, not a model run")
    signals["label_prior_skew"] = label_prior_skew(tb, tr)
    signals["retrieval_dominance"] = loo_retrieval_dominance(tb, tr)
    svg = constraint_gain_from_runs(prompted, con, bench, sem_judge=truth_judge, forced_ids=forced)
    svg_len = constraint_gain_from_runs(prompted, con, bench, sem_judge=lenient_judge, forced_ids=forced)
    svg.details["lenient_exact_as_semantic"] = {k: svg_len.details[k] for k in ("sem_free", "sem_constrained", "semantic_gain")}
    svg.details["allowed_set_mode"] = run_meta(run, root).get("allowed_set", "gold")
    svg.note = "EM = strict canonical parse; semantic = truth value under lenient parse; prompted vs constrained arm"
    signals["semantic_validity_gap"] = svg
    af = abstention_failure_from_run(prompted, bench)
    af.details["per_arm"] = {a: abstention_failure_from_run(runs[(model, a)], bench).value for a in ARMS}
    signals["abstention_failure"] = af
    return ArtifactRiskProfile(f"kripke@{run}", SHORT.get(model, model), signals).apply_gate()


def interventions(model: str = MODELS[1], root: Path = ROOT) -> dict:
    bench = load(root)
    ids = [i.id for i in bench if not i.ood]
    out: dict = {"model": SHORT.get(model, model)}
    per_run = {}
    for run in ("gold", "product"):
        runs = cached_runs(bench, root, run)
        allowed = allowed_set(bench, run_meta(run, root).get("allowed_set", run))
        forced = forced_ids(bench, allowed)
        p, c = runs[(model, "prompted")], runs[(model, "constrained")]
        sig = constraint_gain_from_runs(p, c, bench, sem_judge=truth_judge, forced_ids=forced)
        d = sig.details
        per_run[run] = {"em": {"free": d["em_free"], "constrained": d["em_constrained"], "constrained_minus_free": d["constraint_gain"]},
                        "sem": {"free": d["sem_free"], "constrained": d["sem_constrained"], "constrained_minus_free": d["semantic_gain"]},
                        "svg": sig.value, "forced_rate": d["forced_rate"], "n_forced": len(forced),
                        "forced_subset": d.get("forced_subset"), "free_subset": d.get("free_subset"),
                        "mcnemar_em": d["mcnemar_em"], "mcnemar_sem": d["mcnemar_sem"]}
    out["constraint_ablation"] = {**per_run["product"], "per_run": per_run,
                                  "note": "registered quantity = leak-free (product) run; gold run shown for the pre-intervention state"}
    g, pr = cached_runs(bench, root, "gold"), cached_runs(bench, root, "product")
    gc, pc = g[(model, "constrained")], pr[(model, "constrained")]
    forced_gold = forced_ids(bench, allowed_set(bench, "gold"))
    free_gold = [i for i in ids if i not in forced_gold]
    out["leak_removal"] = {
        "em": {"before": gc.accuracy_on(ids), "after": pc.accuracy_on(ids), "delta": pc.accuracy_on(ids) - gc.accuracy_on(ids)},
        "forced_subset": {"n": len(forced_gold), "em_before": gc.accuracy_on(sorted(forced_gold)), "em_after": pc.accuracy_on(sorted(forced_gold))},
        "free_subset": {"n": len(free_gold), "em_before": gc.accuracy_on(free_gold), "em_after": pc.accuracy_on(free_gold),
                        "outputs_identical": sum(1 for i in free_gold if gc.raw[i] == pc.raw[i])},
        "manipulation_check": len(forced_ids(bench, allowed_set(bench, "product"))) == 0,
    }
    unc = [g[(model, a)] for a in UNCONSTRAINED]
    base = g[(model, "prompted")]
    out["surface_rerender"] = {
        "strict": {"prompted": base.accuracy_on(ids), "mean_over_arms": mean([r.accuracy_on(ids) for r in unc]),
                   "per_arm": {r.rendering: r.accuracy_on(ids) for r in unc}},
        "truth_value": {"prompted": base.rejudge(bench, truth_judge).accuracy_on(ids),
                        "mean_over_arms": mean([r.rejudge(bench, truth_judge).accuracy_on(ids) for r in unc]),
                        "per_arm": {r.rendering: r.rejudge(bench, truth_judge).accuracy_on(ids) for r in unc}},
        "llm": {"accuracy": {"delta": mean([r.accuracy_on(ids) for r in unc]) - base.accuracy_on(ids)}},
    }
    out["abstention_enabled"] = {"ood_answered_rate": {a: abstention_failure_from_run(g[(model, a)], bench).value for a in ARMS},
                                 "note": "the prompt_tuned and constrained_tuned arms already instruct the model to output empty on out-of-language atoms"}
    return out


def published(run: str = "gold", root: Path = ROOT) -> dict:
    return json.loads((root / f"{run}_paper_stats.json").read_text())
