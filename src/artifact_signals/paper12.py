"""Phase 1 of paper 3 on papers 1 and 2, offline: profiles, interventions, ledger score, report.

    artifact-signals paper12 --out outputs/paper12

Everything here recomputes from the cached outputs vendored under data/external/. Nothing calls
a model. The ledger scored is ledger/paper3_ledger.json; quantities it cannot find are reported
as missing, never silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .benchmarks import cod, kripke
from .ledger import Ledger, format_score, score_ledger

LEDGER = Path(__file__).resolve().parents[2] / "ledger" / "retrospective_papers12.json"
FLOAT_DIGITS = 12


def _rounded(x):
    if isinstance(x, float):
        return float(f"{x:.{FLOAT_DIGITS}g}")  # significant digits: a p of 8e-25 must not become 0.0
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return x


def stable_json(obj) -> str:
    """JSON with floats rounded to 12 significant digits, so the committed outputs are byte-identical across
    CPython versions (3.12 changed float summation, which moves the 16th digit of some means)."""
    return json.dumps(_rounded(json.loads(json.dumps(obj, default=float))), indent=1)


def run_all(out_dir: Path, ledger_path: Path = LEDGER, *, fold_injury: bool = False) -> dict:
    """fold_injury scores cause-of-death under COD v2's secondary scoring (Injury -> ExternalCause)
    everywhere; the default is the preregistered labels the ledger was written against."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, dict] = {}
    results: dict[str, dict] = {}

    for key in cod.LLM_METHODS:
        prof = cod.profile(key, fold_injury=fold_injury)
        profiles[f"cause_of_death@{prof.model}"] = prof.to_dict()
        results[f"cause_of_death@{prof.model}"] = cod.interventions(key, fold_injury=fold_injury)
    results["cause_of_death"] = results["cause_of_death@claude-sonnet-5"]  # registered primary instrument (paper 1)

    for model in kripke.MODELS:
        for run in ("gold", "product"):
            prof = kripke.profile(model, run)
            profiles[f"kripke@{run}@{prof.model}"] = prof.to_dict()
        results[f"kripke@{kripke.SHORT[model]}"] = kripke.interventions(model)
    results["kripke"] = results["kripke@1.5B"]  # registered: the larger of the two published models

    ledger = Ledger.load(ledger_path, verify=False)
    score = score_ledger(ledger, results)

    (out_dir / "profiles.json").write_text(stable_json(profiles))
    (out_dir / "results.json").write_text(stable_json(results))
    (out_dir / "ledger_score.json").write_text(stable_json(score))
    report = render_report(profiles, results, score, fold_injury=fold_injury)
    (out_dir / "RESULTS_offline.md").write_text(report)
    return {"profiles": profiles, "results": results, "score": score, "report": report}


def _f(x, d=3):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def probe_sentence(mp: dict) -> str:
    src = "Source: `data/external/cod/probe_summary.json`, `data/external/cod/probe_results.jsonl`."
    verdict = mp["verdict"].split(":")[0]
    if mp["answered"] == 0:
        return (f"Memorisation probe: {mp['n']} of {mp['n']} probe calls returned empty; {verdict}. The exact-code rate "
                f"{mp['exact_code']:.2f} is over zero answered items, so it is not a score. {src}")
    return (f"Memorisation probe: {mp['answered']} of {mp['n']} probe calls answered; exact-code rate {mp['exact_code']:.2f}; "
            f"{verdict}. {src}")


def _cod_v2_rows() -> list[str]:
    """Reproduce COD v2's Injury-folded headline from the cached predictions."""
    test, _ = cod.load()
    hard = cod.hard_ids(test, cod.dictsim(test))
    runs = cod.cached_runs(test, fold_injury=True)
    v2 = cod.published_paper_stats()["folded"]
    f = runs["B_fuzzy"]
    rows = []
    for key in cod.LLM_METHODS:
        m = runs[key]
        fo = sum(1 for i in f.correct if f.correct[i] and not m.correct[i])
        mo = sum(1 for i in f.correct if m.correct[i] and not f.correct[i])
        rows += [f"| COD v2 folded: {m.model} accuracy | {v2[key]['all']['acc']:.4f} | {m.accuracy:.4f} |",
                 f"| COD v2 folded: {m.model} on HARD | {v2[key]['hard']['acc']:.4f} | {m.accuracy_on(hard):.4f} |",
                 f"| COD v2 folded: {m.model} vs fuzzy, fuzzy only / model only | {v2[key]['all']['fuzzy_only']} / "
                 f"{v2[key]['all']['model_only']} | {fo} / {mo} |"]
    return rows


def render_report(profiles: dict, results: dict, score: dict, *, fold_injury: bool = False) -> str:
    scoring = ("Cause-of-death is scored with Injury folded into ExternalCause (COD v2 secondary scoring), "
               "so its profile, interventions and ledger values differ from the preregistered-label report."
               if fold_injury else
               "Cause-of-death is scored on the preregistered labels; COD v2's Injury-folded scoring is reproduced in "
               "the table below and in full by `artifact-signals paper12 --fold-injury` (written to outputs/paper12_folded).")
    L = ["# Development set: retrospective reconstruction on papers 1 and 2", "",
         "Recomputed by `artifact-signals paper12` from `data/external/`. No model was run. Papers 1 and 2 are the",
         "development set: the instrument was built by inspecting them, so nothing here validates it. The ledger",
         "below is a protocol demonstration and is labelled retrospective by the scorer.", "", scoring, "",
         "## Reproduction check (published vs recomputed)", "",
         "| quantity | published | recomputed |", "|---|---:|---:|"]
    pub = cod.published()["metrics"]
    if fold_injury:  # compare with the scoring actually used, or the table reads as a reproduction failure
        v2 = cod.published_paper_stats()["folded"]["M_comm_sonnet5"]
        pub = {**pub, "M_comm_sonnet5": {"all": v2["all"]["acc"], "hard": v2["hard"]["acc"]}}
    c = results["cause_of_death"]
    tag = " (folded)" if fold_injury else ""
    L += [f"| COD fuzzy dictionary accuracy | {pub['B_fuzzy']['all']:.4f} | {c['leakage_resplit']['before']['retrieval_accuracy']:.4f} |",
          f"| COD claude-sonnet-5 accuracy{tag} | {pub['M_comm_sonnet5']['all']:.4f} | {c['leakage_resplit']['before']['llm_accuracy']:.4f} |",
          f"| COD HARD subset n | 33 | {c['leakage_resplit']['n_kept']} |",
          f"| COD fuzzy on HARD | {pub['B_fuzzy']['hard']:.4f} | {c['leakage_resplit']['after']['retrieval_accuracy']:.4f} |",
          f"| COD claude-sonnet-5 on HARD{tag} | {pub['M_comm_sonnet5']['hard']:.4f} | {c['leakage_resplit']['after']['llm_accuracy']:.4f} |",
          f"| COD HARD pairwise (LLM only / retrieval only) | 14 / 1 | {c['leakage_resplit']['after']['pairwise']['llm_only']} / {c['leakage_resplit']['after']['pairwise']['retrieval_only']} |"]
    for m in kripke.MODELS:
        ps = kripke.published("gold")["per_model"][m]
        k = results[f"kripke@{kripke.SHORT[m]}"]["constraint_ablation"]["per_run"]["gold"]
        L.append(f"| Kripke {kripke.SHORT[m]} gold: strict exact prompted / constrained | {ps['prompted']['strict_exact']} / {ps['constrained']['strict_exact']} | "
                 f"{round(k['em']['free'] * 80)} / {round(k['em']['constrained'] * 80)} |")
        L.append(f"| Kripke {kripke.SHORT[m]} gold: forced fixtures | {kripke.published('gold')['forced_fixtures']} | {k['n_forced']} |")
        pp = kripke.published("product")["per_model"][m]
        kp = results[f"kripke@{kripke.SHORT[m]}"]["constraint_ablation"]["per_run"]["product"]
        L.append(f"| Kripke {kripke.SHORT[m]} product: strict exact constrained | {pp['constrained']['strict_exact']} | {round(kp['em']['constrained'] * 80)} |")
    L += _cod_v2_rows()
    L += ["", "## Artifact Risk Profiles", ""]
    for name, p in profiles.items():
        L += [f"### {name}", "", "| signal | level | value | note |", "|---|---|---:|---|"]
        for k, s in p["signals"].items():
            extra = ""
            d = s["details"]
            if k == "retrieval_dominance" and d.get("sim_correctness_corr") is not None:
                extra = f"corr={d['sim_correctness_corr']:+.3f}, gap={d['gap']:+.3f}, best={d.get('best_method')}"
            if k == "semantic_validity_gap" and "constraint_gain" in d:
                extra = f"cg={d['constraint_gain']:+.3f}, sg={d['semantic_gain']:+.3f}" + (f", forced_rate={d['forced_rate']:.3f}" if "forced_rate" in d else "")
            if k == "format_sensitivity" and d.get("prompt_variance"):
                pv = d["prompt_variance"]
                extra = (f"prompt variance (SD over {pv['n_prompts']} prompts, not the gate statistic)={pv['value']:.3f}; "
                         + ", ".join(f"{r}={v['accuracy']:.3f}" for r, v in pv["per_rendering"].items())
                         + f"; truth-value SD={pv['truth_value']['value']:.3f}")
            if k == "partial_input_accuracy" and d.get("label_marginal"):
                extra = f"label-vs-uniform={d['label_marginal']['tv_vs_uniform']:.3f}"
                if d.get("model_truth_dist_prompted"):
                    extra += "; model truth dist: " + ", ".join(f"{a}={b}" for a, b in d["model_truth_dist_prompted"].items())
            if k == "scoring_key_validity" and "asymmetric_labels" in d:
                extra = (f"label-space asymmetry={d['label_space_asymmetry']:.3f} ({', '.join(d['asymmetric_labels'])}); "
                         "predictions in those labels: " + ", ".join(f"{m}={c}" for m, c in d["predictions_in_asymmetric_labels"].items())
                         + f"; rule null rate={d['rule_null_rate']:.3f}")
            if k == "abstention_failure" and d.get("per_arm"):
                extra = ", ".join(f"{a}={v:.2f}" for a, v in d["per_arm"].items())
            note = (s.get("note") or "") + (("; " + extra) if (extra and s.get("note")) else extra)
            flag = " (GATED)" if s["gated"] else ""
            if d.get("proxy"):
                flag += " (proxy)"
            L.append(f"| {k}{flag} | {s['level']} | {_f(s['value'])} | {note} |")
        L.append("")
    L += ["## Interventions", ""]
    c = results["cause_of_death"]
    L += ["### cause_of_death / claude-sonnet-5, leakage-aware resplit (drop items with dictionary similarity >= threshold)", "",
          "| threshold | n kept | LLM acc | retrieval acc | gap | sim-correctness corr | LLM only / retrieval only |", "|---:|---:|---:|---:|---:|---:|---:|",
          f"| none | {c['n']} | {c['leakage_resplit']['before']['llm_accuracy']:.3f} | {c['leakage_resplit']['before']['retrieval_accuracy']:.3f} | "
          f"{c['leakage_resplit']['before']['gap']:+.3f} | {c['leakage_resplit']['before']['sim_correctness_corr']:+.3f} | "
          f"{c['leakage_resplit']['before']['pairwise']['llm_only']} / {c['leakage_resplit']['before']['pairwise']['retrieval_only']} |"]
    for thr, s in c["leakage_resplit"]["sweep"].items():
        a = s["after"]
        L.append(f"| < {thr} | {s['n_kept']} | {a['llm_accuracy']:.3f} | {a['retrieval_accuracy']:.3f} | {a['gap']:+.3f} | "
                 f"{_f(a['sim_correctness_corr'])} | {a['pairwise']['llm_only']} / {a['pairwise']['retrieval_only']} |")
    r = c["rebalance_and_filter"]
    L += ["", f"Cap-based rebalancing (<= {r['cap']} per chapter, n={r['n_after']}): LLM accuracy {r['before']['llm_accuracy']:.3f} -> {r['after']['llm_accuracy']:.3f}, "
          f"retrieval {r['before']['retrieval_accuracy']:.3f} -> {r['after']['retrieval_accuracy']:.3f}, gap {r['before']['gap']:+.3f} -> {r['after']['gap']:+.3f}; "
          f"label TV vs uniform {r['label_marginal']['tv_label_vs_uniform_before']:.3f} -> {r['label_marginal']['tv_label_vs_uniform']:.3f}.", ""]
    mp = c.get("memorization_probe")
    if mp:
        L += [probe_sentence(mp), ""]
    for m in kripke.MODELS:
        k = results[f"kripke@{kripke.SHORT[m]}"]
        L += [f"### kripke / {kripke.SHORT[m]}, constraint ablation (prompted vs constrained), EM = strict parse, semantic = truth value", "",
              "| run | forced rate | EM free | EM constrained | EM gain | sem free | sem constrained | sem gain | SVG |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for run, d in k["constraint_ablation"]["per_run"].items():
            L.append(f"| {run} | {d['forced_rate']:.3f} | {d['em']['free']:.3f} | {d['em']['constrained']:.3f} | {d['em']['constrained_minus_free']:+.3f} | "
                     f"{d['sem']['free']:.3f} | {d['sem']['constrained']:.3f} | {d['sem']['constrained_minus_free']:+.3f} | {d['svg']:+.3f} |")
        lr = k["leak_removal"]
        L += ["", f"Leak removal (gold -> product admissible set), constrained arm: EM {lr['em']['before']:.3f} -> {lr['em']['after']:.3f} "
              f"(forced subset n={lr['forced_subset']['n']}: {lr['forced_subset']['em_before']:.3f} -> {lr['forced_subset']['em_after']:.3f}; "
              f"free subset n={lr['free_subset']['n']}: {lr['free_subset']['em_before']:.3f} -> {lr['free_subset']['em_after']:.3f}, "
              f"{lr['free_subset']['outputs_identical']}/{lr['free_subset']['n']} outputs byte-identical).",
              "Surface re-rendering over the three unconstrained arms: strict EM " + ", ".join(f"{a}={v:.3f}" for a, v in k["surface_rerender"]["strict"]["per_arm"].items())
              + "; truth-value accuracy " + ", ".join(f"{a}={v:.3f}" for a, v in k["surface_rerender"]["truth_value"]["per_arm"].items()) + ".",
              "Abstention on the 80 out-of-language fixtures, answered rate per arm: " + ", ".join(f"{a}={v:.2f}" for a, v in k["abstention_enabled"]["ood_answered_rate"].items()) + ".", ""]
    L += ["## Ledger score", "", "```", format_score(score), "```", "",
          "Status: retrospective. The ledger's wording predates inspecting the cached outputs, but the papers' results were "
          "public and the mapping from each prediction to a concrete cached quantity (model, run, judge) was fixed while building "
          "this report. The prospective ledger for the test set is `ledger/prospective_template.json`; its freeze procedure is in its meta.", ""]
    return "\n".join(L)
