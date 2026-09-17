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


def run_all(out_dir: Path, ledger_path: Path = LEDGER) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, dict] = {}
    results: dict[str, dict] = {}

    for key in cod.LLM_METHODS:
        prof = cod.profile(key)
        profiles[f"cause_of_death@{prof.model}"] = prof.to_dict()
        results[f"cause_of_death@{prof.model}"] = cod.interventions(key)
    results["cause_of_death"] = results["cause_of_death@claude-sonnet-5"]  # registered primary instrument (paper 1)

    for model in kripke.MODELS:
        for run in ("gold", "product"):
            prof = kripke.profile(model, run)
            profiles[f"kripke@{run}@{prof.model}"] = prof.to_dict()
        results[f"kripke@{kripke.SHORT[model]}"] = kripke.interventions(model)
    results["kripke"] = results["kripke@1.5B"]  # registered: the larger of the two published models

    ledger = Ledger.load(ledger_path, verify=False)
    score = score_ledger(ledger, results)

    (out_dir / "profiles.json").write_text(json.dumps(profiles, indent=1, default=float))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (out_dir / "ledger_score.json").write_text(json.dumps(score, indent=1, default=float))
    report = render_report(profiles, results, score)
    (out_dir / "RESULTS_offline.md").write_text(report)
    return {"profiles": profiles, "results": results, "score": score, "report": report}


def _f(x, d=3):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def render_report(profiles: dict, results: dict, score: dict) -> str:
    L = ["# Development set: retrospective reconstruction on papers 1 and 2", "",
         "Recomputed by `artifact-signals paper12` from `data/external/`. No model was run. Papers 1 and 2 are the",
         "development set: the instrument was built by inspecting them, so nothing here validates it. The ledger",
         "below is a protocol demonstration and is labelled retrospective by the scorer.", "",
         "## Reproduction check (published vs recomputed)", "",
         "| quantity | published | recomputed |", "|---|---:|---:|"]
    pub = cod.published()["metrics"]
    c = results["cause_of_death"]
    L += [f"| COD fuzzy dictionary accuracy | {pub['B_fuzzy']['all']:.4f} | {c['leakage_resplit']['before']['retrieval_accuracy']:.4f} |",
          f"| COD claude-sonnet-5 accuracy | {pub['M_comm_sonnet5']['all']:.4f} | {c['leakage_resplit']['before']['llm_accuracy']:.4f} |",
          f"| COD HARD subset n | 33 | {c['leakage_resplit']['n_kept']} |",
          f"| COD fuzzy on HARD | {pub['B_fuzzy']['hard']:.4f} | {c['leakage_resplit']['after']['retrieval_accuracy']:.4f} |",
          f"| COD claude-sonnet-5 on HARD | {pub['M_comm_sonnet5']['hard']:.4f} | {c['leakage_resplit']['after']['llm_accuracy']:.4f} |",
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
            if k == "format_sensitivity" and d.get("per_rendering"):
                extra = ", ".join(f"{r}={v['accuracy']:.3f}" for r, v in d["per_rendering"].items())
                if "truth_value" in d:
                    extra += f"; truth-value SD={d['truth_value']['value']:.3f}"
            if k == "label_prior_skew" and d.get("model_marginal"):
                top = sorted(d["model_marginal"].items(), key=lambda kv: -kv[1])[:3]
                extra = "model top: " + ", ".join(f"{a}={b:.2f}" for a, b in top) + f"; label-vs-uniform={d['tv_label_vs_uniform']:.3f}"
            if k == "abstention_failure" and d.get("per_arm"):
                extra = ", ".join(f"{a}={v:.2f}" for a, v in d["per_arm"].items())
            note = (s.get("note") or "") + ((" | " + extra) if extra else "")
            flag = " (GATED)" if s["gated"] else ""
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
          f"label TV vs uniform {r['label_prior_skew']['tv_label_vs_uniform_before']:.3f} -> {r['label_prior_skew']['tv_label_vs_uniform']:.3f}.", ""]
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
              f"Surface re-rendering over the three unconstrained arms: strict EM " + ", ".join(f"{a}={v:.3f}" for a, v in k["surface_rerender"]["strict"]["per_arm"].items())
              + "; truth-value accuracy " + ", ".join(f"{a}={v:.3f}" for a, v in k["surface_rerender"]["truth_value"]["per_arm"].items()) + ".",
              f"Abstention on the 80 out-of-language fixtures, answered rate per arm: " + ", ".join(f"{a}={v:.2f}" for a, v in k["abstention_enabled"]["ood_answered_rate"].items()) + ".", ""]
    L += ["## Ledger score", "", "```", format_score(score), "```", "",
          "Status: retrospective. The ledger's wording predates inspecting the cached outputs, but the papers' results were "
          "public and the mapping from each prediction to a concrete cached quantity (model, run, judge) was fixed while building "
          "this report. The prospective ledger for the test set is `ledger/prospective_template.json`; its freeze procedure is in its meta.", ""]
    return "\n".join(L)
