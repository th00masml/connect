# Paper 3, Phase 1 on papers 1 and 2: offline diagnostics from cached outputs

Recomputed by `artifact-signals paper12` from `data/external/`. No model was run.

## Reproduction check (published vs recomputed)

| quantity | published | recomputed |
|---|---:|---:|
| COD fuzzy dictionary accuracy | 0.7067 | 0.7067 |
| COD claude-sonnet-5 accuracy | 0.6667 | 0.6667 |
| COD HARD subset n | 33 | 33 |
| COD fuzzy on HARD | 0.3636 | 0.3636 |
| COD claude-sonnet-5 on HARD | 0.7576 | 0.7576 |
| COD HARD pairwise (LLM only / retrieval only) | 14 / 1 | 14 / 1 |
| Kripke 0.5B gold: strict exact prompted / constrained | 26 / 51 | 26 / 51 |
| Kripke 0.5B gold: forced fixtures | 25 | 25 |
| Kripke 0.5B product: strict exact constrained | 36 | 36 |
| Kripke 1.5B gold: strict exact prompted / constrained | 8 / 52 | 8 / 52 |
| Kripke 1.5B gold: forced fixtures | 25 | 25 |
| Kripke 1.5B product: strict exact constrained | 37 | 37 |

## Artifact Risk Profiles

### cause_of_death@qwen2.5:14b

| signal | value | note |
|---|---:|---|
| format_sensitivity | n/a | single prompt in the cached run; needs a live re-run under standard_renderings |
| partial_input_accuracy | 0.108 | proxy from the label prior, not a model run |
| label_prior_skew | 0.273 |  | model top: Infectious=0.17, Digestive=0.12, Circulatory=0.11; label-vs-uniform=0.408 |
| retrieval_dominance | 1.262 |  | corr=+0.055, gap=-0.147, best=char3 |
| semantic_validity_gap | 0.003 |  | cg=+0.003, sg=+0.000 |
| abstention_failure | n/a | no unanswerable probes in the cached run; interventions.make_unanswerable for a live run |

### cause_of_death@claude-sonnet-5

| signal | value | note |
|---|---:|---|
| format_sensitivity | n/a | single prompt in the cached run; needs a live re-run under standard_renderings |
| partial_input_accuracy | 0.108 | proxy from the label prior, not a model run |
| label_prior_skew | 0.187 |  | model top: Digestive=0.16, Infectious=0.15, Circulatory=0.10; label-vs-uniform=0.408 |
| retrieval_dominance | 1.060 |  | corr=-0.047, gap=-0.040, best=char3 |
| semantic_validity_gap | 0.000 |  | cg=+0.000, sg=+0.000 |
| abstention_failure | n/a | no unanswerable probes in the cached run; interventions.make_unanswerable for a live run |

### kripke@gold@0.5B

| signal | value | note |
|---|---:|---|
| format_sensitivity | 0.137 | SD over the three unconstrained prompt arms (naive / prompted / prompt_tuned), strict parse | naive=0.000, prompted=0.325, prompt_tuned=0.237; truth-value SD=0.188 |
| partial_input_accuracy (GATED) | 0.050 | proxy from the label prior, not a model run; gated: format sensitivity 0.137 > 0.03 |
| label_prior_skew (GATED) | 0.550 | gated: format sensitivity 0.137 > 0.03 | model top: TRUE=1.00, FALSE=0.00; label-vs-uniform=0.050 |
| retrieval_dominance (GATED) | 1.778 | gated: format sensitivity 0.137 > 0.03 | corr=+0.389, gap=-0.350, best=loo-char3 |
| semantic_validity_gap (GATED) | 0.125 | EM = strict canonical parse; semantic = truth value under lenient parse; prompted vs constrained arm; gated: format sensitivity 0.137 > 0.03 | cg=+0.312, sg=+0.187, forced_rate=0.312 |
| abstention_failure (GATED) | 1.000 | gated: format sensitivity 0.137 > 0.03 | naive=1.00, prompted=1.00, prompt_tuned=1.00, constrained=1.00, constrained_tuned=1.00 |

### kripke@product@0.5B

| signal | value | note |
|---|---:|---|
| format_sensitivity | 0.142 | SD over the three unconstrained prompt arms (naive / prompted / prompt_tuned), strict parse | naive=0.000, prompted=0.338, prompt_tuned=0.237; truth-value SD=0.190 |
| partial_input_accuracy (GATED) | 0.050 | proxy from the label prior, not a model run; gated: format sensitivity 0.142 > 0.03 |
| label_prior_skew (GATED) | 0.550 | gated: format sensitivity 0.142 > 0.03 | model top: TRUE=1.00, FALSE=0.00; label-vs-uniform=0.050 |
| retrieval_dominance (GATED) | 1.778 | gated: format sensitivity 0.142 > 0.03 | corr=+0.389, gap=-0.350, best=loo-char3 |
| semantic_validity_gap (GATED) | 0.112 | EM = strict canonical parse; semantic = truth value under lenient parse; prompted vs constrained arm; gated: format sensitivity 0.142 > 0.03 | cg=+0.112, sg=+0.000, forced_rate=0.000 |
| abstention_failure (GATED) | 1.000 | gated: format sensitivity 0.142 > 0.03 | naive=1.00, prompted=1.00, prompt_tuned=1.00, constrained=1.00, constrained_tuned=1.00 |

### kripke@gold@1.5B

| signal | value | note |
|---|---:|---|
| format_sensitivity | 0.130 | SD over the three unconstrained prompt arms (naive / prompted / prompt_tuned), strict parse | naive=0.312, prompted=0.100, prompt_tuned=0.000; truth-value SD=0.095 |
| partial_input_accuracy (GATED) | 0.050 | proxy from the label prior, not a model run; gated: format sensitivity 0.130 > 0.03 |
| label_prior_skew (GATED) | 0.386 | gated: format sensitivity 0.130 > 0.03 | model top: FALSE=0.94, TRUE=0.06; label-vs-uniform=0.050 |
| retrieval_dominance (GATED) | 1.600 | gated: format sensitivity 0.130 > 0.03 | corr=-0.401, gap=-0.300, best=loo-char3 |
| semantic_validity_gap (GATED) | 0.400 | EM = strict canonical parse; semantic = truth value under lenient parse; prompted vs constrained arm; gated: format sensitivity 0.130 > 0.03 | cg=+0.550, sg=+0.150, forced_rate=0.312 |
| abstention_failure (GATED) | 1.000 | gated: format sensitivity 0.130 > 0.03 | naive=1.00, prompted=1.00, prompt_tuned=1.00, constrained=1.00, constrained_tuned=1.00 |

### kripke@product@1.5B

| signal | value | note |
|---|---:|---|
| format_sensitivity | 0.135 | SD over the three unconstrained prompt arms (naive / prompted / prompt_tuned), strict parse | naive=0.312, prompted=0.062, prompt_tuned=0.000; truth-value SD=0.084 |
| partial_input_accuracy (GATED) | 0.050 | proxy from the label prior, not a model run; gated: format sensitivity 0.135 > 0.03 |
| label_prior_skew (GATED) | 0.359 | gated: format sensitivity 0.135 > 0.03 | model top: FALSE=0.91, TRUE=0.09; label-vs-uniform=0.050 |
| retrieval_dominance (GATED) | 1.778 | gated: format sensitivity 0.135 > 0.03 | corr=-0.363, gap=-0.350, best=loo-char3 |
| semantic_validity_gap (GATED) | 0.388 | EM = strict canonical parse; semantic = truth value under lenient parse; prompted vs constrained arm; gated: format sensitivity 0.135 > 0.03 | cg=+0.400, sg=+0.013, forced_rate=0.000 |
| abstention_failure (GATED) | 1.000 | gated: format sensitivity 0.135 > 0.03 | naive=1.00, prompted=1.00, prompt_tuned=1.00, constrained=1.00, constrained_tuned=1.00 |

## Interventions

### cause_of_death / claude-sonnet-5, leakage-aware resplit (drop items with dictionary similarity >= threshold)

| threshold | n kept | LLM acc | retrieval acc | gap | sim-correctness corr | LLM only / retrieval only |
|---:|---:|---:|---:|---:|---:|---:|
| none | 300 | 0.667 | 0.707 | -0.040 | -0.047 | 51 / 63 |
| < 0.4 | 16 | 0.875 | 0.188 | +0.688 | -0.086 | 11 / 0 |
| < 0.5 | 33 | 0.758 | 0.364 | +0.394 | -0.241 | 14 / 1 |
| < 0.6 | 69 | 0.696 | 0.493 | +0.203 | -0.187 | 20 / 6 |
| < 0.7 | 140 | 0.679 | 0.550 | +0.129 | -0.124 | 37 / 19 |
| < 0.8 | 227 | 0.661 | 0.639 | +0.022 | -0.100 | 47 / 42 |

Cap-based rebalancing (<= 10 per chapter, n=139): LLM accuracy 0.667 -> 0.741, retrieval 0.707 -> 0.698, gap -0.040 -> +0.043; label TV vs uniform 0.331 -> 0.180.

### kripke / 0.5B, constraint ablation (prompted vs constrained), EM = strict parse, semantic = truth value

| run | forced rate | EM free | EM constrained | EM gain | sem free | sem constrained | sem gain | SVG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold | 0.312 | 0.325 | 0.637 | +0.312 | 0.450 | 0.637 | +0.187 | +0.125 |
| product | 0.000 | 0.338 | 0.450 | +0.112 | 0.450 | 0.450 | +0.000 | +0.112 |

Leak removal (gold -> product admissible set), constrained arm: EM 0.637 -> 0.450 (forced subset n=25: 1.000 -> 0.400; free subset n=55: 0.473 -> 0.473, 55/55 outputs byte-identical).
Surface re-rendering over the three unconstrained arms: strict EM naive=0.000, prompted=0.325, prompt_tuned=0.237; truth-value accuracy naive=0.000, prompted=0.450, prompt_tuned=0.312.
Abstention on the 80 out-of-language fixtures, answered rate per arm: naive=1.00, prompted=1.00, prompt_tuned=1.00, constrained=1.00, constrained_tuned=1.00.

### kripke / 1.5B, constraint ablation (prompted vs constrained), EM = strict parse, semantic = truth value

| run | forced rate | EM free | EM constrained | EM gain | sem free | sem constrained | sem gain | SVG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold | 0.312 | 0.100 | 0.650 | +0.550 | 0.500 | 0.650 | +0.150 | +0.400 |
| product | 0.000 | 0.062 | 0.463 | +0.400 | 0.450 | 0.463 | +0.013 | +0.388 |

Leak removal (gold -> product admissible set), constrained arm: EM 0.650 -> 0.463 (forced subset n=25: 1.000 -> 0.400; free subset n=55: 0.491 -> 0.491, 55/55 outputs byte-identical).
Surface re-rendering over the three unconstrained arms: strict EM naive=0.312, prompted=0.100, prompt_tuned=0.000; truth-value accuracy naive=0.525, prompted=0.500, prompt_tuned=0.312.
Abstention on the 80 out-of-language fixtures, answered rate per arm: naive=1.00, prompted=1.00, prompt_tuned=1.00, constrained=1.00, constrained_tuned=1.00.

## Ledger score

```
Ledger 55e189554255  scored 5/14 (voided 0, missing 9)
  hit rate 0.800  (p vs coin 0.188)   Brier 0.140
  magnitude          n=1   hit rate 0.000
  metric_specific    n=3   hit rate 1.000
  manipulation_check n=1   hit rate 1.000
  vs naive (change preds, n=2): model 0.500  naive 0.000  model-only 1  naive-only 0  sign-test p 1
  [MISS] P01      cause_of_death/leakage_resplit  llm.accuracy.delta down 0.08  conf 0.70  value +0.091
  [HIT ] P02      cause_of_death/leakage_resplit  gap.llm_minus_retrieval.delta up 0.05  conf 0.70  value +0.434
  [HIT ] P03      cause_of_death/leakage_resplit  retrieval_dominance.sim_correctness_corr below 0.15  conf 0.80  value -0.241
  [HIT ] P04      kripke/constraint_ablation  em.constrained_minus_free up 0.15  conf 0.80  value +0.400
  [HIT ] P05      kripke/constraint_ablation  sem.constrained_minus_free flat 0.03  conf 0.80  value +0.013
  [----] P06      kripke/constraint_ablation  svg.spearman_vs_model_size above -0.5  conf 0.60  value n/a
  [----] P07      folio/none  partial_input_accuracy.excess_over_chance above 0.1  conf 0.65  value n/a
  [----] P08      folio/rebalance_and_filter  llm.accuracy.delta down 0.0  conf 0.75  value n/a
  [----] P09      folio/rebalance_and_filter  invariance.agreement_after above 0.6  conf 0.60  value n/a
  [----] P10      logiqa/none  format_sensitivity.value above 0.03  conf 0.50  value n/a
  [----] P11      synthmodal_control/leakage_resplit  llm.accuracy.delta flat 0.03  conf 0.85  value n/a
  [----] P12      synthmodal_control/rebalance_and_filter  llm.accuracy.delta flat 0.03  conf 0.85  value n/a
  [----] P13      synthmodal_control/surface_rerender  llm.accuracy.delta flat 0.03  conf 0.85  value n/a
  [----] P14      cause_of_death/abstention_enabled  ranking.kendall_tau_vs_unabstained below 1.0  conf 0.55  value n/a
```

Quantity mapping note: the ledger was written from the two papers' stated findings before the cached outputs were inspected, but the mapping from each prediction to a concrete cached quantity (which model, which run, which judge) was fixed while building this report. A real pre-registration fixes both in advance.
