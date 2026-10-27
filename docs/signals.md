# Diagnostic signals: estimators and clean values

Each diagnostic is a `Signal` in `src/artifact_signals/signals.py` with a value, an optional
cluster-bootstrap CI (10,000 resamples, `stats.N_BOOT`), a details dict, and a level saying what it is
a property of: `dataset`, `protocol`, `model` or `model_x_benchmark`. The profile is reported as a
vector and never collapsed to a scalar. "Clean value" is what a benchmark without the artifact
should show. "Offline (v0.2)" says what `artifact-signals paper12` computes from cached outputs,
without a model call.

| ID | Signal | Level |
|---|---|---|
| D-001 | `format_sensitivity` | model_x_benchmark |
| D-002 | `partial_input_accuracy` | dataset |
| D-003 | `retrieval_dominance` | dataset |
| D-004 | `semantic_validity_gap` (with `forced_rate`) | protocol |
| D-005 | `abstention_failure` | model_x_benchmark |
| D-006 | `scoring_key_validity` | dataset |

## D-001 format_sensitivity

Estimator: population SD of in-distribution accuracy over k = 5 meaning-preserving renderings
(`render.standard_renderings`: option permutation, answer-token relabelling, instruction paraphrase,
premise reorder, delimiter change). CI: cluster bootstrap of that SD. Details: per-rendering accuracy
and invalid rate, range, item agreement, McNemar of each rendering against the baseline.

Clean value: within re-run noise. Gate: above 0.03 the other signals are marked uninterpretable at
the benchmark level (`ArtifactRiskProfile.apply_gate`).

Offline (v0.2): n/a for both benchmarks. Cause-of-death has a single cached prompt. Kripke has three
different prompts (naive / prompted / prompt_tuned); their SD (0.130 to 0.142) is reported as
`details.prompt_variance`, not as this statistic, and the gate is not evaluated
(`gate_evaluated = false`).

## D-002 partial_input_accuracy

Estimator: best excess over chance of the same model given a deliberately insufficient input
(`render.partial_input_renderings`: options only, question only, premises only). Details carry the
label marginal and its total-variation distance from uniform, since a skewed prior matters only if a
model can exploit it without the input.

Clean value: 0 (chance).

Offline (v0.2): a proxy, not a model run, marked `(proxy)` in the report. Cause-of-death: majority
baseline accuracy minus 1/22 = 0.108. Kripke: majority truth value's share minus 0.5 = 0.050.

## D-003 retrieval_dominance

Estimator: accuracy of the best non-parametric nearest-neighbour baseline (fuzzy, char-3gram TF-IDF,
BM25) divided by LLM accuracy, with the point-biserial correlation between nearest-neighbour
similarity and item-level LLM correctness. The correlation is the shortcut evidence; the ratio is
suggestive only.

Clean value: low ratio, correlation near 0.

Offline (v0.2): computed. Cause-of-death against the 3,006-string dictionary; Kripke leave-one-out
over the 80 in-language fixtures (no separate training split exists).

## D-004 semantic_validity_gap (constraint gain, forced-answer rate)

Estimator: CG = EM(constrained) - EM(free); SVG = CG - (Sem(constrained) - Sem(free)). Alongside it,
`forced_rate`: the share of items whose admissible set contains exactly one answer, computed from the
constraint and the items before any model runs.

Clean value: SVG 0; forced_rate 0.

Offline (v0.2): computed. Kripke gold run forced_rate 0.312 (25/80), leak-free product run 0.
Cause-of-death compares strict decoding with the paper's lenient parse on the same outputs.

## D-005 abstention_failure

Estimator: share of out-of-distribution or undetermined probes answered rather than declined when
an explicit abstain option is in the prompt; risk-coverage AUC when model confidence is available.

Clean value: 0.

Offline (v0.2): Kripke 1.00 in every arm of both models (80 out-of-language fixtures per arm);
cause-of-death n/a (no unanswerable probes were cached).

## D-006 scoring_key_validity

Asks whether the key the benchmark is scored against can be trusted, before any model runs.

Estimator, three parts:

- value: disagreement rate between the benchmark label and an independent re-scoring of the same
  item (`independent_key`), over items the independent key can score. None without such a key.
- rule null rate: share of items the scoring rule (`rule`, or else the independent key) cannot key
  at all.
- label-space asymmetry: labels that one compared system can emit and another structurally cannot,
  as a share of the label space, with the number of predictions each system put in those labels.

Clean value: 0 for all three.

Reference behaviour: on SynthModal-Control the independent key is the three-valued model checker
re-evaluating each item's stored formula and frame (`synth.checker_key`); value 0, rule null rate 0
(`tests/test_scoring_key.py`).

Offline (v0.2): cause-of-death value n/a. Its chapter labels are derived from the ICD codes by the
same rule (`cod.chapter_from_code`, a port of paper 1's rule), so re-keying would only test
consistency (0 of 300 disagree). Rule null rate 0.000. Label-space asymmetry 2/22 = 0.091: the
dictionary baseline can never emit Injury or Special, the models can, and they put 15 (Sonnet) and
31 (Qwen) predictions in Injury. This is the asymmetry COD v2 corrects by folding Injury into
ExternalCause (`artifact-signals paper12 --fold-injury`). Kripke: not computed.

## Auxiliary, not a profile signal

`label_prior_skew`: TV distance between the model's answer marginal and the label marginal. Kept in
the library; its dataset half is reported inside D-002.
