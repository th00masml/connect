# Predicting Benchmark Failure Before Benchmark Repair

**Pre-Registered Diagnostic Signals for Evaluation Artifacts in LLM Reasoning Benchmarks**

Workshop-ready research proposal. Paper 3 of the research program *Reliable Evaluation of Reasoning in Language Models*.

---

## A. Title

**Predicting Benchmark Failure Before Benchmark Repair: Pre-Registered Diagnostic Signals for Evaluation Artifacts in LLM Reasoning Benchmarks**

The obvious alternative, *Diagnostic Signals of Benchmark Artifacts in Language Model Evaluation*, loses. It names a measurement exercise, and a measurement exercise is retrospective by construction: you report six numbers about datasets already suspected of being broken and nothing can come out false. The predictive title commits the paper to a falsifiable object, a ledger of registered predictions that can be wrong, and that is the only version of this work worth a reviewer's time.

---

## B. Abstract (200 words)

Benchmark artifacts are usually discovered after the fact: a benchmark is published, adopted, and only later shown to be solvable by a partial-input baseline or a lexical nearest neighbour. We ask whether that discovery can be moved earlier. We propose treating artifact diagnostics as forecasting instruments rather than post-hoc descriptions, and we test them by pre-registering predictions about what happens to benchmark scores once artifacts are removed. Six cheap diagnostics (format sensitivity, partial-input accuracy, label-prior skew, retrieval dominance, constrained-decoding gain versus semantic gain, and abstention failure) are measured on five reasoning benchmarks, two of them our own and one a synthetic negative control. Predictions about the direction, the relative magnitude, and the metric-specific pattern of post-intervention score change are registered with stated confidences before any intervention runs. We then apply targeted interventions (leakage-aware resplitting, label rebalancing, surface re-rendering, constraint ablation, abstention-enabled scoring), each with a manipulation check, and score the ledger against a naive predictor by hit rate and by calibration. The contribution is methodological: a reusable protocol, an open instrument, and first evidence on whether artifact diagnostics transfer across benchmark families. All experiments fit on a single 24 GB GPU.

---

## C. Research gap

Artifact analysis in NLP is retrospective, per-dataset, and uncoordinated. The canonical result in this genre takes a published benchmark and shows that a degenerate model solves most of it: hypothesis-only NLI, passage-free reading comprehension, answer-option-only multiple choice. Each such finding is valuable and each arrives too late to matter, after the benchmark has already shaped a year of modelling work.

Three gaps follow.

First, nobody has tested whether artifact diagnostics are *predictive*. The literature reports diagnostics and it reports repairs, but it does not register the diagnostic-derived prediction before the repair and then check it. So we do not know whether a partial-input baseline of 68% tells you anything quantitative about what happens when you filter the data, or whether it is a qualitative red flag and nothing more.

Second, the diagnostics live in disjoint literatures. Contamination and leakage detection, label-prior and partial-input analysis, prompt and format sensitivity, constrained decoding, and selective prediction each have their own venues and their own datasets. They are almost never measured jointly, on the same items, with the same models, under the same decoding settings. Their conditional structure is therefore unknown: we cannot say whether format sensitivity confounds the measurement of retrieval dominance, or whether a constraint gain is interpretable at all when label priors are skewed.

Third, and most damaging, artifact studies rarely include a negative control. They are run on datasets already believed to be flawed, which guarantees a positive finding and tells us nothing about specificity. A diagnostic that fires on everything is a thermometer that reads 39 degrees on a corpse.

This paper targets all three: joint measurement, predictive validation, and an explicit clean control.

---

## D. Main hypothesis

> **Artifact-driven benchmark performance is forecastable.** A small vector of diagnostics, measurable on a benchmark before any repair and at negligible cost, predicts the direction, the relative magnitude, and the metric-specific pattern of score change under targeted de-artifacting interventions, and does so better than a naive predictor which asserts only that scores go down.

The naive-predictor clause is the load-bearing part. Almost every intervention reduces almost every score, so directional accuracy is nearly free. The hypothesis is only interesting if the diagnostics buy something the trivial predictor cannot: which *metric* moves (exact match versus semantic validity), *how much* relative to other benchmarks in the set, and which items survive.

---

## E. Sub-hypotheses

**H1 (retrieval dominance predicts where the gap opens).** Where a fuzzy or BM25 nearest-neighbour baseline reaches a high fraction of LLM accuracy, leakage-aware resplitting will collapse the baseline and widen the LLM-minus-retrieval gap; the LLM's own score may rise, fall or hold, and H1 makes no claim about it. The similarity-correctness correlation is the mechanism check: on retrieval-driven benchmarks it is positive before the resplit and near zero after. (The offline run on cause-of-death is what forced this wording: the LLM's own accuracy *rose* on the novel slice, the gap widened by 0.43.)

**H2 (the semantic-validity gap is a property of the benchmark, not the model).** Where constrained decoding produces a large exact-match gain and a near-zero semantic-correctness gain, that dissociation will be stable across model scales and will not shrink as base capability rises. If the gap is instead a model deficiency, it should close with capability. This is the cleanest discriminating test in the study, because the two accounts make opposite predictions about the model-size trend.

**H3 (partial input plus label skew predicts a residual hard core).** Above-chance partial-input accuracy and skewed answer marginals predict that rebalancing and light adversarial filtering reduce headline accuracy while leaving a residual subset whose item-difficulty ordering is preserved across models. Artifact removal should compress scores toward the hard core, not scramble the difficulty structure. If filtering scrambles difficulty ordering, the filter removed signal rather than artifact, and the intervention itself is invalid.

**H4 (format sensitivity is a gate, not a peer signal).** Score variance across semantically equivalent surface renderings (option permutation, answer-token relabelling, instruction paraphrase) moderates every other diagnostic. On high-variance benchmarks, the other five diagnostics will show wider between-rendering spread than their own between-model spread, making them uninterpretable as benchmark properties. The methodological position this paper takes: measure format sensitivity first and refuse to report the other diagnostics as benchmark-level facts until it is below a pre-registered threshold.

**H5 (abstention failure predicts overstated novel-case capability).** Benchmarks where models answer nearly all out-of-distribution or unanswerable probes overstate performance on their novel slice. Introducing an explicit "insufficient information" option and abstention-aware scoring will reorder models, and the reordering will be larger where abstention failure was higher.

**H6 (specificity / negative control).** On the synthetic control benchmark, where all six diagnostics sit near their clean values by construction, every intervention will produce score changes statistically indistinguishable from re-run noise. Any diagnostic that fires on the control, or any intervention that moves the control, falsifies the instrument rather than the benchmark.

---

## F. Related work categories

Dataset artifacts and partial-input baselines. Adversarial filtering and dataset ablation methods. Shortcut learning and spurious correlation in supervised NLP. Prompt sensitivity, option-order bias, and surface-form effects in LLM evaluation. Constrained and grammar-guided decoding, and the validity of format-conditioned metrics. Training-data contamination and leakage detection. Retrieval versus parametric generalization, including nearest-neighbour and RAG-style baselines for classification. Selective prediction, calibration, and abstention. Measurement theory and construct validity applied to NLP evaluation. Pre-registration, multiverse analysis, and reproducibility practice imported from experimental psychology.

Specific citation is deferred to the full write-up, with two exceptions worth naming because the design depends on them: partial-input baselines as the canonical artifact test, and lightweight adversarial filtering as the canonical repair. The paper's claim is not that either is new; it is that neither has been validated as a *forecast*.

---

## G. Experimental design

### G.0 Phase 1 on benchmarks A and B is already done, offline

Both source repositories ship their cached model outputs (`cause-of-death-coding/outputs/`,
`kripke-modal-accessibility/outputs/{gold,product}/`). The instrument reads them directly
(`artifact-signals paper12`), reproduces every published number to the item, and computes
the profile and the offline-computable interventions without a model call. The full report is
`outputs/paper12/RESULTS_offline.md`. What it already settles:

- **Kripke fails the format-sensitivity gate by a factor of four.** Strict exact match across the
  three unconstrained prompt arms has SD 0.13 to 0.14 for both models (gate: 0.03); even the
  truth-value judge that reads through format noise gives SD 0.08 to 0.19. Every other signal on
  that benchmark is reported gated. H4 is not a hypothesis there, it is the observed regime.
- **The constraint leak is measurable before the repair.** With the admissible set built from
  gold judgements, 25 of 80 fixtures admit only one truth value. The instrument reports this as
  `forced_rate` = 0.312 inside the semantic-validity-gap signal, computed from the constraint and
  the fixtures alone, no model needed. The leak-free replication has `forced_rate` = 0 and the
  55 unaffected fixtures come back byte-identical, which is the manipulation check passing.
- **The semantic-validity gap survives the repair.** On the leak-free run the 1.5B model's exact
  match rises by 40 points under the constraint while its truth-value accuracy moves by 1.3
  points (SVG = 0.39). Registered predictions P04 and P05 both hold.
- **Retrieval dominance on cause-of-death predicts the gap, not the LLM's own delta.** Prediction
  P01 (LLM accuracy falls by more than 8 points on the novel slice) fails: it rises by 9 points.
  P02 (the LLM-minus-retrieval gap widens by at least 5 points) holds with room to spare
  (+0.43). The artifact inflates the *baseline*, and a directional prediction about the LLM's
  own score was the naive predictor's mistake, not a diagnostic's. H1 is restated below
  accordingly.
- **Nobody abstains, in 1,600 outputs.** Abstention failure is 1.00 in every arm of every model,
  including the arms whose prompt says to output empty on out-of-language atoms.
- **Cause-of-death has no format artifact.** Strict decoding and the paper's lenient parser agree
  on all 300 outputs of both models; SVG = 0. The memorization probe is 0.00. Its artifact is
  the lexical one only, and the leakage sweep shows it cleanly: as the dictionary-similarity cap
  drops from 0.8 to 0.4 the retrieval baseline falls from 0.64 to 0.19 while the LLM climbs from
  0.66 to 0.88.

The ledger scores 4 of 5 scorable predictions (P06 needs a third model size, the rest need
benchmarks C to E). The honest caveat is in the report: the ledger's wording predates reading the
cached outputs, but the mapping from prediction to concrete cached quantity was fixed while
building the report. The full study fixes both in advance.

### G.1 Materials

Five benchmarks, chosen so that the set spans artifact types rather than topics.

| Benchmark | Source | Expected dominant artifact |
|---|---|---|
| Historical cause-of-death coding | Paper 1, own | lexical leakage, retrieval shortcut |
| Kripke modal accessibility | Paper 2, own | format/constraint effects, semantic-validity gap |
| FOLIO | public | label prior, partial-input solvability |
| LogiQA | public | option-order and distractor artifacts |
| SynthModal-Control | generated for this paper | none, by construction |

SynthModal-Control is generated from a small grammar over frames and accessibility relations, with balanced labels, randomized surface forms, and verified ground truth by model checking. It is a few hundred items from a generator script, not a dataset-creation effort, and it exists solely to test specificity. Without it the study cannot distinguish a working diagnostic from a diagnostic that always says yes.

Per benchmark: a locked measurement split of roughly 400 items for diagnostics, and a disjoint evaluation split of roughly 600 items for interventions. No item appears in both.

### G.2 Models

Three open-weight models spanning at least an 8x parameter range, run locally at temperature 0 with fixed seeds, plus one or two API models for external validity. Model identity is a covariate, never a subject of the paper. Anything that only reproduces on one model is reported as such.

### G.3 Two-phase protocol

**Phase 1, diagnosis.** Compute all six diagnostics on the measurement split. Compute format sensitivity first and apply the H4 gate. Write the prediction ledger: for each (benchmark, intervention) pair, a prediction of direction, of magnitude bucket, of which metric moves, and a stated subjective probability. Hash the ledger, tag it in the repository, deposit on OSF. Nothing from the evaluation split has been touched at this point.

**Phase 2, intervention.** Apply the interventions of Section J to the evaluation split, run the manipulation checks, score, and compare against the ledger.

The phase boundary is the whole experiment. It is enforced by keeping the evaluation split in a separate encrypted archive whose key is released by the tagged commit.

### G.4 Judging semantic correctness

Where a verifier exists, use it. Kripke items are checked by actual semantic evaluation over the model structure. FOLIO items go to a first-order theorem prover. Only the cause-of-death benchmark, where correctness is a coding judgment, requires an LLM judge, and there the judge is validated against 150 human-annotated items with agreement reported as Cohen's kappa; if kappa falls below 0.7, the semantic metric for that benchmark is reported as unreliable and H2 is tested on the other benchmarks only. Using an LLM judge to measure whether LLM benchmark gains are real is circular, and the paper should say so out loud rather than quietly accept it.

### G.5 Statistics

Item-level paired analysis throughout. McNemar for paired binary outcomes, cluster bootstrap (10k resamples, clustering on source item) for confidence intervals on score differences, Holm correction within each hypothesis family. Ledger evaluation uses two scores: hit rate against the naive always-drops predictor, tested by exact binomial, and Brier score over the stated confidences, with a reliability diagram. Reporting only "8 of 10 correct" over twelve predictions is not a result; the standard error on that is roughly 13 points. Calibration over a larger set of finer-grained predictions is the honest version, so the ledger is designed to contain 40 to 60 atomic predictions rather than a dozen coarse ones.

### G.6 Compute

Five benchmarks x 1000 items x 5 renderings x 4 models x 2 decoding conditions is on the order of 200k generations of a few hundred tokens each. Local 7B to 14B inference with vLLM on one 24 GB GPU handles the bulk in a few days of wall-clock; API models are run on a 300-item stratified subsample. Total budget: one GPU, roughly 300 EUR of API credit, one researcher, six to ten weeks.

---

## H. Diagnostic signals

Six signals, each with a cheap estimator and a stated clean value.

**1. Format sensitivity (FS).** Standard deviation of accuracy across k = 5 semantically equivalent renderings: option permutation, answer-token relabelling (TRUE/FALSE vs A/B vs yes/no), instruction paraphrase, premise reordering, whitespace and delimiter change. Clean value: within re-run noise. Gate threshold pre-registered at 3 accuracy points.

**2. Partial-input accuracy (PIA).** Accuracy of the same model given a deliberately insufficient input: answer options only, hypothesis only, query without premises. Clean value: chance. This is the most direct artifact indicator in the set and the cheapest to run.

**3. Label-prior skew (LPS).** Total variation distance between the model's marginal answer distribution and the benchmark's label marginal, plus the benchmark's own deviation from uniform. Separates "the data is skewed" from "the model is skewed", which are routinely conflated.

**4. Retrieval dominance (RD).** Accuracy of a non-parametric baseline (character n-gram TF-IDF plus token-set fuzzy match, and BM25) divided by LLM accuracy, reported alongside the point-biserial correlation between nearest-neighbour similarity and item-level LLM correctness. The correlation matters more than the ratio: a strong baseline is suggestive, but correctness tracking lexical proximity is the actual shortcut evidence.

**5. Constraint gain and semantic-validity gap (CG, SVG), with forced-answer rate.** CG = EM(constrained) - EM(free). SVG = CG - (Sem(constrained) - Sem(free)). A large positive SVG says the constraint bought form and not content. Alongside it, the *forced-answer rate*: the fraction of items on which the admissible set contains exactly one answer, computed from the constraint and the items before any model runs. Paper 2's leak was a forced rate of 0.31; a constraint with a non-zero forced rate is answering the benchmark itself. This signal is Paper 2's finding, promoted to an instrument.

**6. Abstention failure (AF).** Fraction of unanswerable or out-of-distribution probes answered rather than declined, measured with an explicit abstention option present in the prompt. Complemented by the risk-coverage AUC when the model's own confidence is used to trigger abstention.

The six are reported as a vector, the **Artifact Risk Profile**, and deliberately not collapsed into a scalar index. A scalar would be more quotable and would destroy the conditional structure (H4's gating, H2's metric specificity) that makes the predictions falsifiable in the first place. Resisting the composite is part of the methodological contribution.

---

## I. Pre-registered predictions

Illustrative entries from the ledger, each stated with its confidence, each falsifiable.

1. Cause-of-death, leakage-aware resplit: LLM accuracy falls by more than 8 points absolute; the retrieval baseline falls by more, so the LLM-minus-retrieval gap *widens* by at least 5 points. Confidence 0.7. (Paper 1 predicts widening because the LLM advantage was concentrated on novel cases; if the gap narrows instead, H1's causal story is wrong.)
2. Cause-of-death, the point-biserial correlation between fuzzy similarity and correctness drops below 0.15 after resplitting. Confidence 0.8. This is the manipulation check for intervention 1.
3. Kripke, constraint ablation: EM drops by more than 15 points when the grammar is removed; verified semantic correctness changes by less than 3 points. Confidence 0.8.
4. Kripke, the SVG does not shrink monotonically with model size across the four models. Confidence 0.6. This is the H2 discriminator and the prediction most likely to fail.
5. FOLIO, partial-input accuracy exceeds chance by more than 10 points before intervention. Confidence 0.65.
6. FOLIO, after label rebalancing and partial-input filtering: headline accuracy falls, and the cross-model Spearman correlation of item difficulty on the surviving items stays above 0.6. Confidence 0.6.
7. LogiQA, format sensitivity exceeds the 3-point gate, and therefore its RD and SVG estimates are reported as ungated and uninterpretable at the benchmark level. Confidence 0.5.
8. SynthModal-Control, every intervention: no score change exceeds the re-run noise band. Confidence 0.85.
9. Across all benchmarks, abstention-aware scoring changes the model ranking on at least one benchmark, and that benchmark is the one with the highest pre-intervention AF. Confidence 0.55.
10. Ledger-level: hit rate exceeds the naive always-drops predictor by at least 20 points on the subset of predictions that specify *which metric* moves. Confidence 0.6.

The full ledger contains 40 to 60 such entries and is fixed before Phase 2.

---

## J. Intervention strategy

Every intervention obeys a minimal-edit principle: it may remove an artifact, it may not change the task. Every intervention carries a manipulation check that verifies it reduced the signal it targeted, and an invariance check that verifies it did not alter the construct.

| Intervention | Targets | Manipulation check |
|---|---|---|
| Leakage-aware resplit (nearest-neighbour similarity capped between train and test; novel-term-only slice) | RD | similarity-correctness correlation drops |
| Label rebalancing plus light adversarial filtering of partial-input-solvable items | LPS, PIA | partial-input accuracy returns to chance |
| Surface re-rendering with randomized option order, answer tokens, and instruction phrasing; score as the mean over renderings | FS | between-rendering variance falls below gate |
| Constraint ablation: free decoding versus grammar-constrained, with the semantic verifier held fixed | CG, SVG | EM and semantic metrics separate as designed |
| Abstention-enabled scoring: explicit "insufficient information" option, risk-coverage reporting, OOD probes injected at 10% | AF | abstention rate on OOD probes rises above 50% |

The invariance check is the part usually skipped. Adversarial filtering that removes 40% of items and also removes all the hard reasoning is not a repair, it is a different benchmark. The check is the cross-model difficulty-ordering correlation from H3: if it collapses, the intervention is reported as invalid and the corresponding predictions are voided rather than scored, with the voiding rule fixed in advance.

---

## K. Expected outcomes

The most likely result is partial success, and the paper is designed so that partial success is the interesting case. Directional predictions should come in well above chance and only modestly above the naive predictor. Metric-specific predictions, the ones that say exact match moves while semantic correctness does not, should be where the diagnostics earn their keep. Magnitude-bucket predictions should be the weakest, because the mapping from a diagnostic value to an effect size is not calibrated by anything yet, and the paper's honest contribution there is the first calibration data.

Deliverables regardless of outcome: an open-source instrument that computes the Artifact Risk Profile for an arbitrary benchmark in a few GPU-hours; the hash-stamped ledger and its scoring; five re-released evaluation splits with their repairs and manipulation checks; and a short practitioner's checklist for benchmark authors, which is the artifact most likely to be used by other people.

---

## L. Failure cases that remain publishable

**Diagnostics do not predict.** Hit rate indistinguishable from the naive predictor. This is a strong negative result and arguably the more useful paper: the field currently treats a high partial-input baseline as an implicit forecast about post-repair behaviour, and showing that the forecast does not hold means artifact diagnostics are qualitative flags only. Title becomes *Artifact Diagnostics Do Not Transfer Across Benchmark Families*.

**Diagnostics fire everywhere, including the control.** A specificity failure. Publishable as a calibration study, with recalibrated thresholds and a demonstration that the standard clean-value assumptions (chance-level partial input, near-zero format variance) are wrong in practice for instruction-tuned models.

**Interventions do not move scores.** Benchmarks are healthier than assumed. Publishable as a reassuring null with tight confidence intervals, which is rare and citable, provided the manipulation checks confirm the interventions actually removed the artifacts.

**Format sensitivity swamps everything.** Every benchmark fails the H4 gate and no other diagnostic is interpretable. This is a result about the measurement regime rather than about the benchmarks, and it directly motivates the follow-up work: format sensitivity would have to be controlled before any artifact analysis in the literature can be trusted, including the ones already published.

All four are reportable at an evaluation-focused workshop (BlackboxNLP, GEM, the Eval track at *ACL venues), and all four leave the instrument intact.

---

## M. How this connects Paper 1 and Paper 2

Paper 1 asks when semantic generalization beats retrieval and answers it on one domain. Paper 2 asks when benchmark improvement fails to reflect reasoning improvement and answers it on one benchmark. Both are single-case findings, and both are, structurally, the same finding: a reported score was produced by something other than the capability it claimed to measure. In Paper 1 that something is lexical proximity. In Paper 2 it is format compliance.

Paper 3 takes each of those findings and demotes it from a result to an instrument. Retrieval dominance becomes signal 4. The semantic-validity gap becomes signal 5. The question shifts from "was this particular benchmark measuring what it claimed" to "can we tell in advance, and how much does knowing help". That is the move from two case studies to a method, and it is the move that turns three repositories into a program.

It also gives the earlier two papers a job in the program's narrative that they cannot do alone: they are the validation cases where the ground truth about the artifact is already known, which is what makes the forecasting test possible at all. You cannot validate a forecasting instrument on benchmarks whose true artifact status is unknown.

---

## N. How this strengthens a future grant proposal

An NCN SONATA panel reads three separate benchmark papers as three separate hobbies. It reads a diagnostic instrument plus two validation cases as a programme with a method at its centre, and the one-line summary it needs becomes available:

> I build diagnostic methods that distinguish genuine language-model capabilities from evaluation artifacts.

Concretely, the proposal gains four things. A method rather than a collection of results, which is what the methodological-contribution criterion rewards. A feasibility argument backed by completed work, since Papers 1 and 2 are the pilot data for the forecasting claim and the reviewer can check that the pilot exists. A compute profile that matches the funding level honestly, one GPU and no foundation-model training, which removes the usual scepticism about whether a small Polish group can execute an LLM project. And a work-package structure that writes itself: WP1 extends the diagnostic battery and its calibration, WP2 builds the intervention library and its manipulation checks, WP3 tests transfer across benchmark families and languages, including Polish-language reasoning evaluation where artifact analysis is essentially absent, WP4 releases the instrument and the practitioner checklist.

The Polish-language angle deserves emphasis in the grant text specifically. Artifact analysis for non-English reasoning benchmarks is nearly nonexistent, translated benchmarks carry translation artifacts on top of the original ones, and a validated instrument is exactly the tool needed to check them. That is a defensible national-relevance argument that does not require inventing one.

---

## Scope and non-goals

No new large dataset. No foundation-model training. No leaderboard. No claim about which model is best. The unit of study is the benchmark, not the model, and model identity is a covariate throughout.
