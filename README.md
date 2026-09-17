# connect

Research program: **Reliable Evaluation of Reasoning in Language Models**.

Central question: how do we distinguish genuine model capabilities from benchmark
artifacts, retrieval shortcuts, format constraints, and evaluation effects?

| # | Paper | Question |
|---|---|---|
| 1 | Historical cause-of-death coding | When does semantic generalization genuinely outperform retrieval? |
| 2 | Kripke modal accessibility benchmark | When does benchmark improvement fail to reflect reasoning improvement? |
| 3 | [Predicting Benchmark Failure Before Benchmark Repair](papers/03-benchmark-artifact-signals/PROPOSAL.md) | Can we forecast artifact-driven performance before repairing the benchmark? |

Paper 3 promotes the findings of papers 1 and 2 from results to instruments:
retrieval dominance and the semantic-validity gap become two of five diagnostics
in an Artifact Risk Profile, validated by pre-registered prediction rather than
post-hoc description.

## `artifact_signals`: the instrument

A pure-Python package (no hard dependencies) that implements the paper 3 protocol.

```
pip install -e ".[dev]"        # add [fast] for rapidfuzz, [hf] for transformers, [api] for OpenAI-compatible endpoints
pytest                         # 38 tests, ~30 s, no GPU
python examples/synth_pipeline.py   # full two-phase dry run on SynthModal-Control with dummy models
```

### What it does

**Five diagnostics** (`signals.py`), reported as a vector and never collapsed to a scalar:

| Signal | Estimator | Clean value |
|---|---|---|
| `format_sensitivity` | SD of accuracy over 5 equivalent renderings (option permutation, token relabelling, instruction paraphrase, premise reorder, delimiter) | within re-run noise; gate at 0.03 |
| `partial_input_accuracy` | best excess over chance with options-only / question-only / premises-only input | 0 |
| `retrieval_dominance` | best of fuzzy / char-TF-IDF / BM25 nearest-neighbour accuracy over LLM accuracy, with the similarity-correctness point-biserial | low ratio, corr near 0 |
| `semantic_validity_gap` | (EM gain from constrained decoding) minus (semantic gain from constrained decoding) | 0 |
| `abstention_failure` | fraction of undetermined probes answered when an explicit abstain option is offered; risk-coverage AUC | 0 |

Format sensitivity is computed first and acts as a gate: above the threshold, the other
four are marked uninterpretable at the benchmark level. The label prior is reported inside
partial-input accuracy (a prior only matters if a model can exploit it without the input);
`label_prior_skew` remains in the library as an auxiliary and is not part of the paper.

**Five interventions** (`interventions.py`), each with a manipulation check on the signal it
targets, plus a cross-model invariance check that voids predictions when a filter removes
signal rather than artifact:
leakage-aware resplit, label rebalancing with partial-input filtering, surface re-rendering
(score as the mean over renderings), constraint ablation, and abstention-enabled scoring
with injected OOD probes.

**A prediction ledger** (`ledger.py`): predictions with direction, threshold, confidence and
family; canonical SHA-256 stamp; scoring by hit rate, Brier score, reliability table, per-family
hit rate, and a paired sign test against the naive predictor that says every change goes down.
Every ledger carries a status the scorer prints first: `retrospective` (outcomes were knowable
when it was written; a protocol demonstration) or `prospective` (hashed and deposited before any
model ran; the only kind that counts as a result). See `ledger/README.md`.

Every signal carries a **level**: `dataset`, `protocol`, `model`, or `model_x_benchmark`. The
forced-answer rate is protocol-level and needs no model; format sensitivity is an interaction.
The profile mixes levels on purpose and labels each one.

**SynthModal-Control** (`modal.py`, `synth.py`): a generated Kripke-semantics benchmark
with a real three-valued model checker. Balanced labels, per-item novel names so no lexical
overlap, and principled "insufficient information" probes (items whose truth value is
genuinely undetermined). It is the negative control: a diagnostic that fires here is broken.

**Model adapters** (`models.py`): dummy models that encode the failure modes the instrument
should detect (oracle, position bias, constant answer, nearest-neighbour retrieval, a model
whose only problem is format noise), plus `HFModel` (transformers, choice constraint by
candidate log-prob scoring) and `OpenAICompatModel` (soft constraint, flagged as such).

### Papers 1 and 2, offline

The two source repositories ship their cached model outputs. `scripts/sync_external.py`
vendors the slices the analysis needs into `data/external/`, and

```
artifact-signals paper12 --out outputs/paper12 --report
```

reproduces every published number to the item (tests in `tests/test_benchmarks.py`), computes
the Artifact Risk Profile for each benchmark and model, runs the offline interventions
(leakage sweep and rebalancing on cause-of-death; constraint ablation, leak removal, prompt
re-rendering and abstention on Kripke), and scores the ledger. Report:
[`outputs/paper12/RESULTS_offline.md`](outputs/paper12/RESULTS_offline.md).

Headline numbers from that run: Kripke fails the format gate (SD 0.13 over prompt arms vs the
0.03 gate) and nobody abstains in 1,600 outputs; its constraint leak shows up as
`forced_rate = 0.31` before any model runs and the leak-free replication keeps SVG at 0.39.
Cause-of-death has no format artifact (SVG = 0), and its lexical artifact inflates the baseline
rather than the model: under the leakage sweep retrieval falls 0.64 to 0.19 while the LLM rises
0.66 to 0.88. The retrospective ledger scores 4 of 5; the miss (P01) predicted the LLM's own score would drop.
Papers 1 and 2 are the development set, so that score demonstrates the protocol and validates
nothing; the prospective ledger for FOLIO, LogiQA and the synthetic control is the paper's result.

### CLI

```
artifact-signals synth    --n 400 --ood-frac 0.1 --out control.jsonl
artifact-signals diagnose --benchmark eval.jsonl --train train.jsonl --model hf:Qwen/Qwen2.5-7B-Instruct \
                          --semantic lenient --cache .cache --out profile.json
artifact-signals ledger hash  ledger/prospective_template.json --stamp
artifact-signals ledger score ledger/retrospective_papers12.json results.json --void P09
```

Model specs: `dummy:oracle:0.8`, `dummy:first`, `dummy:constant:TRUE`, `dummy:retrieval`,
`dummy:noisy:0.6:0.5`, `hf:<model_id>`, `openai:<model>[@base_url]`.

### Layout

```
papers/03-benchmark-artifact-signals/PROPOSAL.md   the proposal
src/artifact_signals/                              the instrument
src/artifact_signals/benchmarks/{cod,kripke}.py    adapters for papers 1 and 2 (reproduce, profile, intervene)
data/external/                                     vendored slices of both source repos (scripts/sync_external.py)
outputs/paper12/                                   offline Phase 1 on papers 1 and 2
ledger/retrospective_papers12.json                 development-set ledger (protocol demonstration)
ledger/prospective_template.json                   test-set ledger, unfrozen; freeze procedure in its meta
examples/synth_pipeline.py                         two-phase dry run on the synthetic control
tests/                                             38 tests
```
