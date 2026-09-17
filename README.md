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
retrieval dominance and the semantic-validity gap become two of six diagnostics
in an Artifact Risk Profile, validated by pre-registered prediction rather than
post-hoc description.

## `artifact_signals`: the instrument

A pure-Python package (no hard dependencies) that implements the paper 3 protocol.

```
pip install -e ".[dev]"        # add [fast] for rapidfuzz, [hf] for transformers, [api] for OpenAI-compatible endpoints
pytest                         # 33 tests, a few seconds, no GPU
python examples/synth_pipeline.py   # full two-phase dry run on SynthModal-Control with dummy models
```

### What it does

**Six diagnostics** (`signals.py`), reported as a vector and never collapsed to a scalar:

| Signal | Estimator | Clean value |
|---|---|---|
| `format_sensitivity` | SD of accuracy over 5 equivalent renderings (option permutation, token relabelling, instruction paraphrase, premise reorder, delimiter) | within re-run noise; gate at 0.03 |
| `partial_input_accuracy` | best excess over chance with options-only / question-only / premises-only input | 0 |
| `label_prior_skew` | TV distance between the model's answer marginal and the label marginal; label-vs-uniform reported separately | 0 |
| `retrieval_dominance` | best of fuzzy / char-TF-IDF / BM25 nearest-neighbour accuracy over LLM accuracy, with the similarity-correctness point-biserial | low ratio, corr near 0 |
| `semantic_validity_gap` | (EM gain from constrained decoding) minus (semantic gain from constrained decoding) | 0 |
| `abstention_failure` | fraction of undetermined probes answered when an explicit abstain option is offered; risk-coverage AUC | 0 |

Format sensitivity is computed first and acts as a gate: above the threshold, the other
five are marked uninterpretable at the benchmark level.

**Five interventions** (`interventions.py`), each with a manipulation check on the signal it
targets, plus a cross-model invariance check that voids predictions when a filter removes
signal rather than artifact:
leakage-aware resplit, label rebalancing with partial-input filtering, surface re-rendering
(score as the mean over renderings), constraint ablation, and abstention-enabled scoring
with injected OOD probes.

**A pre-registered ledger** (`ledger.py`): predictions with direction, threshold, confidence
and family; canonical SHA-256 stamp; scoring by hit rate, Brier score, reliability table,
per-family hit rate, and a paired sign test against the naive predictor that says every
change goes down. `ledger/paper3_ledger.json` holds the illustrative entries from the proposal.

**SynthModal-Control** (`modal.py`, `synth.py`): a generated Kripke-semantics benchmark
with a real three-valued model checker. Balanced labels, per-item novel names so no lexical
overlap, and principled "insufficient information" probes (items whose truth value is
genuinely undetermined). It is the negative control: a diagnostic that fires here is broken.

**Model adapters** (`models.py`): dummy models that encode the failure modes the instrument
should detect (oracle, position bias, constant answer, nearest-neighbour retrieval, a model
whose only problem is format noise), plus `HFModel` (transformers, choice constraint by
candidate log-prob scoring) and `OpenAICompatModel` (soft constraint, flagged as such).

### CLI

```
artifact-signals synth    --n 400 --ood-frac 0.1 --out control.jsonl
artifact-signals diagnose --benchmark eval.jsonl --train train.jsonl --model hf:Qwen/Qwen2.5-7B-Instruct \
                          --semantic lenient --cache .cache --out profile.json
artifact-signals ledger hash  ledger/paper3_ledger.json --stamp
artifact-signals ledger score ledger/paper3_ledger.json results.json --void P09
```

Model specs: `dummy:oracle:0.8`, `dummy:first`, `dummy:constant:TRUE`, `dummy:retrieval`,
`dummy:noisy:0.6:0.5`, `hf:<model_id>`, `openai:<model>[@base_url]`.

### Layout

```
papers/03-benchmark-artifact-signals/PROPOSAL.md   the proposal
src/artifact_signals/                              the instrument
ledger/paper3_ledger.json                          illustrative pre-registered predictions
examples/synth_pipeline.py                         two-phase dry run
tests/                                             33 tests
```
