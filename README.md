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
pytest                         # 57 tests, about 1 min on CPU, no GPU, no network (was 38 tests, ~30 s in v0.1.0)
scripts/verify.sh              # regenerate every CPU-reproducible claim and compare; see ARTIFACT.md
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

A sixth signal, `scoring_key_validity` (D-006, added in v0.2), checks the scoring key itself:
disagreement with an independent re-scoring, the rule null rate, and label-space asymmetry
between compared systems. It is not part of the retrospective ledger. Estimators and clean values
for all six (D-001 to D-006) are in [`docs/signals.md`](docs/signals.md).

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

Headline numbers from that run. Kripke's format-sensitivity gate is not evaluated offline: the
cached run has three different prompts (naive / prompted / prompt_tuned), not the five
meaning-preserving renderings the gate is defined on, so their accuracy SD (0.130 to 0.142 across
the four model x run profiles) is a prompt-variance figure, not the gate statistic.
*Correction (v0.2):* this sentence originally read "Kripke fails the format gate (SD 0.13 over
prompt arms vs the 0.03 gate)" and the report marked the other Kripke signals GATED; the
statistic was the wrong one, so the gate decision is withdrawn until a live five-rendering run.
Nobody abstains in 1,600 outputs per run; Kripke's constraint leak shows up as
`forced_rate = 0.31` before any model runs, and the leak-free replication keeps SVG at 0.39 for
the 1.5B model (0.11 for 0.5B; the sentence previously gave 0.39 without naming the model).
Cause-of-death has essentially no format artifact (SVG 0.000 for claude-sonnet-5, 0.003 for
qwen2.5:14b, where strict and lenient parsing differ on 1 of 300 outputs; previously "SVG = 0"),
and its lexical artifact inflates the baseline
rather than the model: under the leakage sweep retrieval falls 0.64 to 0.19 while the LLM rises
0.66 to 0.88. The cause-of-death memorisation probe is inconclusive, not zero: 30 of 30 probe
calls returned empty, so its exact-code rate of 0.00 is over zero answered items
([`data/external/cod/probe_summary.json`](data/external/cod/probe_summary.json)).
The retrospective ledger scores 4 of 5; the miss (P01) predicted the LLM's own score would drop.
P01 to P05 restate the two papers' own published tables (PROPOSAL, Section I), so those hits are
retrospective by construction.
Papers 1 and 2 are the development set, so that score demonstrates the protocol and validates
nothing; the prospective ledger for FOLIO, LogiQA and the synthetic control is the paper's result.

Offline, `partial_input_accuracy` is a majority-class proxy, not a model run with the input
withheld: cause-of-death reports the majority baseline's accuracy minus 1/22 chance (0.108), and
Kripke reports the majority truth value's share minus 0.5 (0.050). The report marks these rows
`(proxy)`. The model-run estimator in the table above needs a live run.

### Ledger hashes

`ledger/` files are never edited. Their SHA-256, and that of the scored output, are recorded in
[`SHA256SUMS`](SHA256SUMS); `python scripts/hash_ledger.py --check` (or `sha256sum -c SHA256SUMS`)
recomputes them and exits non-zero on any mismatch. Canonical hash = `Ledger.sha256()`, the value
the scorer prints.

| file | file sha256 | canonical sha256 |
|---|---|---|
| `ledger/retrospective_papers12.json` | `b33a7641dfc3d95958fed7b1c159380c1e16e87b8194411197add75f1f16a204` | `b2290d6a0187ba9500635516f8f9347977647cc474a99f4dd609015d489e9c76` |
| `ledger/prospective_template.json` | `7172b7e1b58941f4805859c92a07ad3a31f9563fe0f56042b3a27bffc9a45e36` | `7ad157f0d905ab4020ac97371e5ac62f7d3e686c6f64b632c458dcf5bee4423a` |
| `outputs/paper12/ledger_score.json` | `a5b1fadf7d79f9e3c5f56eb3160f7929a911483197d462d2840f9df547aba43e` | n/a (generated) |

Both ledgers have status `retrospective`; the prospective template is unfrozen and unstamped.
`ledger_score.json` is regenerated by `artifact-signals paper12`, so its hash changes whenever
the scorer's output does.

### CLI

```
artifact-signals synth    --n 400 --ood-frac 0.1 --out control.jsonl
artifact-signals fetch    --source folio --split validation --out folio.jsonl   # needs [data]; or --rows FILE, no download
artifact-signals diagnose --benchmark eval.jsonl --train train.jsonl --model hf:Qwen/Qwen2.5-7B-Instruct \
                          --semantic lenient --cache .cache --out profile.json
artifact-signals ledger hash  ledger/prospective_template.json             # print the canonical hash
# artifact-signals ledger hash ledger/prospective_template.json --stamp    # freeze step only: rewrites the file
artifact-signals ledger score ledger/retrospective_papers12.json results.json --void P09
```

Model specs: `dummy:oracle:0.8`, `dummy:first`, `dummy:constant:TRUE`, `dummy:retrieval`,
`dummy:noisy:0.6:0.5`, `hf:<model_id>`, `openai:<model>[@base_url]`, `cached:<model name>`.

`HFModel` loads full, unquantised weights on a single device; there is no 4-bit or multi-GPU
path. 70B-class runs are therefore made on rented hardware with `--cache DIR`, the cache is copied
back, and they are scored here from the saved outputs with `--model cached:hf:<model_id> --cache DIR`,
which replays the cache, never loads weights, and raises on any prompt it has no output for.

### Layout

```
papers/03-benchmark-artifact-signals/PROPOSAL.md   the proposal
docs/signals.md                                    estimators and clean values, D-001 to D-006
src/artifact_signals/                              the instrument
src/artifact_signals/benchmarks/{cod,kripke}.py    adapters for papers 1 and 2 (reproduce, profile, intervene)
data/external/                                     vendored slices of both source repos (scripts/sync_external.py)
outputs/paper12/                                   offline Phase 1 on papers 1 and 2
ledger/retrospective_papers12.json                 development-set ledger (protocol demonstration)
ledger/prospective_template.json                   test-set ledger, unfrozen; freeze procedure in its meta
examples/synth_pipeline.py                         two-phase dry run on the synthetic control
tests/                                             57 tests
scripts/verify.sh, scripts/hash_ledger.py          one-command verification; ledger hashes (SHA256SUMS)
ARTIFACT.md                                        every claim with the command that regenerates it
```
