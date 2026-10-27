# Artifact

`connect` v0.2.0: the `artifact-signals` instrument for paper 3 of the programme *Reliable
Evaluation of Reasoning in Language Models*, with its offline Phase 1 on papers 1 and 2.

## Contents

| Path | What it is |
|---|---|
| `src/artifact_signals/` | the instrument: six diagnostics, five interventions, ledger, SynthModal-Control, model adapters |
| `src/artifact_signals/benchmarks/` | adapters for paper 1 (cause-of-death coding) and paper 2 (Kripke modal accessibility) |
| `data/external/` | vendored slices of the two companion repositories (not yet public); provenance in its README |
| `outputs/paper12/` | committed expected outputs of the offline Phase 1 run |
| `ledger/` | retrospective (development-set) ledger and the unfrozen prospective template |
| `SHA256SUMS` | file hashes of the two ledgers and the scored output |
| `docs/signals.md` | estimator and clean value for each signal, D-001 to D-006 |
| `papers/03-benchmark-artifact-signals/PROPOSAL.md` | the proposal |
| `scripts/verify.sh` | the single entry point that regenerates every CPU-reproducible claim |
| `tests/` | 64 tests, CPU only, no network |

## One command

```
pip install -e ".[dev]"
scripts/verify.sh
```

Exits 0 only if: the test suite passes; the four files in `outputs/paper12/` regenerate
byte-identically; every row of the reproduction table has published == recomputed under both the
preregistered and the COD v2 folded scoring; the ledger hashes match `SHA256SUMS`; and
SynthModal-Control is deterministic for a fixed seed. A one-byte change to
`outputs/paper12/results.json` makes it exit 1. CPU only, no network, no API key, about 1.5 to 2
minutes. Outputs are byte-identical on Python 3.10, 3.11, 3.12 and 3.13.

## Claims

"Ran" means run in the session that prepared v0.2.0 (2026-10-27), on CPU, Python 3.13.
Every number below is copied from the named output field; values shown to two or three
decimals are rounded from those fields.

| # | Claim | Command | Hardware | Runtime | Expected output | Ran |
|---|---|---|---|---|---|---|
| 1 | The adapters reproduce papers 1 and 2's published numbers to the item: 12 rows (COD fuzzy 0.7067, Sonnet 0.6667, HARD n 33, fuzzy HARD 0.3636, Sonnet HARD 0.7576, pairwise 14/1; Kripke strict EM 26/51 and 8/52, forced 25, product 36 and 37) | `artifact-signals paper12 --out outputs/paper12` | CPU | ~20 s | Reproduction table in `outputs/paper12/RESULTS_offline.md`; published == recomputed on every row | yes |
| 2 | COD v2 Injury-folded headline reproduced: Sonnet 0.7133, Qwen 0.6567, both 0.7576 on HARD; fuzzy-only / model-only 52/54 and 60/45 | same, rows "COD v2 folded"; full run with `--fold-injury` | CPU | ~20 s | published == recomputed on 6 rows | yes |
| 3 | Kripke prompt variance (SD over 3 prompts) 0.130 to 0.142; the format gate is not evaluated offline | `paper12` | CPU | ~20 s | `profiles.json`: `signals.format_sensitivity.details.prompt_variance.value` = 0.137, 0.142, 0.130, 0.135; `gate_evaluated` false | yes |
| 4 | Nobody abstains: 0 empty outputs among the 1,600 per Kripke run (800 on out-of-language fixtures) | `paper12` | CPU | ~20 s | `abstention_failure` 1.000 in every arm of both models | yes |
| 5 | Constraint leak measurable before any model: forced_rate 0.312 (25/80) gold, 0 product; 55/55 unaffected outputs byte-identical | `paper12` | CPU | ~20 s | `results.json`: `kripke@*.constraint_ablation.per_run.*.forced_rate`, `leak_removal.free_subset.outputs_identical` | yes |
| 6 | Leak-free SVG 0.3875 (1.5B), 0.1125 (0.5B), exact field values; 1.5B EM gain +0.400 vs truth-value gain +0.013 | `paper12` | CPU | ~20 s | `profiles.json`: `kripke@product@*.signals.semantic_validity_gap` | yes |
| 7 | Cause-of-death SVG 0.000 (Sonnet), 0.003 (Qwen, 1 of 300 outputs differs) | `paper12` | CPU | ~20 s | `profiles.json`: `cause_of_death@*.signals.semantic_validity_gap.value` | yes |
| 8 | Leakage sweep, similarity cap 0.8 to 0.4: retrieval 0.639 to 0.188, LLM 0.661 to 0.875 | `paper12` | CPU | ~20 s | `results.json`: `cause_of_death.leakage_resplit.sweep.{0.8,0.4}.after` | yes |
| 9 | Retrospective ledger: 5 of 7 scored, 4 hits (P01 misses), hit rate 0.800, Brier 0.140; P06 and P14 not scorable offline. Retrospective by construction: P01 to P05 restate the papers' own tables | `paper12` | CPU | ~20 s | `ledger_score.json`; canonical ledger hash `b2290d6a0187…` | yes |
| 10 | Memorisation probe inconclusive: 30 of 30 calls returned empty | `paper12` | CPU | ~20 s | `results.json`: `cause_of_death.memorization_probe.answered` = 0 | yes |
| 11 | Partial-input accuracy offline is a proxy: 0.108 (COD), 0.050 (Kripke) | `paper12` | CPU | ~20 s | rows marked `(proxy)` | yes |
| 12 | D-006 on cause-of-death: label-space asymmetry 2/22 = 0.091 (Injury, Special); Injury predictions Sonnet 15, Qwen 31, dictionary 0; rule null rate 0 | `paper12` | CPU | ~20 s | `profiles.json`: `cause_of_death@*.signals.scoring_key_validity.details` | yes |
| 13 | D-006 is 0 on SynthModal-Control against the model checker, and counts injected label flips exactly | `pytest tests/test_scoring_key.py` | CPU | ~6 s | 7 passed | yes |
| 14 | SynthModal-Control is deterministic for a fixed seed | `scripts/verify.sh`, step 6 | CPU | <1 s | `synth --n 400 --seed 0 --ood-frac 0.1` sha256 `aeb68a806a3d188393f66394c6cc03c3f00153b9499b548b9b641ec96b9e1a3d` | yes |

### Not reproducible from this repository today

| Claim or path | Needs | Status |
|---|---|---|
| Live diagnostics: format sensitivity over five renderings, model-run partial-input accuracy, abstention on cause-of-death | a model: `hf:` (GPU), `openai:` (API key), or `cached:` with saved outputs | code path present and tested with dummy models; never run on a real model |
| A real small model (Qwen2.5-0.5B) through SynthModal-Control | Hugging Face download | not run: huggingface.co is not reachable from the environment that prepared v0.2.0 |
| FOLIO and LogiQA profiles | `pip install -e ".[data]"` and a download | converters tested on synthetic fixtures only; field layouts not checked against the live datasets |
| The prospective ledger result | the freeze procedure in `ledger/prospective_template.json` meta, then the live runs | does not exist yet |
| `examples/synth_pipeline.py` ledger hash | none | the dry-run ledger embeds its wall-clock creation time, so its hash differs on every run |

## Ledger and protocol hashes

| File | File sha256 | Canonical (`Ledger.sha256()`) | Status field |
|---|---|---|---|
| `ledger/retrospective_papers12.json` | `b33a7641dfc3d95958fed7b1c159380c1e16e87b8194411197add75f1f16a204` | `b2290d6a0187ba9500635516f8f9347977647cc474a99f4dd609015d489e9c76` | retrospective |
| `ledger/prospective_template.json` | `7172b7e1b58941f4805859c92a07ad3a31f9563fe0f56042b3a27bffc9a45e36` | `7ad157f0d905ab4020ac97371e5ac62f7d3e686c6f64b632c458dcf5bee4423a` | retrospective (unfrozen) |
| `outputs/paper12/ledger_score.json` | `a5b1fadf7d79f9e3c5f56eb3160f7929a911483197d462d2840f9df547aba43e` | n/a | generated |
| `papers/03-benchmark-artifact-signals/PROPOSAL.md` | `73dd0e3d2aa421248e6e09b0c6b2739d284f18f96e7ce0ecf7894b8c4a5c5b65` | n/a | proposal, edited in v0.2.0 (corrections) |

Check with `python scripts/hash_ledger.py --check` or `sha256sum -c SHA256SUMS`. The ledger files
are byte-identical to the v0.1.0 commit 8e14c04.

## Badges (ACM artifact review terms), honestly

| Badge | Earned today? | Why |
|---|---|---|
| Available | no | The repository is public, but there is no archival deposit with a DOI yet. `.zenodo.json` is prepared; the deposit is the author's decision. |
| Functional | yes, for the offline path | Documented, complete for the claims above, and exercised by one command (`scripts/verify.sh`) that fails on a planted mismatch. The live path is not exercised on a real model. |
| Reusable | no | The instrument is packaged with no runtime dependencies and has a CLI, but the real-model adapters (`hf:`, `openai:`) and the public-benchmark loaders have not been run against real models or live datasets. |
