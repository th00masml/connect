# Changelog

## 0.2.0 (unreleased)

### Corrections to earlier claims (originals kept in place in README and PROPOSAL)

- Memorisation probe: "0.00" was an exact-code rate over zero answered items. 30 of 30 probe
  calls returned empty; the probe is INCONCLUSIVE (`data/external/cod/probe_summary.json`).
- Kripke format gate: the offline figure (0.130 to 0.142) is the SD over three different prompts,
  not the five meaning-preserving renderings the gate is defined on. It is now reported as prompt
  variance; the gate is not evaluated offline and the other Kripke signals are no longer marked gated.
- Cause-of-death SVG: strict and lenient parsing agree on 300/300 Sonnet outputs (0.000) and
  299/300 Qwen outputs (0.003), not on all outputs of both models.
- Leak-free Kripke SVG of 0.39 is the 1.5B value; 0.5B is 0.11.
- `partial_input_accuracy` offline is a majority-class proxy (0.108 cause-of-death, 0.050 Kripke),
  now marked `(proxy)` in the report and the README.
- P01 to P05 restate the two source papers' own tables; the proposal says so.

### Added

- `scoring_key_validity` (D-006) and `docs/signals.md` with estimators and clean values for
  D-001 to D-006.
- COD v2 Injury-fold scoring: `fold_injury` in the cause-of-death adapter and
  `artifact-signals paper12 --fold-injury`; the report reproduces the v2 folded headline.
- `artifact-signals fetch` for FOLIO and LogiQA, from the hub or a local rows file.
- `cached:<model name>` model spec: score saved outputs (for example a rented 70B run) without
  loading weights.
- `scripts/hash_ledger.py` and `SHA256SUMS` for the ledgers and the scored output.
- `scripts/verify.sh`, CI on Python 3.10 to 3.13, ruff config, `LICENSE` (MIT), `DATA_LICENSE.md`,
  `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`, `ARTIFACT.md`.
- Vendored `probe_results.jsonl` and `paper_stats.json` from the paper 1 companion repository.

### Changed

- Cluster bootstrap defaults to 10,000 resamples everywhere (was 500 / 1,000 / 300), as registered
  in PROPOSAL G.5.
- Ledger scoring reports a per-family binomial p and the Holm-adjusted p across families.
- `outputs/paper12/*.json` floats are rounded to 12 decimals so outputs are byte-identical on
  Python 3.10 to 3.13.
- Links to the two private companion repositories replaced by role names.

## 0.1.0

Initial instrument, offline Phase 1 on papers 1 and 2, retrospective ledger.
