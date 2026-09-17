# Vendored slices of the two source repositories

Regenerate with `python scripts/sync_external.py` from sibling clones.

## `cod/`  from https://github.com/th00masml/cause-of-death-coding

`test.json` (300 held-out strings), `dictionary.json` (3,006 lookup strings), the cached
predictions of the four methods, `test_dictsim.json` (best char-3gram cosine to the
dictionary per test item), the raw model outputs, and the published summary.
Underlying data: ICD10h historic strings (Reid, Garrett, Hiltunen Maltesdotter, Cambridge,
2024), CC-BY 4.0. Only the derived chapter-level slice is vendored.

## `kripke/`  from https://github.com/th00masml/kripke-modal-accessibility

`fixtures.jsonl` (160 fixtures: 80 in-language, 80 out-of-language), the raw outputs of the
`gold` run (admissible set = gold judgements, leaks 25/80) and the `product` replication
(admissible set = world x formula x truth, leak-free), slimmed to the fields the analysis
reads, plus each run's `run_meta.json` and the published `paper_stats.json`.
