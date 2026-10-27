# Freezing the prospective ledger and depositing it

The prospective ledger is the paper's result only if its hash was deposited before any model ran
on a test-set benchmark (PROPOSAL G.3, `ledger/README.md`). Every step below is done by the author.
The tooling checks and builds; nothing here uploads anything.

## 0. Before freezing

| item | state on 2026-10-27 |
|---|---|
| 40-60 atomic predictions in `ledger/prospective_template.json` | 7 (P07-P13) |
| every prediction's quantity exists in a development-set `results.json` | P07, P09, P10 do not |
| no model has run on FOLIO, LogiQA or SynthModal-Control evaluation items | author to attest |
| `rebalance_and_filter` on the control: with or without the partial-input filter (decides what P12 means) | open; `examples/synth_pipeline.py` uses no filter |
| evaluation split held in an encrypted archive whose key the tagged commit releases (PROPOSAL G.3) | not implemented |

`artifact-signals ledger check ledger/prospective_template.json outputs/paper12/results.json`
lists what is missing and exits 1 until the first two rows are satisfied. The last three rows are
decisions and attestations no tool can make.

## 1. Write and check

Edit `ledger/prospective_template.json`: set `title` (drop "TEMPLATE, unfrozen"), write the 40-60
predictions (id, benchmark, intervention, quantity, direction, threshold, confidence, family, note),
keep `meta.status` = `retrospective`. Then:

```
artifact-signals ledger check ledger/prospective_template.json outputs/paper12/results.json
```

## 2. Freeze, re-hash, commit, tag

```
artifact-signals ledger freeze ledger/prospective_template.json outputs/paper12/results.json
sha256sum ledger/retrospective_papers12.json ledger/prospective_template.json outputs/paper12/ledger_score.json > SHA256SUMS
python scripts/hash_ledger.py --check
scripts/verify.sh
git add ledger/prospective_template.json SHA256SUMS
git commit -m "Freeze the prospective ledger"
git tag -a v0.2.0 -m "artifact-signals 0.2.0; prospective ledger frozen"
git push origin HEAD v0.2.0
```

`freeze` sets `meta.status` to `prospective`, records `meta.frozen_at`, stamps the sha256 into the
file, and refuses a file that is already stamped. Update the hash tables in `README.md` and
`ARTIFACT.md` in a follow-up commit; the tag is what gets deposited.

## 3. Build the bundle

```
python scripts/build_deposit.py --tag v0.2.0
```

Writes `dist/deposit-v0.2.0/` (gitignored) from the tag, not the working tree:
`connect-v0.2.0.tar.gz` (reproducible source archive), the frozen ledger, `PROPOSAL.md`,
`MANIFEST.json` (tag, commit, ledger status, canonical hash, file hashes) and `SHA256SUMS`. It refuses
if the ledger at the tag is not `prospective` or its stamp does not verify.

## 4. OSF registration (the pre-registration)

Create a registration on osf.io and upload every file in `dist/deposit-v0.2.0/`. Suggested fields:

| field | value |
|---|---|
| title | Predicting Benchmark Failure Before Benchmark Repair: prospective prediction ledger |
| description | the ledger's canonical sha256 and the tag's commit, copied from `MANIFEST.json` |
| contributors | [author name: to confirm] |
| licence | CC BY 4.0 for the ledger and protocol; the code archive is MIT |
| files | the five files of the bundle |

Record the registration's DOI and date. Registration dates are what makes the ledger prospective.

## 5. Zenodo (the software archive)

Enable `th00masml/connect` in Zenodo's GitHub integration, then create a GitHub release from tag
`v0.2.0`; Zenodo archives it with the metadata in `.zenodo.json` and mints a DOI. Then add
`doi:` to `CITATION.cff` and the DOI to the README in a new commit.

## 6. Only then

Run models on the test-set benchmarks, score with
`artifact-signals ledger score ledger/prospective_template.json <test results.json>`, and report the
score against the deposited hash.
