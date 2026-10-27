# Contributing

## Development install

Python 3.10 or newer. The package has no runtime dependencies; `[dev]` adds pytest and ruff.

```
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Checks

```
pytest                           # full suite, CPU only, no network, under two minutes
ruff check .                     # lint
scripts/verify.sh                # regenerate every CPU-reproducible claim and compare with the committed outputs
```

CI runs the same three on Python 3.10 to 3.13. The `hf`, `api` and `data` extras (GPU models,
API models, Hugging Face downloads) are not exercised in CI.

## Ground rules

- Keep `dependencies = []`. New capability goes in an optional extra or under `scripts/`.
- If a change moves a number in `outputs/paper12/`, regenerate with
  `artifact-signals paper12 --out outputs/paper12` and commit the outputs with the code.
- Never edit a file under `ledger/` by hand once stamped. The one sanctioned change is the freeze
  (`artifact-signals ledger freeze`, see `docs/DEPOSIT.md`). Hashes are checked by `scripts/hash_ledger.py --check`.
- A wrong claim in the README, the proposal or a results file is corrected in place: keep the
  original figure, give the corrected one and a one-line reason.
