#!/usr/bin/env bash
# Regenerate every CPU-reproducible claim and compare it with what is committed.
# Exits non-zero on the first mismatch. No GPU, no network, no API key.
#   scripts/verify.sh            (uses `python` on PATH; set PYTHON=... to override)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

step() { printf '\n== %s\n' "$*"; }

step "1/6 test suite"
"$PY" -m pytest -q

step "2/6 offline Phase 1 on papers 1 and 2: regenerate and byte-compare with outputs/paper12"
"$PY" -m artifact_signals.cli paper12 --out "$TMP/paper12" > /dev/null
for f in RESULTS_offline.md profiles.json results.json ledger_score.json; do
  cmp "outputs/paper12/$f" "$TMP/paper12/$f"
  echo "identical: outputs/paper12/$f"
done

step "3/6 COD v2 Injury-folded scoring"
"$PY" -m artifact_signals.cli paper12 --fold-injury --out "$TMP/folded" > /dev/null

step "4/6 reproduction tables: published == recomputed on every row (both scorings)"
"$PY" - "$TMP/paper12/RESULTS_offline.md" "$TMP/folded/RESULTS_offline.md" <<'PYEOF'
import sys
bad = 0
for path in sys.argv[1:]:
    text = open(path, encoding="utf-8").read()
    table = text.split("## Reproduction check")[1].split("\n## ")[0]
    rows = [l for l in table.splitlines() if l.startswith("| ") and not l.startswith("| quantity")]
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        ok = cells[1] == cells[2]
        bad += not ok
        if not ok:
            print(f"MISMATCH {path}: {row}")
    print(f"{len(rows)} rows checked in {path.split('/')[-2]}")
sys.exit(1 if bad else 0)
PYEOF

step "5/6 ledger hashes"
"$PY" scripts/hash_ledger.py --check

step "6/6 SynthModal-Control is deterministic for a fixed seed"
"$PY" -m artifact_signals.cli synth --n 400 --seed 0 --ood-frac 0.1 --out "$TMP/a.jsonl" > /dev/null
"$PY" -m artifact_signals.cli synth --n 400 --seed 0 --ood-frac 0.1 --out "$TMP/b.jsonl" > /dev/null
cmp "$TMP/a.jsonl" "$TMP/b.jsonl"
echo "identical: synth --n 400 --seed 0 --ood-frac 0.1 (sha256 $("$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$TMP/a.jsonl"))"

printf '\nverify: all checks passed\n'
