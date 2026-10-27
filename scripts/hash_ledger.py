"""Print and check the SHA-256 of the prediction ledgers and the scored ledger output.

  python scripts/hash_ledger.py            # print file hashes and each ledger's canonical hash
  python scripts/hash_ledger.py --check    # also compare file hashes with SHA256SUMS; exit 1 on mismatch

Two hashes per ledger. The file hash is over the exact bytes (what SHA256SUMS records, and what
`sha256sum -c SHA256SUMS` checks). The canonical hash is Ledger.sha256(): sorted-key JSON of title,
created, meta and predictions, the value the scorer prints and a stamp would embed. Ledger files
are never rewritten here.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

FILES = ["ledger/retrospective_papers12.json", "ledger/prospective_template.json", "outputs/paper12/ledger_score.json"]
LEDGERS = FILES[:2]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(root: Path) -> dict[str, str]:
    out = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            out[rel.lstrip("*")] = digest
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    # import this script's own package, never code from the tree being checked
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from artifact_signals.ledger import Ledger

    want = expected(a.root) if a.check else {}
    bad = 0
    for rel in FILES:
        got = file_sha256(a.root / rel)
        status = ""
        if a.check:
            status = "ok" if want.get(rel) == got else "MISMATCH"
            bad += status != "ok"
        print(f"{status + ' ' if status else ''}{rel}  file sha256 {got}")
        if rel in LEDGERS:
            led = Ledger.load(a.root / rel, verify=False)
            print(f"    canonical sha256 {led.sha256()}  status {led.status}")
    if a.check:
        print("all hashes match SHA256SUMS" if not bad else f"{bad} file(s) differ from SHA256SUMS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
