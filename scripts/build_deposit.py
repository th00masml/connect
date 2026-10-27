"""Build the deposit bundle for a frozen prospective ledger, from a git tag (never from the working tree).

  python scripts/build_deposit.py --tag v0.3.0 [--ledger ledger/prospective_template.json] [--out dist]

Refuses unless the ledger at that tag has status `prospective` and a sha256 stamp that verifies.
Writes <out>/deposit-<tag>/ with:
  connect-<tag>.tar.gz   source archive of the tag (git archive, gzip with fixed header: reproducible)
  <ledger file>          the frozen ledger as committed at the tag
  PROPOSAL.md            the protocol as committed at the tag
  MANIFEST.json          tag, commit, ledger status and hashes, file hashes
  SHA256SUMS             sha256 of every file above
Nothing is uploaded. The deposit itself (OSF, Zenodo) is done by hand; see docs/DEPOSIT.md.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from artifact_signals.ledger import Ledger  # noqa: E402


def git(repo: Path, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode().strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ledger", default="ledger/prospective_template.json")
    ap.add_argument("--proposal", default="papers/03-benchmark-artifact-signals/PROPOSAL.md")
    ap.add_argument("--out", type=Path, default=HERE / "dist")
    a = ap.parse_args(argv)

    commit = git(a.repo, "rev-list", "-n", "1", a.tag)
    ledger_bytes = git(a.repo, "show", f"{a.tag}:{a.ledger}", binary=True)
    raw = json.loads(ledger_bytes)
    led = _load(raw)
    if led.status != "prospective":
        print(f"refused: {a.ledger} at {a.tag} has status {led.status}, not prospective; freeze it first")
        return 1
    if raw.get("sha256") != led.sha256():
        print(f"refused: {a.ledger} at {a.tag} has no valid sha256 stamp")
        return 1

    dest = a.out / f"deposit-{a.tag}"
    dest.mkdir(parents=True, exist_ok=True)
    tar = git(a.repo, "archive", "--format=tar", f"--prefix=connect-{a.tag}/", a.tag, binary=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(tar)
    files = {
        f"connect-{a.tag}.tar.gz": buf.getvalue(),
        Path(a.ledger).name: ledger_bytes,
        "PROPOSAL.md": git(a.repo, "show", f"{a.tag}:{a.proposal}", binary=True),
    }
    manifest = {
        "tag": a.tag, "commit": commit,
        "ledger_path": a.ledger, "ledger_status": led.status, "ledger_frozen_at": led.meta.get("frozen_at"),
        "ledger_n_predictions": len(raw["predictions"]),
        "ledger_canonical_sha256": led.sha256(),
        "files_sha256": {name: sha256(data) for name, data in files.items()},
    }
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    for name, data in files.items():
        (dest / name).write_bytes(data)
    (dest / "SHA256SUMS").write_text("".join(f"{sha256(d)}  {n}\n" for n, d in sorted(files.items())))
    print(f"wrote {dest}")
    print((dest / "SHA256SUMS").read_text(), end="")
    return 0


def _load(raw: dict) -> Ledger:
    from artifact_signals.ledger import Prediction

    return Ledger(raw["title"], [Prediction(**p) for p in raw["predictions"]], raw.get("created", ""), raw.get("meta", {}))


if __name__ == "__main__":
    sys.exit(main())
