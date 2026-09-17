"""Copy the slices of the two source repositories that the offline diagnostics need.

  python scripts/sync_external.py [--cod PATH] [--kripke PATH]

Defaults look for sibling clones at ../th00masml/<repo> or ../<repo>. Kripke raw outputs are
slimmed to the fields the analysis uses (the constraint traces are dropped).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external"

COD_FILES = [
    "data/derived/test.json", "data/derived/dictionary.json", "outputs/test_dictsim.json",
    "outputs/pred_B_majority.json", "outputs/pred_B_fuzzy.json", "outputs/pred_M_local_qwen14b.json",
    "outputs/pred_M_comm_sonnet5.json", "outputs/raw_comm.jsonl", "outputs/raw_local.jsonl",
    "outputs/probe_summary.json", "outputs/summary.json",
]
KRIPKE_KEEP = ("id", "present", "world", "formula", "gold", "model", "arm", "output")


def find(name: str) -> Path | None:
    for cand in (ROOT.parent / "th00masml" / name, ROOT.parent / name):
        if (cand / ".git").exists() or (cand / "README.md").exists():
            return cand
    return None


def sync_cod(src: Path) -> None:
    dst = DEST / "cod"
    dst.mkdir(parents=True, exist_ok=True)
    for rel in COD_FILES:
        shutil.copy(src / rel, dst / Path(rel).name)
    print(f"cod: copied {len(COD_FILES)} files from {src}")


def sync_kripke(src: Path) -> None:
    dst = DEST / "kripke"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "data" / "fixtures.jsonl", dst / "fixtures.jsonl")
    for run in ("gold", "product"):
        rows = [json.loads(l) for l in (src / "outputs" / run / "raw.jsonl").open() if l.strip()]
        with (dst / f"{run}_raw.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in KRIPKE_KEEP}) + "\n")
        shutil.copy(src / "outputs" / run / "run_meta.json", dst / f"{run}_run_meta.json")
        shutil.copy(src / "outputs" / run / "paper_stats.json", dst / f"{run}_paper_stats.json")
        print(f"kripke/{run}: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cod", type=Path, default=find("cause-of-death-coding"))
    ap.add_argument("--kripke", type=Path, default=find("kripke-modal-accessibility"))
    a = ap.parse_args()
    if not a.cod or not a.kripke:
        raise SystemExit("source clones not found; pass --cod and --kripke")
    sync_cod(a.cod)
    sync_kripke(a.kripke)


if __name__ == "__main__":
    main()
