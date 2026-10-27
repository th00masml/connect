import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["ledger/retrospective_papers12.json", "ledger/prospective_template.json", "outputs/paper12/ledger_score.json"]


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "hash_ledger.py"), "--root", str(root), *args],
                          capture_output=True, text=True)


def copy_tree(tmp: Path) -> Path:
    for rel in [*FILES, "SHA256SUMS"]:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp / rel)
    return tmp


def test_committed_ledger_hashes_check_out():
    r = run(ROOT, "--check")
    assert r.returncode == 0, r.stdout + r.stderr
    for rel in FILES:
        assert rel in r.stdout
    assert "canonical" in r.stdout


def test_one_changed_byte_fails_the_check(tmp_path):
    root = copy_tree(tmp_path)
    p = root / "ledger/retrospective_papers12.json"
    p.write_text(p.read_text().replace("0.7", "0.6", 1))
    r = run(root, "--check")
    assert r.returncode == 1 and "MISMATCH ledger/retrospective_papers12.json" in r.stdout
