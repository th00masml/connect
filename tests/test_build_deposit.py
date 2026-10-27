import json
import subprocess
import sys
from pathlib import Path

from artifact_signals.ledger import Ledger, Prediction

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_deposit.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def make_repo(tmp: Path, status: str, tamper: bool = False) -> Path:
    repo = tmp / "repo"
    (repo / "ledger").mkdir(parents=True)
    (repo / "papers").mkdir()
    (repo / "papers" / "PROPOSAL.md").write_text("# protocol\n")
    led = Ledger("t", [Prediction("T1", "b", "i", "x", "flat", 0.1, 0.7)], created="2026-10-27T00:00:00+00:00",
                 meta={"status": status})
    led.save(repo / "ledger" / "prospective.json")
    if tamper:  # edit a prediction after stamping
        p = repo / "ledger" / "prospective.json"
        p.write_text(p.read_text().replace('"threshold": 0.1', '"threshold": 0.2'))
    git(repo, "init", "-q")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "x",
        "--date", "2026-10-27T00:00:00+00:00")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "tag", "-a", "v9", "-m", "v9")
    return repo


def build(repo: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), "--tag", "v9", "--ledger", "ledger/prospective.json",
                           "--proposal", "papers/PROPOSAL.md", "--out", str(out)], capture_output=True, text=True)


def test_bundle_is_built_from_the_tag_and_is_reproducible(tmp_path):
    repo = make_repo(tmp_path, "prospective")
    a, b = build(repo, tmp_path / "a"), build(repo, tmp_path / "b")
    assert a.returncode == 0 and b.returncode == 0, a.stdout + a.stderr + b.stdout + b.stderr
    sums_a = (tmp_path / "a" / "deposit-v9" / "SHA256SUMS").read_text()
    assert sums_a == (tmp_path / "b" / "deposit-v9" / "SHA256SUMS").read_text()
    manifest = json.loads((tmp_path / "a" / "deposit-v9" / "MANIFEST.json").read_text())
    assert manifest["tag"] == "v9" and manifest["commit"] == git(repo, "rev-list", "-n", "1", "v9")
    assert manifest["ledger_status"] == "prospective"
    assert manifest["ledger_canonical_sha256"] == Ledger.load(repo / "ledger" / "prospective.json").sha256()
    for name in ("connect-v9.tar.gz", "prospective.json", "PROPOSAL.md"):
        assert name in sums_a


def test_bundle_refuses_a_ledger_that_is_not_frozen(tmp_path):
    repo = make_repo(tmp_path, "retrospective")
    r = build(repo, tmp_path / "out")
    assert r.returncode == 1 and "not prospective" in r.stdout + r.stderr
    assert not (tmp_path / "out" / "deposit-v9").exists()


def test_bundle_refuses_a_ledger_edited_after_stamping(tmp_path):
    repo = make_repo(tmp_path, "prospective", tamper=True)
    r = build(repo, tmp_path / "out")
    assert r.returncode == 1 and "no valid sha256 stamp" in r.stdout
