"""Freezing a prospective ledger: readiness check, freeze, and refusal to re-stamp."""

import json

import pytest

from artifact_signals.cli import main
from artifact_signals.ledger import Ledger, Prediction, freeze, freeze_problems

DEV = json.load(open("outputs/paper12/results.json"))


def _ledger(n: int) -> Ledger:
    preds = [Prediction(f"T{k:02d}", "folio", "leakage_resplit", "llm.accuracy.delta", "flat", 0.03, 0.7) for k in range(n)]
    return Ledger("test-set ledger", preds, created="2026-10-27T00:00:00+00:00", meta={"status": "retrospective"})


def test_shipped_template_is_not_freeze_ready():
    problems = freeze_problems(Ledger.load("ledger/prospective_template.json", verify=False), DEV)
    assert any("7 predictions" in p for p in problems)
    for pid in ("P07", "P09", "P10"):
        assert any(p.startswith(pid) for p in problems), pid
    assert not any(p.startswith(pid) for p in problems for pid in ("P08", "P11", "P12", "P13"))


def test_ready_ledger_has_no_problems_and_freezes_to_prospective(tmp_path):
    led = _ledger(40)
    assert freeze_problems(led, DEV) == []
    frozen = freeze(led, DEV)
    assert frozen.status == "prospective" and frozen.meta["frozen_at"]
    path = tmp_path / "led.json"
    h = frozen.save(path)
    assert Ledger.load(path).sha256() == h  # stamped and verifies


def test_freeze_refuses_an_unready_ledger():
    with pytest.raises(ValueError, match="not freeze-ready"):
        freeze(_ledger(7), DEV)


def test_duplicate_ids_are_a_problem():
    led = _ledger(40)
    led.predictions[1].id = led.predictions[0].id
    assert any("duplicate" in p for p in freeze_problems(led, DEV))


def test_cli_check_exit_codes_and_freeze_refuses_restamp(tmp_path, capsys):
    assert main(["ledger", "check", "ledger/prospective_template.json", "outputs/paper12/results.json"]) == 1
    path = tmp_path / "led.json"
    _ledger(40).save(path)  # an already-stamped file
    with pytest.raises(SystemExit):
        main(["ledger", "freeze", str(path), "outputs/paper12/results.json"])
    with pytest.raises(SystemExit):
        main(["ledger", "hash", str(path), "--stamp"])
    unstamped = tmp_path / "fresh.json"
    body = json.loads(path.read_text())
    body.pop("sha256")
    unstamped.write_text(json.dumps(body))
    assert main(["ledger", "check", str(unstamped), "outputs/paper12/results.json"]) == 0
    assert main(["ledger", "freeze", str(unstamped), "outputs/paper12/results.json"]) == 0
    assert Ledger.load(unstamped).status == "prospective"
