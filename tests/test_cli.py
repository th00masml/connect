import json

from artifact_signals.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    tr, te, prof, led_out = tmp_path / "tr.jsonl", tmp_path / "te.jsonl", tmp_path / "p.json", tmp_path / "s.json"
    assert main(["synth", "--n", "30", "--seed", "1", "--out", str(tr)]) == 0
    assert main(["synth", "--n", "30", "--seed", "2", "--ood-frac", "0.2", "--out", str(te)]) == 0
    assert main(["diagnose", "--benchmark", str(te), "--train", str(tr), "--model", "dummy:oracle:0.8",
                 "--semantic", "lenient", "--n-boot", "20", "--out", str(prof)]) == 0
    d = json.load(open(prof))
    assert set(d["vector"]) == {"format_sensitivity", "partial_input_accuracy", "label_prior_skew",
                                "retrieval_dominance", "semantic_validity_gap", "abstention_failure"}
    assert d["vector"]["format_sensitivity"] == 0.0
    assert main(["ledger", "hash", "ledger/paper3_ledger.json"]) == 0
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"synthmodal_control": {"leakage_resplit": {"llm": {"accuracy": {"delta": 0.01}}}}}))
    assert main(["ledger", "score", "ledger/paper3_ledger.json", str(res), "--out", str(led_out)]) == 0
    s = json.load(open(led_out))
    assert s["n_scored"] == 1 and s["hits"] == 1
    out = capsys.readouterr().out
    assert "Artifact Risk Profile" in out and "[HIT ] P11" in out
