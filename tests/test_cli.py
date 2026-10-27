import json

from artifact_signals.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    tr, te, prof, led_out = tmp_path / "tr.jsonl", tmp_path / "te.jsonl", tmp_path / "p.json", tmp_path / "s.json"
    assert main(["synth", "--n", "30", "--seed", "1", "--out", str(tr)]) == 0
    assert main(["synth", "--n", "30", "--seed", "2", "--ood-frac", "0.2", "--out", str(te)]) == 0
    assert main(["diagnose", "--benchmark", str(te), "--train", str(tr), "--model", "dummy:oracle:0.8",
                 "--semantic", "lenient", "--n-boot", "20", "--out", str(prof)]) == 0
    d = json.load(open(prof))
    assert set(d["vector"]) == {"format_sensitivity", "partial_input_accuracy",
                                "retrieval_dominance", "semantic_validity_gap", "abstention_failure"}
    assert d["vector"]["format_sensitivity"] == 0.0
    assert main(["ledger", "hash", "ledger/retrospective_papers12.json"]) == 0
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"cause_of_death": {"leakage_resplit": {"gap": {"llm_minus_retrieval": {"delta": 0.4}}}}}))
    assert main(["ledger", "score", "ledger/retrospective_papers12.json", str(res), "--out", str(led_out)]) == 0
    s = json.load(open(led_out))
    assert s["n_scored"] == 1 and s["hits"] == 1
    out = capsys.readouterr().out
    assert "Artifact Risk Profile" in out and "[dataset]" in out and "[RETROSPECTIVE]" in out


def test_cli_paper12_fold_injury_reproduces_cod_v2(tmp_path):
    out = tmp_path / "folded"
    assert main(["paper12", "--fold-injury", "--out", str(out)]) == 0
    report = (out / "RESULTS_offline.md").read_text()
    assert "| COD claude-sonnet-5 accuracy (folded) | 0.7133 | 0.7133 |" in report
    assert json.loads((out / "results.json").read_text())["cause_of_death"]["scoring"] == "injury_folded"


def test_cli_fetch_folio_from_local_rows(tmp_path, capsys):
    from artifact_signals.data import Benchmark
    out = tmp_path / "folio.jsonl"
    assert main(["fetch", "--source", "folio", "--rows", "tests/fixtures/folio_rows_5.jsonl", "--out", str(out)]) == 0
    b = Benchmark.from_jsonl(out)
    assert [i.label for i in b] == ["True", "False", "Uncertain", "False"]  # "Unknown" row skipped
    assert b[0].premises == ["All glimbs are snarfs.", "Tova is a glimb."] and b[0].options == ["True", "False", "Uncertain"]
    assert b[0].source_id == b[1].source_id == "901" and b.task == "choice"
    out_text = capsys.readouterr().out
    assert "wrote 4 items" in out_text and "skipped 1" in out_text


def test_cli_fetch_logiqa_from_local_rows(tmp_path):
    from artifact_signals.data import Benchmark
    out = tmp_path / "logiqa.jsonl"
    assert main(["fetch", "--source", "logiqa", "--rows", "tests/fixtures/logiqa_rows_5.jsonl", "--out", str(out)]) == 0
    b = Benchmark.from_jsonl(out)
    assert len(b) == 5 and [i.label for i in b] == ["Kel is green.", "The lamp is off.", "No quibs are lums.", "Cy",
                                                    "The guard is present."]
    assert all(len(i.options) == 4 and i.label in i.options for i in b)
