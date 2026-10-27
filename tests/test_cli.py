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


def test_cli_diagnose_replays_saved_outputs_without_the_model(tmp_path):
    # The 70B path: generate on rented hardware with --cache, copy the cache back, score here.
    tr, te, cache = tmp_path / "tr.jsonl", tmp_path / "te.jsonl", tmp_path / "cache"
    main(["synth", "--n", "30", "--seed", "1", "--out", str(tr)])
    main(["synth", "--n", "30", "--seed", "2", "--ood-frac", "0.2", "--out", str(te)])
    live, replay = tmp_path / "live.json", tmp_path / "replay.json"
    args = ["--benchmark", str(te), "--train", str(tr), "--n-boot", "20", "--cache", str(cache)]
    assert main(["diagnose", "--model", "dummy:oracle:0.8", *args, "--out", str(live)]) == 0
    name = json.load(open(live))["model"]
    assert main(["diagnose", "--model", f"cached:{name}", *args, "--out", str(replay)]) == 0
    assert json.load(open(replay))["vector"] == json.load(open(live))["vector"]


def test_cached_model_refuses_to_invent_missing_outputs(tmp_path):
    import pytest

    from artifact_signals.models import from_spec
    from artifact_signals.render import baseline_rendering
    from artifact_signals.synth import generate
    item = generate(2, seed=0)[0]
    with pytest.raises(LookupError):
        from_spec("cached:hf:some/70B-model").generate([baseline_rendering()(item)])


def test_cli_paper12_fold_injury_never_writes_into_the_committed_outputs(monkeypatch):
    import artifact_signals.paper12 as p12
    seen = {}
    monkeypatch.setattr(p12, "run_all", lambda out, fold_injury=False: seen.update(out=str(out), fold=fold_injury) or {"report": ""})
    main(["paper12", "--fold-injury"])
    assert seen == {"out": "outputs/paper12_folded", "fold": True}
    main(["paper12"])
    assert seen == {"out": "outputs/paper12", "fold": False}


def test_cli_fetch_limit_zero_means_no_limit(tmp_path, capsys):
    assert main(["fetch", "--source", "folio", "--rows", "tests/fixtures/folio_rows_5.jsonl", "--limit", "0",
                 "--out", str(tmp_path / "f.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "wrote 4 items" in out and "skipped 1 " in out


def test_cli_cached_model_requires_cache(tmp_path):
    import pytest
    te = tmp_path / "te.jsonl"
    main(["synth", "--n", "10", "--out", str(te)])
    with pytest.raises(SystemExit):
        main(["diagnose", "--benchmark", str(te), "--model", "cached:hf:x/y"])
