"""Reproduction tests: the adapters must recover the numbers published in the two source repos."""

from artifact_signals.benchmarks import cod, kripke
from artifact_signals.data import ABSTAIN


def test_cod_reproduces_published_numbers():
    test, dic = cod.load()
    assert len(test) == 300 and len(dic) == 3006 and test.task == "choice"
    runs = cod.cached_runs(test)
    pub = cod.published()["metrics"]
    for key, name in (("B_majority", "B_majority"), ("B_fuzzy", "B_fuzzy"), ("M_local_qwen14b", "M_local_qwen14b"), ("M_comm_sonnet5", "M_comm_sonnet5")):
        assert abs(runs[key].accuracy - pub[name]["all"]) < 1e-9
    sims = cod.dictsim(test)
    hard = cod.hard_ids(test, sims)
    assert len(hard) == 33
    assert abs(runs["B_fuzzy"].accuracy_on(hard) - pub["B_fuzzy"]["hard"]) < 1e-9
    assert abs(runs["M_comm_sonnet5"].accuracy_on(hard) - pub["M_comm_sonnet5"]["hard"]) < 1e-9


def test_cod_char3_recomputes_dictsim_and_fuzzy_baseline():
    test, dic = cod.load()
    sims = cod.dictsim(test)
    re_sims = cod.recompute_dictsim(test, dic)
    assert max(abs(sims[i] - re_sims[i]) for i in sims) < 1e-9
    prof = cod.profile("M_comm_sonnet5", retrieval_methods=("char3",))
    rd = prof.signals["retrieval_dominance"].details
    assert abs(rd["per_method"]["char3"]["accuracy"] - 0.7066666666666667) < 1e-9
    assert rd["llm_accuracy"] == runs_acc()


def runs_acc():
    return 0.6666666666666666


def test_cod_profile_and_interventions_shape():
    prof = cod.profile("M_comm_sonnet5", retrieval_methods=("char3",))
    v = prof.vector()
    assert v["format_sensitivity"] is None and v["abstention_failure"] is None
    assert v["retrieval_dominance"] > 1.0 and abs(v["semantic_validity_gap"]) < 1e-9
    assert not prof.gated
    res = cod.interventions("M_comm_sonnet5")
    lr = res["leakage_resplit"]
    assert lr["n_kept"] == 33 and lr["after"]["pairwise"]["llm_only"] == 14 and lr["after"]["pairwise"]["retrieval_only"] == 1
    assert lr["gap"]["llm_minus_retrieval"]["delta"] > 0.4       # the LLM-minus-retrieval gap widens by ~0.43 on novel terms
    assert lr["llm"]["accuracy"]["delta"] > 0                    # and the LLM does not drop on the novel slice
    assert res["memorization_probe"]["exact_code"] == 0.0


def test_kripke_reproduces_published_numbers():
    bench = kripke.load()
    assert len(bench) == 160 and len(bench.ood_items()) == 80 and all(i.label == ABSTAIN for i in bench.ood_items())
    for run in ("gold", "product"):
        runs = kripke.cached_runs(bench, run=run)
        pub = kripke.published(run)["per_model"]
        for m in kripke.MODELS:
            for arm in kripke.ARMS:
                r = runs[(m, arm)]
                assert round(r.accuracy * 80) == pub[m][arm]["strict_exact"], (run, m, arm)
                lenient = r.rejudge(bench, kripke.lenient_judge)
                assert round(lenient.accuracy * 80) == pub[m][arm]["lenient_exact"], (run, m, arm)
                assert pub[m][arm]["absent_empty"] == sum(1 for i in bench.ood_items() if r.answers[i.id] == ABSTAIN)
        forced = kripke.forced_ids(bench, kripke.allowed_set(bench, kripke.run_meta(run=run)["allowed_set"]))
        assert len(forced) == kripke.published(run)["forced_fixtures"]


def test_kripke_profile_is_gated_and_nobody_abstains():
    prof = kripke.profile(kripke.MODELS[1], "gold")
    assert prof.gated and prof.signals["format_sensitivity"].value > 0.03
    assert prof.signals["abstention_failure"].value == 1.0
    assert all(v == 1.0 for v in prof.signals["abstention_failure"].details["per_arm"].values())
    svg = prof.signals["semantic_validity_gap"].details
    assert abs(svg["forced_rate"] - 25 / 80) < 1e-9
    res = kripke.interventions(kripke.MODELS[1])
    ca = res["constraint_ablation"]["per_run"]
    assert ca["gold"]["em"]["constrained_minus_free"] > 0.5
    assert ca["product"]["forced_rate"] == 0.0
    assert abs(ca["product"]["sem"]["constrained_minus_free"]) < abs(ca["gold"]["sem"]["constrained_minus_free"])
    assert res["leak_removal"]["free_subset"]["outputs_identical"] == 55 and res["leak_removal"]["manipulation_check"]
