from artifact_signals.data import ABSTAIN, Benchmark, Item
from artifact_signals.evaluate import run_model
from artifact_signals.judge import exact_match, lenient_choice
from artifact_signals.models import ConstantModel, FirstChoiceModel, NoisyFormatModel, OracleModel
from artifact_signals.render import abstention_rendering, choice_prompt, partial_input_renderings, standard_renderings


def bench(n=40):
    items = [Item(id=f"i{k}", premises=[f"fact {k}", "second fact"], question=f"q{k}", options=["alpha", "beta", "gamma"],
                  label=["alpha", "beta", "gamma"][k % 3]) for k in range(n)]
    return Benchmark("toy", items)


def test_decoder_handles_labels_text_and_junk():
    it = bench(1)[0]
    p = choice_prompt(it, scheme="letters")
    assert p.decode("B") == "beta" and p.decode("(b) beta") == "beta" and p.decode("Answer: C.") == "gamma"
    assert p.decode("gamma") == "gamma" and p.decode("Gamma.") == "gamma" and p.decode("The answer is B") == "beta"
    assert p.decode("I think gamma") == ""  # chatter is a decoding failure by design; the lenient judge reads through it
    assert p.decode("Bananas") == ""  # 'B' followed by a word char is not the token B
    p2 = choice_prompt(it, scheme="numbers", permute_seed=7)
    assert set(p2.choice_map.values()) == {"alpha", "beta", "gamma"}
    assert p2.decode("1") == p2.choice_map["1"]


def test_oracle_is_format_invariant_and_first_choice_is_position_biased():
    b = bench()
    gold = {i.id: i.label for i in b}
    o = OracleModel(gold, accuracy=1.0)
    for r in standard_renderings(0):
        assert run_model(o, b, r).accuracy == 1.0
    f = FirstChoiceModel()
    assert run_model(f, b, standard_renderings(0)[0]).accuracy == 14 / 40   # 'alpha' is first in the unpermuted order
    permuted = run_model(f, b, standard_renderings(0)[1]).accuracy
    assert permuted != 14 / 40


def test_constant_model_and_abstention_option():
    b = bench()
    run = run_model(ConstantModel("beta"), b, standard_renderings(0)[0])
    assert all(a == "beta" for a in run.answers.values())
    p = abstention_rendering()(b[0])
    assert ABSTAIN in p.choice_map.values()
    run2 = run_model(ConstantModel(ABSTAIN), b, abstention_rendering())
    assert all(a == ABSTAIN for a in run2.answers.values())


def test_partial_input_renderings_omit_fields():
    it = bench(1)[0]
    texts = {r.name: r(it).text for r in partial_input_renderings("choice")}
    assert "fact 0" not in texts["options_only"] and "q0" not in texts["options_only"]
    assert "fact 0" not in texts["question_only"] and "q0" in texts["question_only"]
    assert "fact 0" in texts["premises_only"] and "q0" not in texts["premises_only"]


def test_noisy_format_model_dissociates_strict_and_lenient_judges():
    b = bench(200)
    gold = {i.id: i.label for i in b}
    m = NoisyFormatModel(gold, accuracy=0.6, noise=0.5, seed=1)
    r = standard_renderings(0)[0]
    free_strict = run_model(m, b, r, exact_match)
    free_lenient = run_model(m, b, r, lenient_choice)
    con_strict = run_model(m, b, r, exact_match, constrained=True)
    assert free_strict.accuracy < free_lenient.accuracy
    assert abs(con_strict.accuracy - free_lenient.accuracy) < 0.05
