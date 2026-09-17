from collections import Counter

from artifact_signals.data import ABSTAIN
from artifact_signals.judge import formula_exact, formula_verifier
from artifact_signals.modal import Frame, evaluate, parse
from artifact_signals.synth import generate


def test_truth_task_balanced_and_verified():
    b = generate(60, seed=1, task="truth", ood_frac=0.1)
    c = Counter(i.label for i in b)
    assert c[ABSTAIN] == 6 and c["TRUE"] == 27 and c["FALSE"] == 27
    for i in b:
        v = evaluate(parse(i.meta["formula"]), Frame.from_dict(i.meta["frame"]), i.meta["world"])
        assert (v is None) == i.ood
        if not i.ood:
            assert i.label == ("TRUE" if v else "FALSE")


def test_novel_names_have_no_lexical_overlap_between_items():
    b = generate(30, seed=2, task="truth")
    atoms = [tuple(i.meta["atoms"]) for i in b]
    assert len(set(atoms)) == len(atoms)


def test_generate_task_labels_verify_and_exact_judge_is_strict():
    b = generate(20, seed=3, task="generate")
    for i in b:
        assert formula_verifier(i, i.label, i.label)
        assert formula_exact(i, i.label, i.label)
        assert not formula_exact(i, "Sure: " + i.label, "")
        assert formula_verifier(i, "", "Sure, the answer is " + i.label + " and that is final.")
