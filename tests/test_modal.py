from artifact_signals.modal import Atom, Bin, Frame, Modal, Not, evaluate, extract_formula, parse, to_str, uses_modal


def frame():
    return Frame(("w0", "w1", "w2"), (("w0", "w1"), ("w0", "w2"), ("w1", "w1")),
                 {"w0": {"p": True, "q": False}, "w1": {"p": True, "q": True}, "w2": {"p": False, "q": True}})


def test_parse_roundtrip_and_precedence():
    f = parse("[]p -> <>q & ~r | s")
    assert to_str(f) == "([]p -> ((<>q & ~r) | s))"
    assert parse("a -> b -> c") == Bin("->", Atom("a"), Bin("->", Atom("b"), Atom("c")))
    assert uses_modal(f) and not uses_modal(parse("p & q"))


def test_model_checking():
    fr = frame()
    assert evaluate(parse("[]q"), fr, "w0") is True        # q true at w1 and w2
    assert evaluate(parse("[]p"), fr, "w0") is False       # p false at w2
    assert evaluate(parse("<>~p"), fr, "w0") is True
    assert evaluate(parse("[]p"), fr, "w2") is True        # vacuous: w2 has no successors
    assert evaluate(parse("<>p"), fr, "w2") is False
    assert evaluate(parse("[][]q"), fr, "w0") is True


def test_three_valued_undetermined():
    fr = frame()
    val = {w: dict(v) for w, v in fr.valuation.items()}
    del val["w2"]["q"]
    fr2 = Frame(fr.worlds, fr.access, val)
    assert evaluate(parse("[]q"), fr2, "w0") is None       # q at w2 unknown, q at w1 true -> undetermined
    assert evaluate(parse("<>q"), fr2, "w0") is True       # forced by w1
    assert evaluate(parse("q | ~q"), fr2, "w2") is None    # strong Kleene, not classical


def test_extract_formula_from_chatter():
    f = extract_formula("Sure! Here is a formula that works: []p -> q . Hope that helps.")
    assert f is not None and to_str(f) == "([]p -> q)"
    assert extract_formula("I cannot answer.") in (None, Atom("I"))  # a lone word parses as an atom; verifier rejects it


def test_describe_lists_facts():
    lines = frame().describe()
    assert lines[0].startswith("Worlds:") and "w0 can see w1" in lines[1] and "At w2: p is false, q is true." in lines
