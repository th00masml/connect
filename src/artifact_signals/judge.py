"""Judges: (item, canonical_answer, raw_output) -> bool.

Two families matter for the semantic-validity gap:
  strict judges (exact match on the decoded answer) reward format compliance;
  lenient/verifier judges read through format noise and check content.
"""

from __future__ import annotations

import re
from typing import Callable

from .data import Item
from .modal import Frame, evaluate, extract_formula, parse, uses_modal

Judge = Callable[[Item, str, str], bool]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(item: Item, answer: str, raw: str = "") -> bool:
    return bool(answer) and _norm(answer) == _norm(item.label)


def lenient_choice(item: Item, answer: str, raw: str = "") -> bool:
    """Accept the gold option if it appears anywhere in the raw output and no other option does."""
    if exact_match(item, answer, raw):
        return True
    if not item.options:
        return False
    low = raw.lower()
    hits = [o for o in item.options if re.search(rf"(?<!\w){re.escape(o.lower())}(?!\w)", low)]
    return hits == [item.label]


def formula_verifier(item: Item, answer: str, raw: str = "") -> bool:
    """Semantic judge for the synth 'generate' task: any formula true at the target world
    that uses a modal operator counts, regardless of its surface form."""
    frame = Frame.from_dict(item.meta["frame"])
    w = item.meta["world"]
    f = None
    for cand in (answer, raw):
        try:
            f = parse(cand.strip())
            break
        except Exception:
            f = extract_formula(cand)
            if f is not None:
                break
    if f is None or not uses_modal(f):
        return False
    return evaluate(f, frame, w) is True


def formula_exact(item: Item, answer: str, raw: str = "") -> bool:
    """Strict judge for the generate task: the decoded answer must parse and print exactly as the label."""
    try:
        from .modal import to_str

        return to_str(parse(answer.strip())) == item.label
    except Exception:
        return False


JUDGES: dict[str, Judge] = {
    "exact": exact_match,
    "lenient": lenient_choice,
    "formula_verifier": formula_verifier,
    "formula_exact": formula_exact,
}
