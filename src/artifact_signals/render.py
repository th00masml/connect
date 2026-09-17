"""Renderings: semantically equivalent surface forms of an item, plus partial-input forms.

A Rendering turns an Item into a Prompt. The Prompt carries the allowed answer tokens
(for constrained decoding), a map from token to canonical option text, and a decoder
from raw model output back to canonical answer. Correctness is always judged on the
canonical answer, so no rendering can change what counts as right.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from .data import ABSTAIN, Item

INSTRUCTIONS = [
    "Answer the question using only the information given.",
    "Based solely on the premises below, choose the correct answer.",
    "Read the context, then select the option that follows from it.",
    "Using the facts provided and nothing else, determine the answer.",
    "Consider the statements and pick the answer they support.",
]

LABEL_SCHEMES: dict[str, Callable[[int], str]] = {
    "letters": lambda i: "ABCDEFGHIJ"[i],
    "numbers": lambda i: str(i + 1),
    "lower_paren": lambda i: f"({'abcdefghij'[i]})",
    "roman": lambda i: ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][i],
}

DELIMITERS = {"newline": "\n", "semicolon": "; ", "bullet": "\n- ", "pipe": " | "}


@dataclass
class Prompt:
    item_id: str
    text: str
    choices: list[str] | None = None            # allowed answer tokens for constrained decoding
    choice_map: dict[str, str] | None = None    # token -> canonical option text
    decode: Callable[[str], str] = lambda s: s.strip()
    regex: str | None = None                    # grammar constraint for free-form tasks
    meta: dict = field(default_factory=dict)


@dataclass
class Rendering:
    name: str
    render: Callable[[Item], Prompt]

    def __call__(self, item: Item) -> Prompt:
        return self.render(item)


_PREFIX = re.compile(r"^\W*(?:the\s+)?(?:final\s+)?(?:answer|option|choice|label)\s*(?:is|:|=)?\s*", re.IGNORECASE)


def _decode_choice(raw: str, choice_map: dict[str, str], options: list[str]) -> str:
    """Strict decoder: a label token at the start of the first line (an 'Answer:' prefix is
    tolerated), or a first line that is the option text itself. Chatter around the answer is
    a decoding failure on purpose; lenient reading belongs to the judge, not the decoder."""
    s = raw.strip()
    first = s.splitlines()[0].strip() if s else ""
    first = _PREFIX.sub("", first)
    for tok, opt in choice_map.items():
        if re.match(rf"^\W*{re.escape(tok)}(?=\W|$)", first, flags=re.IGNORECASE):
            return opt
    low = re.sub(r"[\s\W]+$", "", first.lower())
    for opt in options:
        if low == opt.lower():
            return opt
    if options and low:
        best = max(options, key=lambda o: SequenceMatcher(None, low, o.lower()).ratio())
        if SequenceMatcher(None, low, best.lower()).ratio() >= 0.85:
            return best
    return ""


def choice_prompt(
    item: Item,
    *,
    instruction: int = 0,
    scheme: str = "letters",
    permute_seed: int | None = None,
    premise_seed: int | None = None,
    delimiter: str = "newline",
    include: tuple[str, ...] = ("premises", "question", "options"),
    abstain: bool = False,
) -> Prompt:
    if not item.options:
        raise ValueError(f"item {item.id} has no options")
    options = list(item.options)
    if abstain and ABSTAIN not in options:
        options.append(ABSTAIN)
    order = list(range(len(options)))
    if permute_seed is not None:
        random.Random(f"{permute_seed}:{item.id}").shuffle(order)
    lab = LABEL_SCHEMES[scheme]
    tokens = [lab(i) for i in range(len(options))]
    choice_map = {tokens[i]: options[order[i]] for i in range(len(options))}
    premises = list(item.premises)
    if premise_seed is not None:
        random.Random(f"{premise_seed}:{item.id}").shuffle(premises)
    delim = DELIMITERS[delimiter]
    parts = [INSTRUCTIONS[instruction % len(INSTRUCTIONS)]]
    if "premises" in include and premises:
        parts.append("Premises:" + delim + delim.join(premises))
    if "question" in include:
        parts.append("Question: " + item.question)
    if "options" in include:
        parts.append("Options:\n" + "\n".join(f"{tokens[i]}. {options[order[i]]}" for i in range(len(options))))
    parts.append("Answer with the option label only.")
    decode = lambda raw, cm=choice_map, op=options: _decode_choice(raw, cm, op)  # noqa: E731
    return Prompt(item.id, "\n\n".join(parts), choices=tokens, choice_map=choice_map, decode=decode,
                  meta={"scheme": scheme, "abstain": abstain, "include": include})


def freeform_prompt(
    item: Item,
    *,
    instruction: int = 0,
    premise_seed: int | None = None,
    delimiter: str = "newline",
    include: tuple[str, ...] = ("premises", "question"),
    regex: str | None = None,
) -> Prompt:
    premises = list(item.premises)
    if premise_seed is not None:
        random.Random(f"{premise_seed}:{item.id}").shuffle(premises)
    delim = DELIMITERS[delimiter]
    parts = [INSTRUCTIONS[instruction % len(INSTRUCTIONS)]]
    if "premises" in include and premises:
        parts.append("Context:" + delim + delim.join(premises))
    if "question" in include:
        parts.append("Task: " + item.question)
    return Prompt(item.id, "\n\n".join(parts), regex=regex, meta={"include": include})


def rendering(name: str, regex: str | None = None, **kw) -> Rendering:
    """Task-agnostic rendering: choice items get choice_prompt, others freeform_prompt."""

    def render(item: Item) -> Prompt:
        if item.options:
            return choice_prompt(item, **kw)
        ff_kw = {k: v for k, v in kw.items() if k in ("instruction", "premise_seed", "delimiter")}
        inc = kw.get("include")
        if inc:
            ff_kw["include"] = tuple(i for i in inc if i != "options") or ("question",)
        return freeform_prompt(item, regex=regex, **ff_kw)

    return Rendering(name, render)


def baseline_rendering(regex: str | None = None) -> Rendering:
    return rendering("baseline", regex=regex)


def standard_renderings(seed: int = 0, regex: str | None = None) -> list[Rendering]:
    """Five semantically equivalent surface forms used for the format-sensitivity signal."""
    return [
        rendering("baseline", regex=regex),
        rendering("permute_options", regex=regex, permute_seed=seed),
        rendering("relabel_numbers", regex=regex, scheme="numbers"),
        rendering("paraphrase_instruction", regex=regex, instruction=2),
        rendering("reorder_premises_semicolon", regex=regex, premise_seed=seed + 1, delimiter="semicolon"),
    ]


def partial_input_renderings(task: str = "choice") -> list[Rendering]:
    if task == "choice":
        return [
            rendering("options_only", include=("options",)),
            rendering("question_only", include=("question", "options")),
            rendering("premises_only", include=("premises", "options")),
        ]
    return [
        rendering("question_only", include=("question",)),
        rendering("premises_only", include=("premises",)),
    ]


def abstention_rendering(**kw) -> Rendering:
    return rendering("abstention", abstain=True, **kw)
