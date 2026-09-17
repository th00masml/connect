"""SynthModal-Control: a generated modal-logic benchmark with no artifacts by construction.

Two task types:
  truth     : "Is formula F true at world w?"  options TRUE / FALSE, balanced.
  generate  : "Write a modal formula true at w."  free-form; label is one canonical
              formula, semantic correctness is checked by the model checker.

OOD probes are items whose truth value is genuinely undetermined (an atom in the formula
has no valuation at a relevant world), labelled ABSTAIN.
"""

from __future__ import annotations

import random
import string

from .data import ABSTAIN, Benchmark, Item
from .modal import Frame, atoms_in, evaluate, random_formula, random_frame, to_str, uses_modal

TRUTH_OPTIONS = ["TRUE", "FALSE"]


def _names(rng: random.Random, n_worlds: int, n_atoms: int, novel: bool) -> tuple[list[str], list[str]]:
    if not novel:
        return [f"w{i}" for i in range(n_worlds)], ["p", "q", "r", "s"][:n_atoms]
    used: set[str] = set()

    def tok(k: int) -> str:
        while True:
            t = "".join(rng.choice(string.ascii_lowercase) for _ in range(k))
            if t not in used:
                used.add(t)
                return t

    return [tok(3) for _ in range(n_worlds)], [tok(2) for _ in range(n_atoms)]


def _make_ood(rng: random.Random, frame: Frame, formula, w: str) -> Frame | None:
    atoms = sorted(atoms_in(formula))
    slots = [(u, a) for u in frame.worlds for a in atoms]
    rng.shuffle(slots)
    for u, a in slots:
        val = {k: dict(v) for k, v in frame.valuation.items()}
        val[u].pop(a, None)
        cand = Frame(frame.worlds, frame.access, val)
        if evaluate(formula, cand, w) is None:
            return cand
    return None


def generate(
    n: int,
    *,
    seed: int = 0,
    task: str = "truth",
    n_worlds: tuple[int, int] = (2, 4),
    n_atoms: tuple[int, int] = (2, 3),
    depth: tuple[int, int] = (1, 3),
    ood_frac: float = 0.0,
    novel_names: bool = True,
    name: str | None = None,
) -> Benchmark:
    rng = random.Random(seed)
    n_ood = int(round(n * ood_frac))
    n_in = n - n_ood
    per_class = {"TRUE": (n_in + 1) // 2, "FALSE": n_in // 2}
    counts = {"TRUE": 0, "FALSE": 0}
    items: list[Item] = []
    attempts = 0
    while len(items) < n and attempts < 200 * n:
        attempts += 1
        worlds, atoms = _names(rng, rng.randint(*n_worlds), rng.randint(*n_atoms), novel_names)
        frame = random_frame(rng, worlds, atoms)
        w = worlds[0]
        formula = random_formula(rng, atoms, rng.randint(*depth), force_modal=(task == "generate"))
        val = evaluate(formula, frame, w)
        if val is None:
            continue
        need_ood = sum(1 for i in items if i.ood) < n_ood
        if task == "truth":
            if need_ood and rng.random() < 0.5:
                ood_frame = _make_ood(rng, frame, formula, w)
                if ood_frame is None:
                    continue
                items.append(_truth_item(len(items), ood_frame, formula, w, ABSTAIN, atoms, ood=True))
                continue
            lab = "TRUE" if val else "FALSE"
            if counts[lab] >= per_class[lab]:
                continue
            counts[lab] += 1
            items.append(_truth_item(len(items), frame, formula, w, lab, atoms))
        elif task == "generate":
            if not (val and uses_modal(formula)):
                continue
            items.append(
                Item(
                    id=f"synth-gen-{len(items):04d}",
                    premises=frame.describe(atoms),
                    question=(
                        f"Write one modal formula over the atoms {', '.join(atoms)} that uses at least one "
                        f"modal operator ([] or <>) and is true at world {w}. Use ~ & | -> [] <> and parentheses. "
                        "Output only the formula."
                    ),
                    options=None,
                    label=to_str(formula),
                    source_id=f"synth-gen-{len(items):04d}",
                    meta={"frame": frame.to_dict(), "world": w, "atoms": atoms, "task": "generate"},
                )
            )
        else:
            raise ValueError(task)
    if len(items) < n:
        raise RuntimeError(f"could not generate {n} items (got {len(items)}); relax constraints")
    rng.shuffle(items)
    return Benchmark(name or f"synthmodal-{task}", items)


def _truth_item(k: int, frame: Frame, formula, w: str, label: str, atoms: list[str], ood: bool = False) -> Item:
    return Item(
        id=f"synth-truth-{k:04d}",
        premises=frame.describe(atoms),
        question=f"Is the formula {to_str(formula)} true at world {w}?",
        options=list(TRUTH_OPTIONS),
        label=label,
        ood=ood,
        source_id=f"synth-truth-{k:04d}",
        meta={"frame": frame.to_dict(), "formula": to_str(formula), "world": w, "atoms": atoms, "task": "truth"},
    )
