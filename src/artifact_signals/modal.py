"""Kripke frames, an ASCII modal-logic formula language, and a three-valued model checker.

Syntax:   atoms are identifiers;  ~ negation;  &  |  ->  binary;  [] box;  <> diamond.
Precedence, loosest to tightest:  ->  (right-assoc),  |,  &,  unary (~ [] <>).

Evaluation is strong-Kleene three-valued: an atom with no valuation at a world is
undetermined (None), and undeterminedness propagates unless the connective's value is
already forced. That gives principled "insufficient information" probes for free.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

Truth = Optional[bool]


@dataclass(frozen=True)
class Frame:
    worlds: tuple[str, ...]
    access: tuple[tuple[str, str], ...]
    valuation: dict[str, dict[str, bool]]

    def successors(self, w: str) -> list[str]:
        return [v for u, v in self.access if u == w]

    def atoms(self) -> list[str]:
        return sorted({a for d in self.valuation.values() for a in d})

    def to_dict(self) -> dict:
        return {
            "worlds": list(self.worlds),
            "access": [list(e) for e in self.access],
            "valuation": {w: dict(v) for w, v in self.valuation.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            worlds=tuple(d["worlds"]),
            access=tuple((str(a), str(b)) for a, b in d["access"]),
            valuation={w: {a: bool(t) for a, t in v.items()} for w, v in d["valuation"].items()},
        )

    def describe(self, atoms: list[str] | None = None) -> list[str]:
        """Natural-language premises, one string per fact group."""
        atoms = atoms or self.atoms()
        lines = [f"Worlds: {', '.join(self.worlds)}."]
        if self.access:
            lines.append("Accessibility: " + "; ".join(f"{u} can see {v}" for u, v in self.access) + ".")
        else:
            lines.append("Accessibility: no world can see any world.")
        for w in self.worlds:
            facts = []
            for a in atoms:
                if a in self.valuation.get(w, {}):
                    facts.append(f"{a} is {'true' if self.valuation[w][a] else 'false'}")
            if facts:
                lines.append(f"At {w}: " + ", ".join(facts) + ".")
            else:
                lines.append(f"At {w}: nothing is specified.")
        return lines


# --- formulas -----------------------------------------------------------------


class Formula:
    pass


@dataclass(frozen=True)
class Atom(Formula):
    name: str


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula


@dataclass(frozen=True)
class Bin(Formula):
    op: str  # "&", "|", "->"
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Modal(Formula):
    op: str  # "[]", "<>"
    sub: Formula


def to_str(f: Formula) -> str:
    if isinstance(f, Atom):
        return f.name
    if isinstance(f, Not):
        return "~" + to_str(f.sub)
    if isinstance(f, Modal):
        return f.op + to_str(f.sub)
    if isinstance(f, Bin):
        return f"({to_str(f.left)} {f.op} {to_str(f.right)})"
    raise TypeError(type(f))


def uses_modal(f: Formula) -> bool:
    if isinstance(f, Modal):
        return True
    if isinstance(f, Not):
        return uses_modal(f.sub)
    if isinstance(f, Bin):
        return uses_modal(f.left) or uses_modal(f.right)
    return False


def atoms_in(f: Formula) -> set[str]:
    if isinstance(f, Atom):
        return {f.name}
    if isinstance(f, (Not, Modal)):
        return atoms_in(f.sub)
    if isinstance(f, Bin):
        return atoms_in(f.left) | atoms_in(f.right)
    return set()


def depth(f: Formula) -> int:
    if isinstance(f, Atom):
        return 0
    if isinstance(f, (Not, Modal)):
        return 1 + depth(f.sub)
    if isinstance(f, Bin):
        return 1 + max(depth(f.left), depth(f.right))
    return 0


# --- three-valued evaluation ----------------------------------------------------


def _k_not(a: Truth) -> Truth:
    return None if a is None else (not a)


def _k_and(xs: list[Truth]) -> Truth:
    if any(x is False for x in xs):
        return False
    if any(x is None for x in xs):
        return None
    return True


def _k_or(xs: list[Truth]) -> Truth:
    if any(x is True for x in xs):
        return True
    if any(x is None for x in xs):
        return None
    return False


def evaluate(f: Formula, frame: Frame, w: str) -> Truth:
    if isinstance(f, Atom):
        return frame.valuation.get(w, {}).get(f.name)
    if isinstance(f, Not):
        return _k_not(evaluate(f.sub, frame, w))
    if isinstance(f, Bin):
        a, b = evaluate(f.left, frame, w), evaluate(f.right, frame, w)
        if f.op == "&":
            return _k_and([a, b])
        if f.op == "|":
            return _k_or([a, b])
        if f.op == "->":
            return _k_or([_k_not(a), b])
        raise ValueError(f.op)
    if isinstance(f, Modal):
        vals = [evaluate(f.sub, frame, v) for v in frame.successors(w)]
        return _k_and(vals) if f.op == "[]" else _k_or(vals)
    raise TypeError(type(f))


# --- parsing -------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(\[\]|<>|->|[~&|()]|[A-Za-z_][A-Za-z0-9_]*)")

# Regex approximation of the grammar for regex-constrained decoders (no balanced parens).
FORMULA_REGEX = r"(\[\]|<>|~)*\(?[A-Za-z_][A-Za-z0-9_]*\)?(\s*(&|\||->)\s*(\[\]|<>|~)*\(?[A-Za-z_][A-Za-z0-9_]*\)?)*"


class ParseError(ValueError):
    pass


def tokenize(s: str) -> list[str]:
    out, pos = [], 0
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ParseError(f"bad token at {pos}: {s[pos:pos+10]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, toks: list[str]):
        self.t, self.i = toks, 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, tok: str | None = None) -> str:
        cur = self.peek()
        if cur is None or (tok is not None and cur != tok):
            raise ParseError(f"expected {tok!r}, got {cur!r}")
        self.i += 1
        return cur

    def imp(self) -> Formula:
        left = self.disj()
        if self.peek() == "->":
            self.eat()
            return Bin("->", left, self.imp())
        return left

    def disj(self) -> Formula:
        f = self.conj()
        while self.peek() == "|":
            self.eat()
            f = Bin("|", f, self.conj())
        return f

    def conj(self) -> Formula:
        f = self.unary()
        while self.peek() == "&":
            self.eat()
            f = Bin("&", f, self.unary())
        return f

    def unary(self) -> Formula:
        cur = self.peek()
        if cur == "~":
            self.eat()
            return Not(self.unary())
        if cur in ("[]", "<>"):
            self.eat()
            return Modal(cur, self.unary())
        if cur == "(":
            self.eat()
            f = self.imp()
            self.eat(")")
            return f
        if cur is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cur):
            self.eat()
            return Atom(cur)
        raise ParseError(f"unexpected {cur!r}")


def parse(s: str) -> Formula:
    p = _Parser(tokenize(s))
    f = p.imp()
    if p.peek() is not None:
        raise ParseError(f"trailing tokens from {p.i}")
    return f


def extract_formula(text: str) -> Formula | None:
    """Return a formula embedded in free text: the whole string if it parses, else the longest
    contiguous word window that parses to something other than a bare atom, else a bare atom."""
    try:
        return parse(text)
    except ParseError:
        pass
    words = text.replace(",", " ").replace(".", " ").split()
    atom_fallback = None
    n = len(words)
    for length in range(n, 0, -1):
        for start in range(0, n - length + 1):
            cand = " ".join(words[start:start + length])
            try:
                f = parse(cand)
            except ParseError:
                continue
            if not isinstance(f, Atom):
                return f
            if atom_fallback is None:
                atom_fallback = f
    return atom_fallback


# --- generation ---------------------------------------------------------------


def random_formula(rng: random.Random, atoms: list[str], d: int, *, force_modal: bool = False) -> Formula:
    if d <= 0:
        return Atom(rng.choice(atoms))
    kinds = ["not", "and", "or", "imp", "box", "dia"]
    weights = [1, 2, 2, 2, 3, 3]
    if force_modal:
        kinds, weights = ["box", "dia"], [1, 1]
    k = rng.choices(kinds, weights)[0]
    if k == "not":
        return Not(random_formula(rng, atoms, d - 1))
    if k in ("and", "or", "imp"):
        op = {"and": "&", "or": "|", "imp": "->"}[k]
        return Bin(op, random_formula(rng, atoms, d - 1), random_formula(rng, atoms, rng.randint(0, d - 1)))
    return Modal("[]" if k == "box" else "<>", random_formula(rng, atoms, d - 1))


def random_frame(rng: random.Random, worlds: list[str], atoms: list[str], *, edge_p: float = 0.4) -> Frame:
    access = [(u, v) for u in worlds for v in worlds if rng.random() < edge_p]
    if not any(u == worlds[0] for u, _ in access) and rng.random() < 0.8:
        access.append((worlds[0], rng.choice(worlds)))
    valuation = {w: {a: rng.random() < 0.5 for a in atoms} for w in worlds}
    return Frame(tuple(worlds), tuple(access), valuation)
