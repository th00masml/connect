"""Run a model over a benchmark under one rendering, with an on-disk cache keyed by prompt text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .data import Benchmark
from .judge import Judge, exact_match
from .render import Rendering
from .stats import mean


@dataclass
class Run:
    model: str
    benchmark: str
    rendering: str
    constrained: bool
    answers: dict[str, str] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)
    correct: dict[str, bool] = field(default_factory=dict)
    confidence: dict[str, float | None] = field(default_factory=dict)
    ood: set[str] = field(default_factory=set)

    @property
    def accuracy(self) -> float:
        """Accuracy on in-distribution items. OOD probes are scored separately by abstention_failure;
        counting them here would penalise every rendering that lacks an abstain option."""
        return mean([1.0 if v else 0.0 for i, v in self.correct.items() if i not in self.ood])

    @property
    def accuracy_all(self) -> float:
        return mean([1.0 if v else 0.0 for v in self.correct.values()])

    def accuracy_on(self, ids: list[str]) -> float:
        return mean([1.0 if self.correct[i] else 0.0 for i in ids if i in self.correct])

    def invalid_rate(self) -> float:
        return mean([0.0 if a else 1.0 for a in self.answers.values()])

    def rejudge(self, bench: Benchmark, judge: Judge) -> "Run":
        by = bench.by_id()
        r = Run(self.model, self.benchmark, self.rendering, self.constrained, dict(self.answers), dict(self.raw), {},
                dict(self.confidence), set(self.ood))
        r.correct = {i: judge(by[i], self.answers[i], self.raw[i]) for i in self.answers if i in by}
        return r

    def to_dict(self) -> dict:
        return {"model": self.model, "benchmark": self.benchmark, "rendering": self.rendering,
                "constrained": self.constrained, "accuracy": self.accuracy, "accuracy_all": self.accuracy_all,
                "ood": sorted(self.ood), "answers": self.answers,
                "raw": self.raw, "correct": self.correct, "confidence": self.confidence}


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}

    def _file(self, model: str, rendering: str, constrained: bool) -> Path:
        key = hashlib.sha1(f"{model}|{rendering}|{constrained}".encode()).hexdigest()[:16]
        return self.root / f"{key}.json"

    def load(self, model: str, rendering: str, constrained: bool) -> dict:
        f = self._file(model, rendering, constrained)
        if str(f) not in self._mem:
            self._mem[str(f)] = json.loads(f.read_text()) if f.exists() else {}
        return self._mem[str(f)]

    def save(self, model: str, rendering: str, constrained: bool) -> None:
        f = self._file(model, rendering, constrained)
        f.write_text(json.dumps(self._mem.get(str(f), {})))


def run_model(
    model,
    bench: Benchmark,
    rendering: Rendering,
    judge: Judge = exact_match,
    *,
    constrained: bool = False,
    cache: Cache | None = None,
    batch_size: int = 16,
) -> Run:
    prompts = [rendering(it) for it in bench]
    run = Run(model.name, bench.name, rendering.name, constrained, ood={i.id for i in bench if i.ood})
    store = cache.load(model.name, rendering.name, constrained) if cache else {}
    todo = []
    gens: dict[str, tuple[str, float | None]] = {}
    for p in prompts:
        h = hashlib.sha1(p.text.encode()).hexdigest()
        if h in store:
            gens[p.item_id] = (store[h]["text"], store[h].get("confidence"))
        else:
            todo.append((h, p))
    for k in range(0, len(todo), batch_size):
        chunk = todo[k:k + batch_size]
        outs = model.generate([p for _, p in chunk], constrained=constrained)
        for (h, p), g in zip(chunk, outs):
            gens[p.item_id] = (g.text, g.confidence)
            store[h] = {"text": g.text, "confidence": g.confidence}
    if cache and todo:
        cache.save(model.name, rendering.name, constrained)
    for it, p in zip(bench, prompts):
        raw, conf = gens[it.id]
        ans = p.decode(raw)
        run.raw[it.id], run.answers[it.id], run.confidence[it.id] = raw, ans, conf
        run.correct[it.id] = bool(judge(it, ans, raw))
    return run
