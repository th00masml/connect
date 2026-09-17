"""Pre-registered prediction ledger: hash-stamped before intervention, scored after.

A Prediction names a benchmark, an intervention, a dotted quantity path into the results
dict, a direction with threshold, and a subjective confidence. Scoring reports hit rate,
Brier score, a reliability table, per-family hit rates, and a paired comparison against
the naive predictor that says every change quantity goes down.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .stats import binom_test, brier, mcnemar, mean, reliability

DIRECTIONS = ("down", "up", "flat", "above", "below")
FAMILIES = ("direction", "magnitude", "metric_specific", "specificity", "manipulation_check", "other")


@dataclass
class Prediction:
    id: str
    benchmark: str
    intervention: str
    quantity: str
    direction: str
    threshold: float
    confidence: float
    family: str = "direction"
    note: str = ""

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"{self.id}: direction must be one of {DIRECTIONS}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.id}: confidence must be in [0,1]")
        if self.family not in FAMILIES:
            raise ValueError(f"{self.id}: family must be one of {FAMILIES}")

    def holds(self, value: float) -> bool:
        d, t = self.direction, self.threshold
        if d == "down":
            return value <= -abs(t)
        if d == "up":
            return value >= abs(t)
        if d == "flat":
            return abs(value) <= abs(t)
        if d == "above":
            return value > t
        return value < t

    @property
    def is_change(self) -> bool:
        return self.quantity.endswith(".delta") or self.quantity == "delta"


@dataclass
class Ledger:
    title: str
    predictions: list[Prediction]
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta: dict = field(default_factory=dict)

    def canonical(self) -> str:
        body = {"title": self.title, "created": self.created, "meta": self.meta,
                "predictions": [asdict(p) for p in sorted(self.predictions, key=lambda p: p.id)]}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        h = self.sha256()
        body = json.loads(self.canonical())
        body["sha256"] = h
        Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return h

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> "Ledger":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        led = cls(d["title"], [Prediction(**p) for p in d["predictions"]], d.get("created", ""), d.get("meta", {}))
        if verify and "sha256" in d and d["sha256"] != led.sha256():
            raise ValueError("ledger hash mismatch: file was edited after stamping")
        return led


def lookup(results: dict, benchmark: str, intervention: str, quantity: str):
    node = results.get(benchmark, {}).get(intervention)
    if node is None:
        return None
    for part in quantity.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def score_ledger(ledger: Ledger, results: dict, *, voided: set[str] | None = None) -> dict:
    voided = voided or set()
    rows, probs, outcomes = [], [], []
    model_hits_change, naive_hits_change = [], []
    for p in ledger.predictions:
        val = lookup(results, p.benchmark, p.intervention, p.quantity)
        status = "voided" if p.id in voided else ("missing" if val is None else "scored")
        hit = p.holds(val) if status == "scored" else None
        rows.append({"id": p.id, "benchmark": p.benchmark, "intervention": p.intervention, "quantity": p.quantity,
                     "direction": p.direction, "threshold": p.threshold, "confidence": p.confidence,
                     "family": p.family, "value": val, "status": status, "hit": hit})
        if status == "scored":
            probs.append(p.confidence)
            outcomes.append(bool(hit))
            if p.is_change:
                model_hits_change.append(bool(hit))
                naive_hits_change.append(val < 0)
    n_scored = len(outcomes)
    hits = sum(outcomes)
    fam = {}
    for f in FAMILIES:
        sub = [r for r in rows if r["status"] == "scored" and r["family"] == f]
        if sub:
            fam[f] = {"n": len(sub), "hit_rate": mean([1.0 if r["hit"] else 0.0 for r in sub])}
    comp = None
    if model_hits_change:
        mc = mcnemar(model_hits_change, naive_hits_change)
        comp = {"n": len(model_hits_change), "model_hit_rate": mean([1.0 if h else 0.0 for h in model_hits_change]),
                "naive_hit_rate": mean([1.0 if h else 0.0 for h in naive_hits_change]),
                "model_only": mc["a_only"], "naive_only": mc["b_only"], "sign_test_p": mc["p"]}
    return {
        "ledger_sha256": ledger.sha256(), "n": len(ledger.predictions), "n_scored": n_scored,
        "n_voided": sum(1 for r in rows if r["status"] == "voided"),
        "n_missing": sum(1 for r in rows if r["status"] == "missing"),
        "hits": hits, "hit_rate": hits / n_scored if n_scored else None,
        "p_vs_coin": binom_test(hits, n_scored, 0.5, "greater") if n_scored else None,
        "brier": brier(probs, outcomes) if n_scored else None,
        "reliability": reliability(probs, outcomes) if n_scored else [],
        "by_family": fam, "vs_naive_on_change_predictions": comp, "predictions": rows,
    }


def format_score(s: dict) -> str:
    lines = [f"Ledger {s['ledger_sha256'][:12]}  scored {s['n_scored']}/{s['n']} (voided {s['n_voided']}, missing {s['n_missing']})"]
    if s["n_scored"]:
        lines.append(f"  hit rate {s['hit_rate']:.3f}  (p vs coin {s['p_vs_coin']:.3g})   Brier {s['brier']:.3f}")
    for f, v in s["by_family"].items():
        lines.append(f"  {f:<18} n={v['n']:<3} hit rate {v['hit_rate']:.3f}")
    c = s["vs_naive_on_change_predictions"]
    if c:
        lines.append(f"  vs naive (change preds, n={c['n']}): model {c['model_hit_rate']:.3f}  naive {c['naive_hit_rate']:.3f}  "
                     f"model-only {c['model_only']}  naive-only {c['naive_only']}  sign-test p {c['sign_test_p']:.3g}")
    for r in s["predictions"]:
        mark = {"scored": ("HIT " if r["hit"] else "MISS"), "voided": "VOID", "missing": "----"}[r["status"]]
        v = "n/a" if r["value"] is None else f"{r['value']:+.3f}"
        lines.append(f"  [{mark}] {r['id']:<8} {r['benchmark']}/{r['intervention']}  {r['quantity']} {r['direction']} {r['threshold']}  "
                     f"conf {r['confidence']:.2f}  value {v}")
    return "\n".join(lines)
