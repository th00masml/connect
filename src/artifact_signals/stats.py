"""Small, dependency-free statistics used across the instrument."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Sequence


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pstdev(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_test(k: int, n: int, p: float = 0.5, alternative: str = "two-sided") -> float:
    if n == 0:
        return 1.0
    pmf = [math.exp(_log_binom_pmf(i, n, p)) for i in range(n + 1)]
    if alternative == "greater":
        return min(1.0, sum(pmf[k:]))
    if alternative == "less":
        return min(1.0, sum(pmf[: k + 1]))
    obs = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= obs * (1 + 1e-9)))


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar on paired binary outcomes. Returns discordant counts and two-sided p."""
    if len(a) != len(b):
        raise ValueError("paired sequences differ in length")
    ab = sum(1 for x, y in zip(a, b) if x and not y)
    ba = sum(1 for x, y in zip(a, b) if y and not x)
    n = ab + ba
    return {"a_only": ab, "b_only": ba, "n_discordant": n, "p": binom_test(min(ab, ba), n, 0.5)}


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    stat: Callable[[Sequence[float]], float] = mean,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    keys = list(groups)
    est = stat(list(values))
    if not keys:
        return est, est, est
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(len(keys)):
            sample.extend(groups[rng.choice(keys)])
        draws.append(stat(sample))
    draws.sort()
    lo = draws[int(math.floor(alpha / 2 * (n_boot - 1)))]
    hi = draws[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))]
    return est, lo, hi


def holm(pvals: Sequence[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def brier(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    if not probs:
        return float("nan")
    return mean([(p - (1.0 if o else 0.0)) ** 2 for p, o in zip(probs, outcomes)])


def reliability(probs: Sequence[float], outcomes: Sequence[bool], bins: int = 5) -> list[dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            out.append({"lo": lo, "hi": hi, "n": len(idx),
                        "mean_conf": mean([probs[i] for i in idx]),
                        "hit_rate": mean([1.0 if outcomes[i] else 0.0 for i in idx])})
        else:
            out.append({"lo": lo, "hi": hi, "n": 0, "mean_conf": None, "hit_rate": None})
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def point_biserial(x: Sequence[float], y: Sequence[bool]) -> float:
    return pearson(x, [1.0 if b else 0.0 for b in y])


def _ranks(x: Sequence[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def risk_coverage_auc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better selective prediction)."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: -confidences[i])
    errors, area, prev_cov, prev_risk = 0, 0.0, 0.0, 0.0
    for k, i in enumerate(order, start=1):
        errors += 0 if correct[i] else 1
        cov, risk = k / n, errors / k
        area += (cov - prev_cov) * (risk + prev_risk) / 2
        prev_cov, prev_risk = cov, risk
    return area
