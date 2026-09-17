"""Non-parametric nearest-neighbour baselines: fuzzy token-set match, char n-gram TF-IDF, BM25.

Pure Python; rapidfuzz is used when installed. All three return, per test text, the
predicted label of the nearest training text and the similarity to it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher

try:  # optional fast path
    from rapidfuzz import fuzz as _rf  # type: ignore
except Exception:  # pragma: no cover
    _rf = None


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def fuzzy_sim(a: str, b: str) -> float:
    if _rf is not None:
        return _rf.token_set_ratio(a, b) / 100.0
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    dice = 2 * len(ta & tb) / (len(ta) + len(tb))
    return 0.5 * dice + 0.5 * SequenceMatcher(None, a.lower(), b.lower()).ratio()


class CharTfidf:
    def __init__(self, docs: list[str], n: tuple[int, int] = (3, 5)):
        self.n = n
        self.df: Counter = Counter()
        self.vecs = [self._tf(d) for d in docs]
        for v in self.vecs:
            self.df.update(v.keys())
        self.N = max(1, len(docs))
        self.vecs = [self._weight(v) for v in self.vecs]

    def _tf(self, d: str) -> Counter:
        s = re.sub(r"\s+", " ", d.lower())
        c: Counter = Counter()
        for k in range(self.n[0], self.n[1] + 1):
            for i in range(max(0, len(s) - k + 1)):
                c[s[i:i + k]] += 1
        return c

    def _weight(self, tf: Counter) -> dict[str, float]:
        v = {g: (1 + math.log(c)) * math.log((1 + self.N) / (1 + self.df.get(g, 0))) + 1.0 for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def query(self, q: str) -> list[float]:
        qv = self._weight(self._tf(q))
        return [sum(qv.get(g, 0.0) * w for g, w in dv.items()) for dv in self.vecs]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokens(d) for d in docs]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 1.0
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.N = len(self.docs)
        self.tfs = [Counter(d) for d in self.docs]

    def idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def query(self, q: str) -> list[float]:
        qt = tokens(q)
        out = []
        for tf, dl in zip(self.tfs, self.dl):
            s = 0.0
            for t in qt:
                if t in tf:
                    f = tf[t]
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


METHODS = ("fuzzy", "tfidf", "bm25")


def nearest_neighbor(
    train_texts: list[str], train_labels: list[str], test_texts: list[str], method: str = "fuzzy"
) -> tuple[list[str], list[float]]:
    if not train_texts:
        return [""] * len(test_texts), [0.0] * len(test_texts)
    if method == "fuzzy":
        sims_fn = lambda q: [fuzzy_sim(q, d) for d in train_texts]  # noqa: E731
    elif method == "tfidf":
        idx = CharTfidf(train_texts)
        sims_fn = idx.query
    elif method == "bm25":
        idx = BM25(train_texts)
        sims_fn = idx.query
    else:
        raise ValueError(method)
    preds, best = [], []
    for q in test_texts:
        sims = sims_fn(q)
        j = max(range(len(sims)), key=sims.__getitem__)
        preds.append(train_labels[j])
        best.append(float(sims[j]))
    return preds, best


def max_similarity(train_texts: list[str], test_texts: list[str], method: str = "fuzzy") -> list[float]:
    _, sims = nearest_neighbor(train_texts, [""] * len(train_texts), test_texts, method)
    return sims
