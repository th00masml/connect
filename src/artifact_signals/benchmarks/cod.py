"""Paper 1: historical cause-of-death coding (th00masml/cause-of-death-coding).

300 held-out historic strings -> one of 22 ICD-10 chapters. Cached: majority and fuzzy
char-3gram dictionary baselines, qwen2.5:14b, claude-sonnet-5, and per-item best dictionary
similarity (the HARD/novel split is similarity < 0.5).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ..data import Benchmark, Item
from ..evaluate import Run
from ..interventions import LEAKAGE
from ..judge import exact_match, lenient_choice
from ..retrieval import char3_sim
from ..signals import ArtifactRiskProfile, Signal, constraint_gain_from_runs, label_prior_skew, retrieval_dominance
from ..stats import macro_f1, mean, point_biserial, tv_distance

ROOT = Path(os.environ.get("ARTIFACT_SIGNALS_DATA", Path(__file__).resolve().parents[3] / "data" / "external")) / "cod"

CHAPTERS = [
    "Infectious", "Neoplasm", "Blood", "Endocrine/metabolic", "Mental", "Nervous", "Eye", "Ear", "Circulatory",
    "Respiratory", "Digestive", "Skin", "Musculoskeletal", "Genitourinary", "Pregnancy", "Perinatal", "Congenital",
    "Ill-defined", "Injury", "ExternalCause", "Factors", "Special",
]
_ALIAS = {"infection": "Infectious", "cancer": "Neoplasm", "tumor": "Neoplasm", "heart": "Circulatory",
          "cardiovascular": "Circulatory", "lung": "Respiratory", "metabolic": "Endocrine/metabolic",
          "endocrine": "Endocrine/metabolic", "psych": "Mental", "neuro": "Nervous", "kidney": "Genitourinary",
          "urinary": "Genitourinary", "childbirth": "Pregnancy", "newborn": "Perinatal", "ill defined": "Ill-defined",
          "unknown": "Ill-defined", "external": "ExternalCause", "injury": "Injury", "poison": "Injury"}

METHODS = {"B_majority": "majority", "B_fuzzy": "fuzzy-char3-dictionary", "M_local_qwen14b": "qwen2.5:14b",
           "M_comm_sonnet5": "claude-sonnet-5"}
LLM_METHODS = ("M_local_qwen14b", "M_comm_sonnet5")
HARD_THRESHOLD = 0.5


def parse_chapter(text: str) -> str | None:
    """Port of paper 1's lenient output parser (case-insensitive substring, then aliases)."""
    t = (text or "").strip().lower()
    for c in CHAPTERS:
        if c.lower() == t:
            return c
    for c in CHAPTERS:
        if c.lower() in t:
            return c
    for k, v in _ALIAS.items():
        if k in t:
            return v
    return None


def load(root: Path = ROOT) -> tuple[Benchmark, Benchmark]:
    """(test, dictionary). Items carry the bare historic string as `question` so retrieval
    similarity is computed on the same text paper 1 used; the prompt framing lives in renderings."""
    test = json.loads((root / "test.json").read_text(encoding="utf-8"))
    dic = json.loads((root / "dictionary.json").read_text(encoding="utf-8"))
    t_items = [Item(id=f"cod-{i:03d}", question=r["string"], options=list(CHAPTERS), label=r["chapter"],
                    source_id=f"cod-{i:03d}", meta={"index": i, "code": r["code"]}) for i, r in enumerate(test)]
    d_items = [Item(id=f"dict-{k:04d}", question=r["string"], options=list(CHAPTERS), label=r["chapter"],
                    meta={"code": r["code"]}) for k, r in enumerate(dic)]
    return Benchmark("cause_of_death", t_items), Benchmark("cause_of_death-dictionary", d_items)


def dictsim(test: Benchmark, root: Path = ROOT) -> dict[str, float]:
    sims = json.loads((root / "test_dictsim.json").read_text())
    return {it.id: float(sims[it.meta["index"]]) for it in test}


def recompute_dictsim(test: Benchmark, dictionary: Benchmark) -> dict[str, float]:
    d_texts = [i.question for i in dictionary]
    return {it.id: max(char3_sim(it.question, d) for d in d_texts) for it in test}


def cached_runs(test: Benchmark, root: Path = ROOT) -> dict[str, Run]:
    runs = {}
    raws = {"M_comm_sonnet5": "raw_comm.jsonl", "M_local_qwen14b": "raw_local.jsonl"}
    for key, name in METHODS.items():
        pred = json.loads((root / f"pred_{key}.json").read_text())["pred"]
        answers = {it.id: pred[it.meta["index"]] for it in test}
        raw = None
        if key in raws and (root / raws[key]).exists():
            rows = {r["i"]: r["raw"] for r in (json.loads(l) for l in (root / raws[key]).open(encoding="utf-8") if l.strip())}
            raw = {it.id: rows.get(it.meta["index"], "") for it in test}
        runs[key] = Run.from_answers(name, test, answers, exact_match, raw=raw, rendering="cached")
    return runs


def hard_ids(test: Benchmark, sims: dict[str, float], threshold: float = HARD_THRESHOLD) -> list[str]:
    return [it.id for it in test if sims[it.id] < threshold]


def _run_macro_f1(run: Run, bench: Benchmark, ids: list[str] | None = None) -> float:
    by = bench.by_id()
    ids = ids or [i.id for i in bench]
    return macro_f1([by[i].label for i in ids], [run.answers[i] for i in ids])


def profile(model_key: str = "M_comm_sonnet5", root: Path = ROOT, *, retrieval_methods=("char3", "bm25")) -> ArtifactRiskProfile:
    return _profile(model_key, Path(root), tuple(retrieval_methods))


@lru_cache(maxsize=16)
def _profile(model_key: str, root: Path, retrieval_methods: tuple[str, ...]) -> ArtifactRiskProfile:
    test, dictionary = load(root)
    runs = cached_runs(test, root)
    llm = runs[model_key]
    signals: dict[str, Signal] = {}
    signals["format_sensitivity"] = Signal("format_sensitivity", None,
                                           note="single prompt in the cached run; needs a live re-run under standard_renderings")
    maj = runs["B_majority"]
    signals["partial_input_accuracy"] = Signal(
        "partial_input_accuracy", maj.accuracy - 1 / len(CHAPTERS),
        details={"proxy": "majority-class baseline stands in for options-only input (no premises exist to withhold)",
                 "majority_accuracy": maj.accuracy, "chance": 1 / len(CHAPTERS)},
        note="proxy from the label prior, not a model run")
    signals["label_prior_skew"] = label_prior_skew(test, llm)
    signals["retrieval_dominance"] = retrieval_dominance(dictionary, test, llm, methods=retrieval_methods)
    # strict decoding (output must be exactly a chapter name) vs the paper's lenient parse
    strict = Run.from_answers(llm.model, test, {i: (r if r in CHAPTERS else "") for i, r in llm.raw.items()}, exact_match,
                              raw=llm.raw, rendering="strict")
    signals["semantic_validity_gap"] = constraint_gain_from_runs(strict, llm, test, sem_judge=lenient_choice)
    signals["abstention_failure"] = Signal("abstention_failure", None,
                                           note="no unanswerable probes in the cached run; interventions.make_unanswerable for a live run")
    return ArtifactRiskProfile("cause_of_death", llm.model, signals).apply_gate()


def interventions(model_key: str = "M_comm_sonnet5", root: Path = ROOT, *, thresholds=(0.4, 0.5, 0.6, 0.7, 0.8),
                  cap: int = 10, seed: int = 0) -> dict:
    """Offline interventions from cached predictions. Returns the results dict the ledger reads."""
    import random

    test, dictionary = load(root)
    runs = cached_runs(test, root)
    llm, fuzzy = runs[model_key], runs["B_fuzzy"]
    sims = dictsim(test, root)
    ids = [i.id for i in test]
    out: dict = {"model": llm.model, "n": len(ids)}

    def block(sub: list[str]) -> dict:
        l, f = llm.accuracy_on(sub), fuzzy.accuracy_on(sub)
        return {"n": len(sub), "llm_accuracy": l, "retrieval_accuracy": f, "gap": l - f,
                "llm_macro_f1": _run_macro_f1(llm, test, sub), "retrieval_macro_f1": _run_macro_f1(fuzzy, test, sub),
                "sim_correctness_corr": point_biserial([sims[i] for i in sub], [llm.correct[i] for i in sub]) if len(sub) > 2 else None,
                "pairwise": {"both": sum(1 for i in sub if llm.correct[i] and fuzzy.correct[i]),
                             "llm_only": sum(1 for i in sub if llm.correct[i] and not fuzzy.correct[i]),
                             "retrieval_only": sum(1 for i in sub if fuzzy.correct[i] and not llm.correct[i]),
                             "neither": sum(1 for i in sub if not llm.correct[i] and not fuzzy.correct[i])}}

    before = block(ids)
    sweep = {}
    for thr in thresholds:
        kept = [i for i in ids if sims[i] < thr]
        if len(kept) < 5:
            continue
        after = block(kept)
        sweep[str(thr)] = {"before": before, "after": after, "n_kept": len(kept), "n_dropped": len(ids) - len(kept),
                           "llm": {"accuracy": {"delta": after["llm_accuracy"] - before["llm_accuracy"]}},
                           "retrieval": {"accuracy": {"delta": after["retrieval_accuracy"] - before["retrieval_accuracy"]}},
                           "gap": {"llm_minus_retrieval": {"delta": after["gap"] - before["gap"]}},
                           "retrieval_dominance": {"sim_correctness_corr": after["sim_correctness_corr"]},
                           "manipulation_check": (after["sim_correctness_corr"] or 0.0) < 0.15}
    reg = sweep[str(HARD_THRESHOLD)]
    out["leakage_resplit"] = {**reg, "registered_threshold": HARD_THRESHOLD, "sweep": sweep}

    # cap-based rebalancing: no class keeps more than `cap` items
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = {}
    for it in test:
        by_label.setdefault(it.label, []).append(it.id)
    kept = []
    for lab, members in sorted(by_label.items()):
        members = list(members)
        rng.shuffle(members)
        kept.extend(members[:cap])
    after = block(kept)
    lab_before = Counter(i.label for i in test)
    lab_after = Counter(test.by_id()[i].label for i in kept)
    uni = {c: 1 / len(lab_before) for c in lab_before}
    out["rebalance_and_filter"] = {
        "cap": cap, "n_after": len(kept), "before": before, "after": after,
        "llm": {"accuracy": {"delta": after["llm_accuracy"] - before["llm_accuracy"]},
                "macro_f1": {"delta": after["llm_macro_f1"] - before["llm_macro_f1"]}},
        "retrieval": {"accuracy": {"delta": after["retrieval_accuracy"] - before["retrieval_accuracy"]}},
        "gap": {"llm_minus_retrieval": {"delta": after["gap"] - before["gap"]}},
        "label_prior_skew": {"tv_label_vs_uniform_before": tv_distance({k: v / len(ids) for k, v in lab_before.items()}, uni),
                             "tv_label_vs_uniform": tv_distance({k: v / len(kept) for k, v in lab_after.items()}, uni)},
    }
    prof = profile(model_key, root)
    svg = prof.signals["semantic_validity_gap"].details
    out["constraint_ablation"] = {"em": {"constrained_minus_free": svg["constraint_gain"]},
                                  "sem": {"constrained_minus_free": svg["semantic_gain"]}, "svg": svg["constraint_gain"] - svg["semantic_gain"],
                                  "note": "strict decoding vs the paper's lenient parse on the same cached outputs"}
    probe = root / "probe_summary.json"
    if probe.exists():
        out["memorization_probe"] = json.loads(probe.read_text())
    out["none"] = {"retrieval_dominance": prof.signals["retrieval_dominance"].to_dict()["details"]}
    return out


def published(root: Path = ROOT) -> dict:
    return json.loads((root / "summary.json").read_text())
