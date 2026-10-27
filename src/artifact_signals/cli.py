"""Command-line entry points.

  artifact-signals synth     --n 400 --task truth --ood-frac 0.1 --out synth.jsonl
  artifact-signals diagnose  --benchmark test.jsonl [--train train.jsonl] --model dummy:oracle:0.8 --out profile.json
  artifact-signals ledger hash  ledger.json
  artifact-signals ledger check  ledger.json outputs/paper12/results.json   (freeze readiness)
  artifact-signals ledger freeze ledger.json outputs/paper12/results.json   (status -> prospective, stamp)
  artifact-signals ledger score ledger.json results.json [--void P07 ...]
"""

from __future__ import annotations

import argparse
import json
import sys

from .data import PUBLIC_SOURCES, Benchmark
from .evaluate import Cache
from .judge import JUDGES
from .ledger import Ledger, format_score, freeze, freeze_problems, score_ledger
from .modal import FORMULA_REGEX
from .models import from_spec
from .render import standard_renderings
from .signals import compute_profile
from .stats import N_BOOT
from .synth import generate


def cmd_synth(a: argparse.Namespace) -> int:
    b = generate(a.n, seed=a.seed, task=a.task, ood_frac=a.ood_frac, novel_names=not a.plain_names)
    b.to_jsonl(a.out)
    labels = {}
    for i in b:
        labels[i.label if not i.ood else "<abstain>"] = labels.get(i.label if not i.ood else "<abstain>", 0) + 1
    print(f"wrote {len(b)} items to {a.out}: {labels}")
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    from_rows, load = PUBLIC_SOURCES[a.source]
    if a.rows:
        rows = [json.loads(line) for line in open(a.rows, encoding="utf-8") if line.strip()]
        n_rows = min(len(rows), a.limit) if a.limit else len(rows)  # the loaders read limit=0 as no limit
        b = from_rows(rows, a.split, a.limit)
    else:
        b = load(a.split, a.limit)
        n_rows = None
    b.to_jsonl(a.out)
    skipped = f" (skipped {n_rows - len(b)} rows with an unusable label)" if n_rows is not None else ""
    print(f"wrote {len(b)} items to {a.out}{skipped}")
    return 0


def cmd_diagnose(a: argparse.Namespace) -> int:
    bench = Benchmark.from_jsonl(a.benchmark)
    train = Benchmark.from_jsonl(a.train) if a.train else None
    gold = {i.id: i.label for i in bench}
    model = from_spec(a.model, gold=gold, train=train.items if train else None)
    judge = JUDGES[a.judge]
    sem = JUDGES[a.semantic] if a.semantic else None
    regex = FORMULA_REGEX if bench.task == "freeform" and a.semantic and a.semantic.startswith("formula") else None
    prof = compute_profile(model, bench, judge, train=train, sem_judge=sem, renderings=standard_renderings(a.seed, regex),
                           fs_gate=a.fs_gate, n_boot=a.n_boot, seed=a.seed, cache=Cache(a.cache) if a.cache else None)
    print(prof.summary())
    if a.out:
        prof.to_json(a.out)
        print(f"wrote {a.out}")
    return 0


def _refuse_if_stamped(path: str) -> None:
    if "sha256" in json.load(open(path, encoding="utf-8")):
        sys.exit(f"{path} is already stamped; a stamped ledger is never rewritten")


def cmd_ledger(a: argparse.Namespace) -> int:
    if a.action == "hash":
        led = Ledger.load(a.ledger, verify=False)
        print(led.sha256())
        if a.stamp:
            _refuse_if_stamped(a.ledger)
            led.save(a.ledger)
            print(f"stamped {a.ledger}")
        return 0
    if a.action in ("check", "freeze"):
        dev = json.load(open(a.results))
        if a.action == "freeze":
            _refuse_if_stamped(a.ledger)
        led = Ledger.load(a.ledger, verify=False)
        problems = freeze_problems(led, dev)
        for p in problems:
            print(f"  not ready: {p}")
        if problems:
            print(f"{a.ledger}: not freeze-ready ({len(problems)} problem(s))")
            return 1
        if a.action == "check":
            print(f"{a.ledger}: freeze-ready ({len(led.predictions)} predictions)")
            return 0
        h = freeze(led, dev).save(a.ledger)
        print(f"froze {a.ledger}: status prospective, sha256 {h}")
        print("next: commit, tag, deposit the file and this hash; only then run any model on a test-set benchmark")
        return 0
    led = Ledger.load(a.ledger)
    results = json.load(open(a.results))
    s = score_ledger(led, results, voided=set(a.void or []))
    print(format_score(s))
    if a.out:
        json.dump(s, open(a.out, "w"), indent=2)
    return 0


def cmd_paper12(a: argparse.Namespace) -> int:
    from pathlib import Path

    from .paper12 import run_all

    out = a.out or ("outputs/paper12_folded" if a.fold_injury else "outputs/paper12")  # never overwrite the committed outputs
    res = run_all(Path(out), fold_injury=a.fold_injury)
    print(res["report"] if a.report else res["report"].split("## Ledger score")[0][-1200:])
    print(f"wrote {out}/profiles.json, results.json, ledger_score.json, RESULTS_offline.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="artifact-signals")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate SynthModal-Control")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--task", choices=["truth", "generate"], default="truth")
    s.add_argument("--ood-frac", type=float, default=0.0)
    s.add_argument("--plain-names", action="store_true", help="use w0/p/q instead of novel random names")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_synth)

    f = sub.add_parser("fetch", help="convert FOLIO or LogiQA to benchmark JSONL (downloads unless --rows is given)")
    f.add_argument("--source", choices=sorted(PUBLIC_SOURCES), required=True)
    f.add_argument("--split", default="validation")
    f.add_argument("--rows", help="local JSONL of dataset rows in the Hugging Face field layout; no download")
    f.add_argument("--limit", type=int)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fetch)

    d = sub.add_parser("diagnose", help="compute the Artifact Risk Profile")
    d.add_argument("--benchmark", required=True)
    d.add_argument("--train")
    d.add_argument("--model", required=True)
    d.add_argument("--judge", choices=list(JUDGES), default="exact")
    d.add_argument("--semantic", choices=list(JUDGES), help="semantic judge for the constraint-gain signal")
    d.add_argument("--fs-gate", type=float, default=0.03)
    d.add_argument("--n-boot", type=int, default=N_BOOT)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--cache")
    d.add_argument("--out")
    d.set_defaults(fn=cmd_diagnose)

    l = sub.add_parser("ledger", help="hash or score a prediction ledger")
    l.add_argument("action", choices=["hash", "score", "check", "freeze"],
                   help="check / freeze take the development-set results.json as the second argument")
    l.add_argument("ledger")
    l.add_argument("results", nargs="?")
    l.add_argument("--stamp", action="store_true", help="with hash: write the sha256 into the file")
    l.add_argument("--void", nargs="*", help="prediction ids voided by a failed invariance check")
    l.add_argument("--out")
    l.set_defaults(fn=cmd_ledger)

    q = sub.add_parser("paper12", help="offline Phase 1 on papers 1 and 2 from cached outputs")
    q.add_argument("--out", help="default outputs/paper12, or outputs/paper12_folded with --fold-injury")
    q.add_argument("--report", action="store_true", help="print the full markdown report")
    q.add_argument("--fold-injury", action="store_true",
                   help="score cause-of-death with Injury folded into ExternalCause (COD v2 secondary scoring)")
    q.set_defaults(fn=cmd_paper12)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd == "ledger" and a.action in ("score", "check", "freeze") and not a.results:
        p.error(f"ledger {a.action} requires a results file")
    if a.cmd == "diagnose" and a.model.startswith("cached:") and not a.cache:
        p.error("a cached: model replays saved outputs, so it needs --cache DIR")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
