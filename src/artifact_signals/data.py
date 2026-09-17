"""Benchmark and item containers with JSONL I/O.

An Item is task-agnostic: premises (context), a question, optional answer options,
and a gold label. For choice tasks the label is the option *text*, never a position
token, so relabelling and permutation renderings cannot change what counts as correct.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

ABSTAIN = "insufficient information"


@dataclass
class Item:
    id: str
    question: str
    label: str
    premises: list[str] = field(default_factory=list)
    options: list[str] | None = None
    ood: bool = False
    source_id: str | None = None
    meta: dict = field(default_factory=dict)

    def text(self) -> str:
        return " ".join([*self.premises, self.question])

    @property
    def cluster(self) -> str:
        return self.source_id or self.id

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(
            id=str(d["id"]),
            question=d["question"],
            label=d["label"],
            premises=list(d.get("premises", [])),
            options=list(d["options"]) if d.get("options") is not None else None,
            ood=bool(d.get("ood", False)),
            source_id=d.get("source_id"),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Benchmark:
    name: str
    items: list[Item]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def __getitem__(self, i: int) -> Item:
        return self.items[i]

    @property
    def task(self) -> str:
        return "choice" if self.items and all(i.options for i in self.items) else "freeform"

    def by_id(self) -> dict[str, Item]:
        return {i.id: i for i in self.items}

    def label_space(self) -> list[str]:
        labels: list[str] = []
        for it in self.items:
            for cand in (it.options or [it.label]):
                if cand not in labels:
                    labels.append(cand)
        return labels

    def in_distribution(self) -> list[Item]:
        return [i for i in self.items if not i.ood]

    def ood_items(self) -> list[Item]:
        return [i for i in self.items if i.ood]

    def subset(self, ids: set[str] | list[str], name: str | None = None) -> "Benchmark":
        keep = set(ids)
        return Benchmark(name or self.name, [i for i in self.items if i.id in keep])

    def split(self, frac: float, seed: int = 0, names: tuple[str, str] = ("train", "test")) -> tuple["Benchmark", "Benchmark"]:
        """Cluster-aware split: items sharing a source_id land on the same side."""
        rng = random.Random(seed)
        clusters = sorted({i.cluster for i in self.items})
        rng.shuffle(clusters)
        cut = int(round(frac * len(clusters)))
        left = set(clusters[:cut])
        a = [i for i in self.items if i.cluster in left]
        b = [i for i in self.items if i.cluster not in left]
        return Benchmark(f"{self.name}-{names[0]}", a), Benchmark(f"{self.name}-{names[1]}", b)

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> "Benchmark":
        path = Path(path)
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(Item.from_dict(json.loads(line)))
        return cls(name or path.stem, items)


# --- optional loaders for public benchmarks (require `datasets`) ---------------


def load_folio(split: str = "validation", limit: int | None = None) -> Benchmark:
    """FOLIO as a 3-way choice task (True / False / Uncertain)."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("yale-nlp/FOLIO", split=split)
    opts = ["True", "False", "Uncertain"]
    items = []
    for k, row in enumerate(ds):
        if limit and k >= limit:
            break
        label = str(row["label"]).strip().capitalize()
        if label not in opts:
            continue
        items.append(
            Item(
                id=f"folio-{split}-{row.get('example_id', k)}",
                premises=[p.strip() for p in str(row["premises"]).split("\n") if p.strip()],
                question=f"Is the following conclusion true, false, or uncertain given the premises? {row['conclusion']}",
                options=opts,
                label=label,
                source_id=str(row.get("story_id", k)),
            )
        )
    return Benchmark("folio", items)


def load_logiqa(split: str = "validation", limit: int | None = None) -> Benchmark:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("lucasmccabe/logiqa", split=split)
    items = []
    for k, row in enumerate(ds):
        if limit and k >= limit:
            break
        options = [str(o) for o in row["options"]]
        items.append(
            Item(
                id=f"logiqa-{split}-{k}",
                premises=[str(row["context"])],
                question=str(row["query"]),
                options=options,
                label=options[int(row["correct_option"])],
                source_id=f"logiqa-{split}-{k}",
            )
        )
    return Benchmark("logiqa", items)
