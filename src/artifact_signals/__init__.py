"""artifact_signals: diagnostic instrument for benchmark artifacts in LLM reasoning evaluation.

The package computes an Artifact Risk Profile (five diagnostics) for a benchmark,
applies targeted de-artifacting interventions with manipulation checks, and scores a
hash-stamped ledger of pre-registered predictions against the observed changes.

No hard dependencies. Optional extras add rapidfuzz, HF transformers, and an
OpenAI-compatible API client.
"""

from .data import ABSTAIN, Benchmark, Item
from .ledger import Ledger, Prediction, score_ledger
from .signals import ArtifactRiskProfile, Signal, compute_profile

__all__ = [
    "ABSTAIN",
    "ArtifactRiskProfile",
    "Benchmark",
    "Item",
    "Ledger",
    "Prediction",
    "Signal",
    "compute_profile",
    "score_ledger",
]
__version__ = "0.1.0"
