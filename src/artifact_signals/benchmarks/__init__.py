"""Adapters for the two source benchmarks of the research program.

Each adapter loads the benchmark into the package's Item/Benchmark model, rebuilds Runs from
the cached model outputs shipped with the source repository, reproduces the published
numbers, and computes the Artifact Risk Profile and the interventions offline.
"""

from . import cod, kripke

__all__ = ["cod", "kripke"]
