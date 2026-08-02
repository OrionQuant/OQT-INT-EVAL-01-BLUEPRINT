"""Strategy Evaluation Tool - evaluation engine package.

Sub-modules are lazily importable; ``from engine.ingestion import ingest_file``
works without pulling metrics/scoring/storage into memory.
"""

__all__ = [
    "models",
    "seed",
    "ingestion",
    "cleaning",
    "metrics",
    "monte_carlo",
    "scoring",
    "storage",
]
