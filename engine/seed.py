"""Reproducibility helpers.

Wraps NumPy RNG creation so the same (input, seed) pair produces byte-identical
Monte Carlo output every time.

Seeds are always returned/stored as *strings* so 128-bit entropy values survive
JSON round-trips through JavaScript (which cannot represent integers > 2^53
exactly as IEEE-754 numbers).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _seed_to_int(seed: int | str) -> int:
    """Convert a user/API seed (int or decimal string) to a Python int for NumPy."""
    if isinstance(seed, int):
        return int(seed)
    s = str(seed).strip()
    if not s:
        raise ValueError("seed string is empty")
    # Accept plain decimal integers (including values > 2^53)
    return int(s, 10)


def make_rng(seed: int | str | None) -> Tuple[np.random.Generator, str]:
    """Create a seeded NumPy Generator.

    When *seed* is None an entropy-based seed is drawn from SeedSequence and
    returned alongside the Generator so callers can persist it for
    reproducibility. The returned seed is always a decimal string.
    """
    if seed is None:
        entropy = int(np.random.SeedSequence().entropy)
        seed_str = str(entropy)
        seed_int = entropy
    else:
        seed_str = str(seed).strip()
        seed_int = _seed_to_int(seed_str)
    return np.random.default_rng(seed_int), seed_str
