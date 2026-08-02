"""Reproducibility helpers.

Wraps NumPy RNG creation so the same (input, seed) pair produces byte-identical
Monte Carlo output every time.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def make_rng(seed: int | None) -> Tuple[np.random.Generator, int]:
    """Create a seeded NumPy Generator.

    When *seed* is None an entropy-based seed is drawn from SeedSequence and
    returned alongside the Generator so callers can persist it for
    reproducibility.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    return np.random.default_rng(seed), int(seed)
