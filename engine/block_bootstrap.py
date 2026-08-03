"""Module 4 — Moving-block bootstrap on the closed-trade PnL series.

Distinct from MC Test C (stationary / Politis–White geometric blocks):
here block length L is fixed and overlapping blocks of length L are drawn
with replacement, then concatenated to length N. Preserves short-range
serial dependence in the trade list without needing price bars.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .models import MCPercentiles, MovingBlockResult, Trade
from .monte_carlo import EPS, PERCENTILES, _run_metrics

DEFAULT_MB_ITERATIONS = 1000


def _default_block_length(n: int) -> int:
    """L ≈ √N, clamped to [1, max(1, N-1)]."""
    if n <= 1:
        return 1
    L = int(max(1, round(np.sqrt(n))))
    return min(L, n)


def moving_block_sample(n: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Indices 0..n-1 formed by concatenating fixed-length overlapping blocks.

    Blocks wrap circularly so every start position is valid.
    """
    if n == 0:
        return np.empty(0, dtype=int)
    L = max(1, min(int(L), n))
    out = np.empty(n, dtype=int)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        take = min(L, n - filled)
        for j in range(take):
            out[filled + j] = (start + j) % n
        filled += take
    return out


def run_moving_block_bootstrap(
    trades: List[Trade],
    rng: np.random.Generator,
    *,
    start_balance: float = 10_000.0,
    iterations: int = DEFAULT_MB_ITERATIONS,
    block_length: Optional[int] = None,
    avg_trades_per_year: float = 1.0,
    rf_annual: float = 0.04,
    iid_robustness_rate: Optional[float] = None,
    iid_sharpe_stability: Optional[float] = None,
) -> MovingBlockResult:
    """Module 4 entry point.

    ``iid_*`` come from MC Test A when available so the dependence gap
    (iid RR − block RR) can be reported as a bounded overfitting / path-
    dependence signal.
    """
    pnl = np.array(
        [t.capped_pnl if t.capped_pnl is not None else t.pnl for t in trades],
        dtype=float,
    )
    n = pnl.size
    L = int(block_length) if block_length is not None else _default_block_length(n)
    L = max(1, min(L, max(1, n)))

    if n == 0 or iterations <= 0:
        empty = [0.0] * len(PERCENTILES)
        return MovingBlockResult(
            iterations=0,
            block_length=L,
            percentiles=MCPercentiles(
                total_return=empty, max_drawdown=empty,
                sharpe_ratio=empty, profit_factor=empty,
            ),
            iid_robustness_rate=iid_robustness_rate,
            iid_sharpe_stability=iid_sharpe_stability,
        )

    atpy = float(avg_trades_per_year) if avg_trades_per_year > EPS else float(n)
    tr = np.empty(iterations, dtype=float)
    mdd = np.empty(iterations, dtype=float)
    sr = np.empty(iterations, dtype=float)
    pf = np.empty(iterations, dtype=float)

    for i in range(iterations):
        idx = moving_block_sample(n, L, rng)
        sampled = pnl[idx]
        tr[i], mdd[i], sr[i], pf[i] = _run_metrics(
            sampled, start_balance, atpy=atpy, rf_annual=rf_annual,
        )

    pct = PERCENTILES
    percentiles = MCPercentiles(
        total_return=[float(np.percentile(tr, p)) for p in pct],
        max_drawdown=[float(np.percentile(mdd, p)) for p in pct],
        sharpe_ratio=[float(np.percentile(sr, p)) for p in pct],
        profit_factor=[float(np.percentile(pf, p)) for p in pct],
    )
    rr = float(np.mean(tr > 0))
    stab = float(np.mean(sr > 1.0))
    mean_sr = float(np.mean(sr))

    gap = None
    if iid_robustness_rate is not None:
        gap = float(iid_robustness_rate) - rr

    return MovingBlockResult(
        iterations=iterations,
        block_length=L,
        percentiles=percentiles,
        robustness_rate=rr,
        sharpe_stability=stab,
        mean_sharpe=mean_sr,
        iid_robustness_rate=iid_robustness_rate,
        iid_sharpe_stability=iid_sharpe_stability,
        block_vs_iid_rr_gap=gap,
    )
