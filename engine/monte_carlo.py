"""Monte Carlo robustness engine — blueprint §5.

Three complementary tests, all driven by a single seeded
``numpy.random.Generator`` for reproducibility:

 * Test A — i.i.d. bootstrap with replacement (naive per-trade)
 * Test B — permutation (no replacement) → path-dependency check
 * Test C — stationary block bootstrap (Politis–White, L ≈ sqrt(N))

Plus ``correlation_flag`` which triggers when Test C's 5th-percentile MDD
is more than 1.5× worse than Test A's — indicating broken i.i.d.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .models import (
    MCPercentiles,
    MCTestResult,
    MonteCarloResult,
    Trade,
)

EPS = 1e-9
PERCENTILES = [0, 5, 25, 50, 75, 95, 100]  # p0, p5, p25, p50, p75, p95, p100
DEFAULT_ITERATIONS = 1000


# --------------------------------------------------------------------------- #
# Internal helpers — single synthetic-run metrics
# --------------------------------------------------------------------------- #


def _run_metrics(pnl: np.ndarray, start_balance: float) -> Tuple[float, float, float, float]:
    """Return (TR, MDD, Sharpe-like, Profit Factor) for one synthetic series.

    Computed as scalars; no heavy-weight Metrics object needed.
    """
    n = pnl.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    equity = np.empty(n + 1, dtype=float)
    equity[0] = start_balance
    equity[1:] = start_balance + np.cumsum(pnl)
    end_bal = float(equity[-1])
    tr = (end_bal - start_balance) / max(EPS, start_balance)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    mdd = float(np.min(dd))

    # Simple per-trade returns vs start balance
    bal_before = equity[:-1]
    bal_before_safe = np.where(np.abs(bal_before) > EPS, bal_before, 1.0)
    R = pnl / bal_before_safe
    mu = float(np.mean(R))
    sigma = float(np.std(R, ddof=1)) if n >= 2 else 0.0
    # Annualisation factor: skip if very few trades — match metrics.py behaviour
    sr = (mu / sigma) if sigma > EPS else 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    sw = float(np.sum(wins))
    sl = float(np.sum(np.abs(losses)))
    pf = sw / max(EPS, sl)
    pf = min(pf, 50.0)

    return tr, mdd, sr, pf


def _dist_summary(
    iterations: int,
    tr_arr: np.ndarray,
    mdd_arr: np.ndarray,
    sr_arr: np.ndarray,
    pf_arr: np.ndarray,
    *,
    include_rr: bool = True,
    actual_mdd: float | None = None,
) -> MCTestResult:
    """Roll percentile arrays + RR / stability / 2x-MDD-probability flags into
    a :class:`MCTestResult`."""
    pct = PERCENTILES
    percentiles = MCPercentiles(
        total_return=[float(np.percentile(tr_arr, p)) for p in pct],
        max_drawdown=[float(np.percentile(mdd_arr, p)) for p in pct],
        sharpe_ratio=[float(np.percentile(sr_arr, p)) for p in pct],
        profit_factor=[float(np.percentile(pf_arr, p)) for p in pct],
    )
    rr = None
    stab = None
    mdd_2x_prob = None
    flagged = None
    if include_rr and iterations > 0:
        rr = float(np.mean(tr_arr > 0))
        stab = float(np.mean(sr_arr > 1.0))
    if actual_mdd is not None and iterations > 0:
        threshold = 2.0 * actual_mdd  # actual_mdd is negative so 2x "worse"
        mdd_2x_prob = float(np.mean(mdd_arr < threshold))
    return MCTestResult(
        iterations=iterations,
        percentiles=percentiles,
        robustness_rate=rr,
        sharpe_stability=stab,
        mdd_2x_probability=mdd_2x_prob,
        flagged_worse_than_a=flagged,
    )


# --------------------------------------------------------------------------- #
# Test A — naive i.i.d. bootstrap with replacement
# --------------------------------------------------------------------------- #


def _test_a(
    pnl: np.ndarray,
    start_balance: float,
    rng: np.random.Generator,
    iterations: int,
) -> MCTestResult:
    n = pnl.size
    tr = np.empty(iterations, dtype=float)
    mdd = np.empty(iterations, dtype=float)
    sr = np.empty(iterations, dtype=float)
    pf = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        sampled = pnl[idx]
        tr[i], mdd[i], sr[i], pf[i] = _run_metrics(sampled, start_balance)
    return _dist_summary(iterations, tr, mdd, sr, pf, include_rr=True)


# --------------------------------------------------------------------------- #
# Test B — shuffled order (no replacement)
# --------------------------------------------------------------------------- #


def _test_b(
    pnl: np.ndarray,
    start_balance: float,
    actual_mdd: float,
    rng: np.random.Generator,
    iterations: int,
) -> MCTestResult:
    n = pnl.size
    tr = np.empty(iterations, dtype=float)
    mdd = np.empty(iterations, dtype=float)
    sr = np.empty(iterations, dtype=float)
    pf = np.empty(iterations, dtype=float)
    for i in range(iterations):
        perm = rng.permutation(n)
        permuted = pnl[perm]
        tr[i], mdd[i], sr[i], pf[i] = _run_metrics(permuted, start_balance)
    return _dist_summary(iterations, tr, mdd, sr, pf, include_rr=True, actual_mdd=actual_mdd)


# --------------------------------------------------------------------------- #
# Test C — stationary block bootstrap (Politis–White)
# --------------------------------------------------------------------------- #


def _stationary_block_sample(
    n: int,
    L_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n`` integer indices 0..n-1 drawn as contiguous blocks with
    geometric length distribution of mean *L_mean*. Wrap-around circular."""
    if n == 0:
        return np.empty(0, dtype=int)
    L_mean = max(1.0, float(L_mean))
    p = 1.0 / L_mean  # probability a new block starts at each position
    out = np.empty(n, dtype=int)
    # Start index for the first block
    cur = int(rng.integers(0, n))
    for i in range(n):
        if i > 0 and rng.random() < p:
            cur = int(rng.integers(0, n))  # new block start
        out[i] = cur
        cur = (cur + 1) % n
    return out


def _test_c(
    pnl: np.ndarray,
    start_balance: float,
    rng: np.random.Generator,
    iterations: int,
) -> MCTestResult:
    n = pnl.size
    L = float(np.sqrt(n)) if n >= 2 else 1.0
    tr = np.empty(iterations, dtype=float)
    mdd = np.empty(iterations, dtype=float)
    sr = np.empty(iterations, dtype=float)
    pf = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = _stationary_block_sample(n, L, rng)
        sampled = pnl[idx]
        tr[i], mdd[i], sr[i], pf[i] = _run_metrics(sampled, start_balance)
    return _dist_summary(iterations, tr, mdd, sr, pf, include_rr=True)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run_monte_carlo(
    trades: List[Trade],
    rng: np.random.Generator,
    *,
    start_balance: float = 10_000.0,
    iterations: int = DEFAULT_ITERATIONS,
    actual_max_drawdown: float,
) -> MonteCarloResult:
    """Run all three MC tests, compute correlation_flag, and return aggregate.

    ``effective_mc_source`` is "test_c" if the correlation flag is raised;
    otherwise "test_a" — used downstream by scoring.py §6 category 4.
    """
    pnl = np.array([t.pnl for t in trades], dtype=float)

    ta = _test_a(pnl, start_balance, rng, iterations)
    tb = _test_b(pnl, start_balance, actual_max_drawdown, rng, iterations)
    tc = _test_c(pnl, start_balance, rng, iterations)

    # Correlation flag: Test C's 5th-percentile MDD is more than 1.5× worse
    # (more negative) than Test A's 5th-percentile MDD.
    ta_mdd_p5 = ta.percentiles.max_drawdown[1]   # PERCENTILES[1] = 5
    tc_mdd_p5 = tc.percentiles.max_drawdown[1]
    # Avoid division by zero; use magnitude comparison
    if abs(ta_mdd_p5) < EPS:
        correlation_flag = abs(tc_mdd_p5) > 0.005  # 0.5% any material difference
    else:
        correlation_flag = (abs(tc_mdd_p5) > 1.5 * abs(ta_mdd_p5))

    # Annotate tc
    tc.flagged_worse_than_a = bool(correlation_flag)

    effective = "test_c" if correlation_flag else "test_a"

    return MonteCarloResult(
        test_a=ta,
        test_b=tb,
        test_c=tc,
        correlation_flag=bool(correlation_flag),
        effective_mc_source=effective,  # type: ignore[arg-type]
    )
