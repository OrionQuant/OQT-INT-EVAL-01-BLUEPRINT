"""Module 3 — CUSUM / regime checks on the realised per-trade return series.

Operates exclusively on the closed-trade return vector R_i. No price bars,
no strategy re-fit — report-analysis layer only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .models import CUSUMStat, RegimeCheck

EPS = 1e-9
# Standardised two-sided CUSUM allowance / decision threshold (Page / Montgomery)
DEFAULT_ALLOWANCE_K = 0.5
DEFAULT_THRESHOLD_H = 4.0
# Half-sample gap flag: |Δμ| > GAP_SIGMA_MULT * pooled SE, or |ΔSR| > SHARPE_GAP
GAP_SIGMA_MULT = 2.0
SHARPE_GAP_ABS = 0.5


def _raw_sharpe(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    return mu / sigma if sigma > EPS else 0.0


def _two_sided_cusum(
    x: np.ndarray,
    *,
    k: float = DEFAULT_ALLOWANCE_K,
    h: float = DEFAULT_THRESHOLD_H,
) -> CUSUMStat:
    """Two-sided CUSUM on a (approximately) unit-variance series.

    After each threshold crossing the statistic is reset to 0 so successive
    regime shifts can be counted.
    """
    n = x.size
    if n == 0:
        return CUSUMStat(threshold_used=float(h))

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if n >= 2 else 0.0
    if sigma < EPS:
        # Degenerate flat series — no detectable shift
        return CUSUMStat(threshold_used=float(h))

    z = (x - mu) / sigma
    s_pos = 0.0
    s_neg = 0.0
    peak_pos = 0.0
    peak_neg = 0.0  # most negative magnitude (≤ 0)
    hits = 0

    for zi in z:
        s_pos = max(0.0, s_pos + float(zi) - k)
        s_neg = min(0.0, s_neg + float(zi) + k)
        peak_pos = max(peak_pos, s_pos)
        peak_neg = min(peak_neg, s_neg)
        if s_pos > h:
            hits += 1
            s_pos = 0.0
        if s_neg < -h:
            hits += 1
            s_neg = 0.0

    return CUSUMStat(
        cumsum_pos_peak=float(peak_pos),
        cumsum_neg_peak=float(peak_neg),
        threshold_hit=hits > 0 or peak_pos > h or peak_neg < -h,
        n_regime_shifts_detected=int(hits),
        threshold_used=float(h),
    )


def run_regime_check(
    returns: np.ndarray,
    *,
    k: float = DEFAULT_ALLOWANCE_K,
    h: float = DEFAULT_THRESHOLD_H,
) -> RegimeCheck:
    """Module 3 entry point: full-series CUSUM + half-sample stability."""
    R = np.asarray(returns, dtype=float)
    n = R.size
    if n < 4:
        # Too short for a meaningful half-split; still run full CUSUM if possible
        full = _two_sided_cusum(R, k=k, h=h) if n >= 2 else CUSUMStat(threshold_used=float(h))
        return RegimeCheck(
            cusum_full=full,
            regime_unstable_flag=bool(full.threshold_hit),
        )

    mid = n // 2
    first = R[:mid]
    second = R[mid:]

    cusum_full = _two_sided_cusum(R, k=k, h=h)
    cusum_1 = _two_sided_cusum(first, k=k, h=h)
    cusum_2 = _two_sided_cusum(second, k=k, h=h)

    mu1 = float(np.mean(first))
    mu2 = float(np.mean(second))
    mean_gap = mu1 - mu2

    s1 = float(np.std(first, ddof=1)) if first.size >= 2 else 0.0
    s2 = float(np.std(second, ddof=1)) if second.size >= 2 else 0.0
    pooled_var = (
        ((first.size - 1) * s1 ** 2 + (second.size - 1) * s2 ** 2)
        / max(1, first.size + second.size - 2)
    )
    se = float(np.sqrt(pooled_var * (1.0 / first.size + 1.0 / second.size))) if pooled_var > 0 else 0.0

    sr1 = _raw_sharpe(first)
    sr2 = _raw_sharpe(second)
    sharpe_gap = sr1 - sr2

    gap_flag = (se > EPS and abs(mean_gap) > GAP_SIGMA_MULT * se) or (abs(sharpe_gap) > SHARPE_GAP_ABS)
    unstable = bool(cusum_full.threshold_hit or gap_flag)

    return RegimeCheck(
        cusum_first_half=cusum_1,
        cusum_second_half=cusum_2,
        cusum_full=cusum_full,
        mean_return_gap=float(mean_gap),
        sharpe_half_gap=float(sharpe_gap),
        regime_unstable_flag=unstable,
    )
