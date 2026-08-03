"""Weighted scoring model (0–100) — blueprint §6.

Pipeline:
 1. §6.4 Hard knockouts — instant 0 if N<30 or PF<1.0 or |MDD|>35 %.
 2. §6.3 Per-metric sigmoid/piecewise normalisation to [0, 10] subscores.
 3. §6.2 Weighted sum across categories → raw_score ∈ [0, 10].
 4. §6.5 Multiplicative penalties (P ∈ [0.5, 1.0]) on top of raw.
 5. §6.6 final_score = clamp(round(10 * raw * P), 0, 100) and label.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .models import (
    CategoryScores,
    Metrics,
    MonteCarloResult,
    MCTestResult,
    PenaltiesApplied,
    ScoringResult,
)

EPS = 1e-9


# --------------------------------------------------------------------------- #
# §6.3 Normalisation primitives
# --------------------------------------------------------------------------- #


def _sigmoid(x: float, x0: float, k: float) -> float:
    """Sigmoid centred at x0 with steepness k. Output ∈ (0, 10)."""
    # Prevent overflow in exp
    z = k * (x - x0)
    z = max(-500.0, min(500.0, z))
    return 10.0 / (1.0 + np.exp(-z))


def _reverse_sigmoid(x: float, x0: float, k: float) -> float:
    """Reverse sigmoid: 10 when x << x0, 0 when x >> x0."""
    return 10.0 - _sigmoid(x, x0, k)


def _piecewise_linear(x: float, points: Tuple[Tuple[float, float], ...]) -> float:
    """Piecewise linear interpolation through sorted (x_i, y_i) anchor points.

    Clamps the output between min and max anchor y-values.
    For 'higher is worse' metrics, pass decreasing y-anchors directly
    (e.g. ((3, 10), (15, 0))) — Fix #11 removed the duplicate reverse helper.
    """
    pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(pts) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i] + EPS)
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


# --------------------------------------------------------------------------- #
# §6.4 Hard knockouts — evaluated BEFORE any weighting
# --------------------------------------------------------------------------- #


def apply_knockouts(metrics: Metrics) -> Optional[ScoringResult]:
    """Return an instant-fail ScoringResult if any knockout condition fires.

    Returns ``None`` when the strategy passes the gate.
    """
    N = metrics.basic.total_trades
    PF = metrics.basic.profit_factor
    MDD = abs(metrics.drawdown.max_drawdown)  # magnitude, positive

    reasons: list[str] = []
    if N < 30:
        reasons.append(
            f"Insufficient sample size: N={N} < 30 trades required for any statistical confidence."
        )
    if PF < 1.0:
        reasons.append(
            f"Profit Factor {PF:.3f} < 1.0 — strategy is unprofitable before costs; no further scoring warranted."
        )
    if MDD > 0.35:
        reasons.append(
            f"Maximum Drawdown magnitude {MDD*100:.1f}% exceeds 35% — catastrophic capital loss inadmissible per risk mandate."
        )

    if not reasons:
        return None

    return ScoringResult(
        status="KNOCKED_OUT",
        knockout_reason=" | ".join(reasons),
        category_scores=None,
        penalties=PenaltiesApplied(),
        raw_score=None,
        final_score=0,
        label="KNOCKED_OUT",
    )


# --------------------------------------------------------------------------- #
# §6.3 Per-category scoring
# --------------------------------------------------------------------------- #


def _pick_effective_mc(mc: MonteCarloResult) -> MCTestResult:
    """Return the MC test to use for Category-4 scoring — blueprint §5.2 caveat."""
    return mc.test_c if mc.effective_mc_source == "test_c" else mc.test_a


def _score_profitability(metrics: Metrics) -> float:
    """Cat 1, w=0.25."""
    b = metrics.basic
    s_pf = _sigmoid(b.profit_factor, x0=1.5, k=2.5)
    s_wr = _sigmoid(b.win_rate, x0=0.50, k=6.0)
    s_pr = _sigmoid(b.payoff_ratio, x0=1.5, k=2.0)
    s_ep = _sigmoid(b.expectancy_per_unit_risk, x0=0.20, k=4.0)
    if b.expectancy_per_unit_risk <= 0:
        s_ep = 0.0
    return 0.35 * s_pf + 0.20 * s_wr + 0.20 * s_pr + 0.25 * s_ep


def _score_risk_adj(metrics: Metrics) -> float:
    """Cat 2, w=0.25."""
    ra = metrics.risk_adj
    s_sr = _sigmoid(ra.sharpe_ratio, x0=1.0, k=2.2)
    if ra.sharpe_ratio <= 0:
        s_sr = 0.0
    s_so = _sigmoid(ra.sortino_ratio, x0=1.5, k=1.8)
    calmar = ra.calmar_ratio if ra.calmar_ratio is not None else 0.0
    s_ca = _sigmoid(calmar, x0=1.0, k=2.5)
    if calmar <= 0:
        s_ca = 0.0
    return 0.40 * s_sr + 0.30 * s_so + 0.30 * s_ca


def _score_drawdown(metrics: Metrics) -> float:
    """Cat 3, w=0.20. MDD is signed (negative) so pass the magnitude through
    reverse sigmoids carefully (reverse sigmoid decreases as x grows)."""
    d = metrics.drawdown
    mdd_mag = abs(d.max_drawdown)  # use magnitude
    # Blueprint: reverse sigmoid x0=-0.20, k=15 — here we use magnitude form
    # MDD_mag=0.10 → ≥9, MDD_mag=0.40 → ≤2
    # So for magnitude: good small → high score, bad large → low score
    s_mdd = _reverse_sigmoid(mdd_mag, x0=0.20, k=15.0)
    # Ulcer Index: UI <= 0.03 → ≥9, reverse sigmoid x0=0.08, k=40
    s_ui = _reverse_sigmoid(d.ulcer_index, x0=0.08, k=40.0)
    # Recovery factor: x0=2.0, k=2.0. >=5 → ≥9, <=0.5 → ≤1
    s_rf = _sigmoid(d.recovery_factor, x0=2.0, k=2.0)
    if d.recovery_factor <= 0.5:
        s_rf = min(s_rf, 1.0)
    return 0.45 * s_mdd + 0.25 * s_ui + 0.30 * s_rf


def _score_robustness(metrics: Metrics, mc: MonteCarloResult) -> float:
    """Cat 4, w=0.15. Uses Test C when correlation_flag, else Test A."""
    eff = _pick_effective_mc(mc)
    rr = eff.robustness_rate if eff.robustness_rate is not None else 0.0
    stab = eff.sharpe_stability if eff.sharpe_stability is not None else 0.0
    z_abs = abs(metrics.behav.runs_z_score)
    pmf = metrics.behav.profitable_month_fraction if metrics.behav.profitable_month_fraction is not None else 0.0

    # Linear piecewise for RR and stability: 0%→0, 50%→4, 80%→7, 95%→10
    pts_rr = ((0.0, 0.0), (0.50, 4.0), (0.80, 7.0), (0.95, 10.0))
    s_rr = _piecewise_linear(rr, pts_rr)
    s_stab = _piecewise_linear(stab, pts_rr)

    # Runs-test Z: reverse sigmoid x0=1.96, k=3. |Z|<1→≥9, |Z|>3→≤2
    s_z = _reverse_sigmoid(z_abs, x0=1.96, k=3.0)

    # % Profitable Months x0=0.60, k=6. >=0.80 → ≥9
    s_pm = _sigmoid(pmf, x0=0.60, k=6.0)

    return 0.35 * s_rr + 0.25 * s_stab + 0.20 * s_z + 0.20 * s_pm


def _score_sanity(metrics: Metrics) -> float:
    """Cat 5, w=0.10."""
    bv = metrics.behav
    g = metrics.growth
    ra = metrics.risk_adj

    s_sk = _sigmoid(g.skew_returns, x0=0.5, k=2.5)
    if g.skew_returns < -0.5:
        s_sk = min(s_sk, 2.0)
    # Excess kurtosis: reverse x0=3, k=1.5. ≤0 → 9, ≥8 → ≤2
    s_ku = _reverse_sigmoid(g.excess_kurtosis_returns, x0=3.0, k=1.5)
    # Tail ratio x0=1.0, k=4. ≥1.5 → 9, ≤0.7 → ≤2
    s_tr = _sigmoid(ra.tail_ratio, x0=1.0, k=4.0)
    if ra.tail_ratio <= 0.7:
        s_tr = min(s_tr, 2.0)
    # Max loss streak: decreasing anchors (higher streak → lower score)
    ls_pts = ((3, 10.0), (6, 6.0), (10, 2.0), (15, 0.0))
    s_ls = _piecewise_linear(float(bv.max_loss_streak), ls_pts)

    return 0.35 * s_sk + 0.25 * s_ku + 0.20 * s_tr + 0.20 * s_ls


def _score_sufficiency(metrics: Metrics, trades_count: int, outliers_tagged: int) -> float:
    """Cat 6, w=0.05."""
    N = metrics.basic.total_trades
    # Trade count piecewise: N<10→0, 10→2, 30→5, 100→8, ≥500→10
    n_pts = ((10, 2.0), (30, 5.0), (100, 8.0), (500, 10.0))
    if N < 10:
        s_n = 0.0
    else:
        s_n = _piecewise_linear(float(N), n_pts)
    # Outlier fraction (higher is worse): ≤1%→10, 5%→6, 15%→2, >25%→0
    frac = outliers_tagged / max(1, trades_count)
    of_pts = ((0.01, 10.0), (0.05, 6.0), (0.15, 2.0), (0.25, 0.0))
    s_of = _piecewise_linear(frac, of_pts)
    return 0.60 * s_n + 0.40 * s_of


# --------------------------------------------------------------------------- #
# §6.5 Multiplicative penalties — applied AFTER weighted sum
# --------------------------------------------------------------------------- #


def _apply_penalties(
    metrics: Metrics, mc: MonteCarloResult
) -> Tuple[PenaltiesApplied, float]:
    """Compute individual penalty factors (each ∈ [0.5, 1.0]) + product P.

    Returns (penalties, P).
    """
    mdd_mag = abs(metrics.drawdown.max_drawdown)
    p_mdd = max(0.5, 1.0 - 2.0 * max(0.0, mdd_mag - 0.40))

    cvar = metrics.risk_adj.cvar_95
    p_cvar = max(0.6, 1.0 - 4.0 * max(0.0, cvar - 0.10))

    eff = _pick_effective_mc(mc)
    rr = eff.robustness_rate if eff.robustness_rate is not None else 0.0
    p_rr = max(0.5, rr / 0.50) if rr < 0.50 else 1.0

    n = metrics.basic.total_trades
    p_n50 = max(0.85, n / 50.0) if n < 50 else 1.0

    P = p_mdd * p_cvar * p_rr * p_n50
    penalties = PenaltiesApplied(
        mdd_40pct=float(p_mdd),
        cvar_10pct=float(p_cvar),
        mc_rr=float(p_rr),
        n_50=float(p_n50),
        penalty_product=float(P),
    )
    return penalties, float(P)


# --------------------------------------------------------------------------- #
# §6.6 Score labels
# --------------------------------------------------------------------------- #


def _score_label(final_score: int) -> str:
    if final_score >= 90:
        return "Exceptional"
    if final_score >= 75:
        return "Strong"
    if final_score >= 60:
        return "Solid"
    if final_score >= 45:
        return "Marginal"
    if final_score >= 30:
        return "Weak"
    return "Poor"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def score_strategy(
    metrics: Metrics,
    mc: MonteCarloResult,
    *,
    trades_count: int,
    outliers_tagged: int,
) -> ScoringResult:
    """§6 full scoring pipeline.

    ``trades_count`` is the total raw trade count (before cleaning) used for
    the outlier-fraction denominator in §6 Cat 6; ``metrics.basic.total_trades``
    is the cleaned count used everywhere else.
    """
    # (1) Knockouts first — short-circuit on failure
    ko = apply_knockouts(metrics)
    if ko is not None:
        return ko

    # (2) Category subscores ∈ [0, 10]
    s_profit = _score_profitability(metrics)
    s_ra = _score_risk_adj(metrics)
    s_dd = _score_drawdown(metrics)
    s_rob = _score_robustness(metrics, mc)
    s_san = _score_sanity(metrics)
    s_suf = _score_sufficiency(metrics, trades_count, outliers_tagged)

    # (3) Weighted raw score ∈ [0, 10]
    raw = (
        0.25 * s_profit
        + 0.25 * s_ra
        + 0.20 * s_dd
        + 0.15 * s_rob
        + 0.10 * s_san
        + 0.05 * s_suf
    )
    raw = float(np.clip(raw, 0.0, 10.0))

    # (4) Multiplicative penalties
    penalties, P = _apply_penalties(metrics, mc)

    # (5) Final integer score 0..100 + label
    final = int(round(np.clip(10.0 * raw * P, 0.0, 100.0)))
    label = _score_label(final)

    return ScoringResult(
        status="SCORED",
        knockout_reason=None,
        category_scores=CategoryScores(
            profitability=float(s_profit),
            risk_adj=float(s_ra),
            drawdown=float(s_dd),
            robustness=float(s_rob),
            sanity=float(s_san),
            sufficiency=float(s_suf),
        ),
        penalties=penalties,
        raw_score=float(raw),
        final_score=final,
        label=label,
    )
