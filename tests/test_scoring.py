"""§6 Weighted scoring tests: weight-sum, knockouts, boundaries, penalties.

Three knockout conditions are each tested against a synthetic mini-metrics set
that intentionally triggers the gate. Weight-sum asserts that category × weight
adds up correctly to a raw score in [0, 10]. Score clamping and penalties are
verified on boundary inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.cleaning import clean_raw_rows
from engine.ingestion import ingest_file
from engine.metrics import compute_all_metrics
from engine.models import (
    BasicMetrics, BehaviouralMetrics, DrawdownMetrics, GrowthMetrics,
    Metrics, MonteCarloResult, MCPercentiles, MCTestResult, RiskAdjustedMetrics,
)
from engine.monte_carlo import run_monte_carlo
from engine.scoring import (
    apply_knockouts,
    score_strategy,
    _sigmoid,
    _piecewise_linear,
    _reverse_sigmoid,
)
from engine.seed import make_rng

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------- #
# Normalisation primitives (§6.3) — sanity
# --------------------------------------------------------------------------- #


class TestNormalisationPrimitives:
    def test_sigmoid_monotonic(self):
        ys = [_sigmoid(x, 0.0, 1.0) for x in (-10, -1, 0, 1, 10)]
        assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))

    def test_sigmoid_inflection(self):
        # At x = x0, score ≈ 5
        assert _sigmoid(1.5, 1.5, 2.0) == pytest.approx(5.0, abs=1e-6)

    def test_sigmoid_bounds(self):
        low = _sigmoid(-1000, 0.0, 1.0)
        high = _sigmoid(1000, 0.0, 1.0)
        assert 0.0 <= low <= 0.01
        assert 9.99 <= high <= 10.0

    def test_reverse_sigmoid(self):
        # Reverse: smaller x → higher score
        assert _reverse_sigmoid(-1.0, 0.0, 1.0) > _reverse_sigmoid(1.0, 0.0, 1.0)

    def test_piecewise_linear_anchors(self):
        pts = ((0.0, 0.0), (0.5, 4.0), (0.8, 7.0), (0.95, 10.0))
        for (x, y) in pts:
            assert _piecewise_linear(x, pts) == pytest.approx(y)

    def test_piecewise_linear_clamp_low(self):
        pts = ((0.5, 2.0), (0.95, 10.0))
        # N<10 → 0 clamped by caller, but make sure piecewise returns lowest anchor
        assert _piecewise_linear(0.01, pts) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Hard knockout tests (§6.4) — each condition individually
# --------------------------------------------------------------------------- #


def _base_metrics(**overrides):
    """Build a Metrics that passes knockouts; then apply overrides to trigger them."""
    basic = BasicMetrics(
        total_trades=100, win_rate=0.60, loss_rate=0.40,
        win_count=60, loss_count=40, tie_count=0,
        average_win=100.0, average_loss=50.0,
        profit_factor=3.0, payoff_ratio=2.0,
        expectancy_per_trade=40.0, expectancy_per_unit_risk=0.80,
        long_count=70, short_count=30,
        long_win_rate=0.61, short_win_rate=0.57,
    )
    growth = GrowthMetrics(
        net_pnl=4000.0, total_return=0.40, cagr=0.42, cagr_flagged_insufficient=False,
        mean_return_per_trade=0.004, median_return_per_trade=0.003,
        skew_returns=0.6, excess_kurtosis_returns=1.0,
    )
    dd = DrawdownMetrics(
        max_drawdown=-0.10, average_drawdown=-0.03, max_drawdown_duration_days=12.0,
        ulcer_index=0.05, recovery_factor=4.0,
    )
    ra = RiskAdjustedMetrics(
        sharpe_ratio=1.8, sharpe_small_sample_corrected=False, sortino_ratio=2.5,
        calmar_ratio=4.2, omega_ratio=1.6, romad=4.0, tail_ratio=1.2,
        var_95=0.01, cvar_95=0.02, annualised=True, avg_trades_per_year=100.0,
    )
    behav = BehaviouralMetrics(
        max_win_streak=8, max_loss_streak=3, profit_streak_ratio=8/3,
        runs_z_score=0.8, best_trade=600.0, worst_trade=-120.0,
        profitable_month_fraction=0.80, monthly_return_volatility=0.03,
    )
    b, g, d, r, bv = basic, growth, dd, ra, behav
    dct = {"basic": b, "growth": g, "drawdown": d, "risk_adj": r, "behav": bv}
    # Merge overrides
    for group, fields in overrides.items():
        dct[group] = type(dct[group])(**{**dct[group].model_dump(), **fields})
    return Metrics(**dct)


def _base_mc(correlation_flag: bool = False, rr: float = 0.9, stab: float = 0.75):
    # Enough shape that downstream code never returns None
    pct = MCPercentiles(
        total_return=[-0.20, -0.05, 0.10, 0.25, 0.40, 0.55, 0.90],
        max_drawdown=[-0.40, -0.25, -0.18, -0.12, -0.08, -0.04, -0.001],
        sharpe_ratio=[-1.0, 0.0, 0.8, 1.5, 2.2, 3.0, 4.5],
        profit_factor=[0.3, 0.8, 1.2, 1.8, 2.5, 3.3, 6.0],
    )
    return MonteCarloResult(
        test_a=MCTestResult(iterations=1000, percentiles=pct, robustness_rate=rr, sharpe_stability=stab, mdd_2x_probability=0.04),
        test_b=MCTestResult(iterations=1000, percentiles=pct, mdd_2x_probability=0.04),
        test_c=MCTestResult(iterations=1000, percentiles=pct, robustness_rate=rr, sharpe_stability=stab, flagged_worse_than_a=correlation_flag),
        correlation_flag=correlation_flag,
        effective_mc_source="test_c" if correlation_flag else "test_a",
    )


class TestKnockouts:
    def test_passing_metrics_no_knockout(self):
        m = _base_metrics()
        assert apply_knockouts(m) is None

    def test_knockout_low_sample_size(self):
        m = _base_metrics(basic={"total_trades": 29})
        ko = apply_knockouts(m)
        assert ko is not None
        assert ko.status == "KNOCKED_OUT"
        assert ko.final_score == 0
        assert "N=29 < 30" in ko.knockout_reason

    def test_knockout_profit_factor_below_1(self):
        m = _base_metrics(basic={"profit_factor": 0.95})
        ko = apply_knockouts(m)
        assert ko is not None
        assert ko.knockout_reason is not None
        assert "Profit Factor 0.95" in ko.knockout_reason

    def test_knockout_mdd_over_35_percent(self):
        # MDD magnitude 0.36 → fails
        m = _base_metrics(drawdown={"max_drawdown": -0.36})
        ko = apply_knockouts(m)
        assert ko is not None
        assert ko.knockout_reason is not None
        assert "36.0%" in ko.knockout_reason or "35%" in ko.knockout_reason

    def test_boundary_mdd_35_percent_passes(self):
        # Exactly 0.35 → passes (strict inequality knockout at > 35 %)
        m = _base_metrics(drawdown={"max_drawdown": -0.35})
        assert apply_knockouts(m) is None

    def test_boundary_pf_1_0_passes(self):
        m = _base_metrics(basic={"profit_factor": 1.0})
        assert apply_knockouts(m) is None

    def test_boundary_n_30_passes(self):
        m = _base_metrics(basic={"total_trades": 30})
        assert apply_knockouts(m) is None

    def test_multiple_knockouts_list_all_reasons(self):
        m = _base_metrics(basic={"total_trades": 10, "profit_factor": 0.5},
                          drawdown={"max_drawdown": -0.50})
        ko = apply_knockouts(m)
        assert ko is not None
        assert ko.knockout_reason.count(" | ") == 2  # two separators for 3 reasons


# --------------------------------------------------------------------------- #
# Weight-sum sanity — score is finite & bounded on realistic + extreme inputs
# --------------------------------------------------------------------------- #


class TestScoreSanity:
    def test_passing_metrics_score_in_0_100(self):
        m = _base_metrics()
        mc = _base_mc()
        s = score_strategy(m, mc, trades_count=100, outliers_tagged=1)
        assert 0 <= s.final_score <= 100
        assert s.status == "SCORED"
        assert s.label in {"Exceptional", "Strong", "Solid", "Marginal", "Weak", "Poor"}
        assert s.raw_score is not None
        assert 0.0 <= s.raw_score <= 10.0
        assert s.category_scores is not None
        # Category scores also bounded
        for v in s.category_scores.model_dump().values():
            assert 0.0 <= v <= 10.0

    def test_category_weights_sum_to_one(self):
        m = _base_metrics()
        mc = _base_mc()
        s = score_strategy(m, mc, trades_count=100, outliers_tagged=0)
        cats = s.category_scores.model_dump()
        w = {"profitability": 0.25, "risk_adj": 0.25, "drawdown": 0.20,
             "robustness": 0.15, "sanity": 0.10, "sufficiency": 0.05}
        raw_reconstructed = sum(w[c] * cats[c] for c in w)
        # Reconstructed should match raw_score to floating tolerance
        assert raw_reconstructed == pytest.approx(s.raw_score, rel=1e-3)

    def test_penalty_product_reduces_score(self):
        # Build metrics that PASS knockouts but trigger cvar/N/MC penalties:
        #  * N = 35 (≥30 passes, but < 50 → soft N penalty)
        #  * CVaR = 15% → cvar_10pct penalty
        #  * MDD = 30% (just below the 35% knockout floor → passes, no MDD_40 penalty)
        #  * low MC robustness rate rr=0.30 → mc_rr penalty
        m = _base_metrics(
            basic={"total_trades": 35},
            drawdown={"max_drawdown": -0.30},
            risk_adj={"cvar_95": 0.15},
        )
        mc = _base_mc(rr=0.30)
        s = score_strategy(m, mc, trades_count=35, outliers_tagged=0)
        # At least one individual penalty should be < 1.0 (cvar, n_50, or mc_rr)
        assert s.status == "SCORED", f"Expected SCORED, got KO: {s.knockout_reason}"
        individual_penalties = [s.penalties.cvar_10pct, s.penalties.mc_rr, s.penalties.n_50]
        assert any(p < 1.0 for p in individual_penalties), \
            f"Expected at least one penalty < 1.0, got {individual_penalties}"
        # Penalty product should reduce final score compared to pre-penalty
        # Individual penalties: mdd ≥0.5, cvar ≥0.6, mc_rr ≥0.5, n_50 ≥0.85 → product ≥ 0.1275
        assert 0.12 <= s.penalties.penalty_product <= 0.999

    def test_knocked_out_score_is_zero_without_subscores(self):
        m = _base_metrics(basic={"total_trades": 10})
        s = score_strategy(m, _base_mc(), trades_count=10, outliers_tagged=0)
        assert s.final_score == 0
        assert s.category_scores is None
        assert s.raw_score is None
        assert s.label == "KNOCKED_OUT"


# --------------------------------------------------------------------------- #
# End-to-end scoring via sample CSV (smoke)
# --------------------------------------------------------------------------- #


class TestSampleE2EScoring:
    def test_sample_score_is_deterministic(self, tmp_path):
        raw = (FIXTURES / "sample_trades.csv").read_bytes()
        rows, _, _, _ = ingest_file(raw, "s.csv")
        trades, report = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        metrics, _, _ = compute_all_metrics(trades, start_balance=10000.0, n_trials=1)
        rng1, seed1 = make_rng(42)
        rng2, seed2 = make_rng(42)
        mc1 = run_monte_carlo(trades, rng1, start_balance=10000.0, iterations=200,
                              actual_max_drawdown=metrics.drawdown.max_drawdown)
        mc2 = run_monte_carlo(trades, rng2, start_balance=10000.0, iterations=200,
                              actual_max_drawdown=metrics.drawdown.max_drawdown)
        s1 = score_strategy(metrics, mc1, trades_count=report.rows_received, outliers_tagged=report.outliers_tagged)
        s2 = score_strategy(metrics, mc2, trades_count=report.rows_received, outliers_tagged=report.outliers_tagged)
        assert s1.final_score == s2.final_score
        assert seed1 == seed2 == "42"
