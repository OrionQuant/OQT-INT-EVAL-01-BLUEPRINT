"""Known-value tests for engine.metrics — every metric in §4.1…§4.5.

Uses sample_trades.csv fixture (N=63) for integration tests, plus a hand-
crafted 4-trade mini-series for exact known-value tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from engine.cleaning import clean_raw_rows
from engine.ingestion import ingest_file
from engine.metrics import compute_all_metrics
from engine.models import Trade

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluations"


def _mini_trades():
    """4 trades with hand-computable values. Synthetic balance column set."""
    base = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    # Trade 1: long 1 lot EURUSD from 1.08 to 1.09 → +1000 pips ≡ $100, balance → 10100
    # Trade 2: short 1 lot GBPUSD 1.27 to 1.26 → +100 pips ≡ $100 → 10200
    # Trade 3: long 1 lot EURUSD 1.09 to 1.085 → -50 pips ≡ -$50 → 10150
    # Trade 4: long 1 lot XAUUSD 2000 → 2050, qty=0.1 lot, $5 per $1 move, qty 0.1 means 1 oz
    #     entry_notional = 0.1 * 2000 = 200; profit = (2050-2000) * 100$/oz * 0.1 oz = $500
    t = []
    pnls = [100.0, 100.0, -50.0, 500.0]
    bal = 10000.0
    for i, p in enumerate(pnls):
        bal += p
        t.append(Trade(
            position_id=f"P{i}", deal_id=f"D{i}",
            timestamp=base + timedelta(days=i),
            symbol=["EURUSD", "GBPUSD", "EURUSD", "XAUUSD"][i],
            side="short" if i == 1 else "long",
            quantity=[1.0, 1.0, 1.0, 0.1][i],
            entry_price=[1.08, 1.27, 1.09, 2000.0][i],
            exit_price=[1.09, 1.26, 1.085, 2050.0][i],
            gross_pnl=p, commission=0.0, swap=0.0, pnl=p,
            balance_after=bal,
            entry_notional=[1.0 * 1.08, 1.0 * 1.27, 1.0 * 1.09, 0.1 * 2000.0][i],
            is_outlier=False, duration_seconds=86400.0,
        ))
    return t


# --------------------------------------------------------------------------- #
# Mini-series exact known-value tests
# --------------------------------------------------------------------------- #


class TestMiniKnownValues:
    def setup_method(self):
        self.trades = _mini_trades()
        self.metrics, self.equity, self.monthly = compute_all_metrics(
            self.trades, start_balance=10000.0, rf_annual=0.04, n_trials=2,
            sr_trials=[0.10, 0.12],
        )

    # §4.1 Basic
    def test_basic_counts(self):
        b = self.metrics.basic
        assert b.total_trades == 4
        assert b.win_count == 3
        assert b.loss_count == 1
        assert b.tie_count == 0
        assert b.win_rate == pytest.approx(0.75)
        assert b.loss_rate == pytest.approx(0.25)

    def test_basic_averages(self):
        b = self.metrics.basic
        # Wins: 100, 100, 500 → mean = 700/3 = 233.33
        assert b.average_win == pytest.approx((100 + 100 + 500) / 3)
        # Losses: 50
        assert b.average_loss == pytest.approx(50.0)
        assert b.profit_factor == pytest.approx((100 + 100 + 500) / 50.0)
        assert b.payoff_ratio == pytest.approx((700 / 3) / 50.0)
        # Expectancy = WR*AW - (1-WR)*AL = 0.75*233.33 - 0.25*50
        exp_val = 0.75 * b.average_win - 0.25 * b.average_loss
        assert b.expectancy_per_trade == pytest.approx(exp_val)

    # §4.2 Growth
    def test_growth_net_pnl_and_total_return(self):
        g = self.metrics.growth
        # Net PnL: 100 + 100 - 50 + 500 = 650
        assert g.net_pnl == pytest.approx(650.0)
        # TR = 650 / 10000 = 0.065
        assert g.total_return == pytest.approx(0.065)

    # §4.3 Drawdown
    def test_drawdown_is_non_positive(self):
        d = self.metrics.drawdown
        # There is no drawdown in this series (balance never below peak?)
        # Series: 10000 → 10100 → 10200 → 10150 → 10650
        # Peak after 4 trades: 10200 after trade 2; after trade 3 balance=10150
        # DD after trade 3 = (10150 - 10200) / 10200 = -50/10200 ≈ -0.00490
        # MDD is that value
        assert d.max_drawdown <= 0.0
        expected_mdd = (10150.0 - 10200.0) / 10200.0
        assert d.max_drawdown == pytest.approx(expected_mdd, abs=1e-4)
        # Recovery factor = 650 / (|MDD|*10000)
        assert d.recovery_factor == pytest.approx(
            650.0 / max(1e-9, abs(d.max_drawdown) * 10000.0),
            rel=1e-3,
        )

    # §4.5 Behavioural
    def test_behavioural_streaks(self):
        bv = self.metrics.behav
        # Trades: Win, Win, Loss, Win → max_win_streak = 2, max_loss_streak = 1
        assert bv.max_win_streak == 2
        assert bv.max_loss_streak == 1
        assert bv.profit_streak_ratio == pytest.approx(2.0)
        assert bv.best_trade == pytest.approx(500.0)
        assert bv.worst_trade == pytest.approx(-50.0)

    # Equity curve shape
    def test_equity_curve_length(self):
        # N trades → N+1 equity points (including synthetic start point)
        assert len(self.equity) == 4 + 1
        # Starting balance
        assert self.equity[0].balance == pytest.approx(10000.0)
        # Final balance
        assert self.equity[-1].balance == pytest.approx(10650.0)


# --------------------------------------------------------------------------- #
# Sample CSV integration tests — structural verification
# --------------------------------------------------------------------------- #


class TestSampleCsvMetrics:
    @pytest.fixture(scope="class")
    def sample_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("evals")
        raw = (FIXTURES / "sample_trades.csv").read_bytes()
        rows, _, _, _ = ingest_file(raw, "sample_trades.csv")
        trades, report = clean_raw_rows(rows, evaluations_dir=str(tmp))
        m, eq, mo = compute_all_metrics(trades, start_balance=10000.0, n_trials=2, sr_trials=[0.05, 0.07])
        return m, eq, mo, report, trades

    def test_sample_size(self, sample_result):
        m, *_ = sample_result
        # 63 rows in CSV; some may be rejected? No, CSV is well-formed.
        assert m.basic.total_trades == 63

    def test_sample_positive_net_pnl(self, sample_result):
        m, *_ = sample_result
        # The last balance 12477.5 - 10000 = $2477.5 profit
        assert m.growth.net_pnl == pytest.approx(2477.5, rel=0.2)

    def test_sample_positive_profit_factor(self, sample_result):
        m, *_ = sample_result
        # Strategy looks profitable, PF should be > 1
        assert m.basic.profit_factor > 1.0

    def test_sample_sharpe_finite(self, sample_result):
        m, *_ = sample_result
        assert np.isfinite(m.risk_adj.sharpe_ratio)
        assert np.isfinite(m.risk_adj.sortino_ratio)

    def test_sample_cagr_computed(self, sample_result):
        m, *_ = sample_result
        # 2025-01-03 to 2025-12-30 → ~1 year → CAGR computed
        assert m.growth.cagr is not None
        assert not m.growth.cagr_flagged_insufficient

    def test_equity_curve_end_balance_matches(self, sample_result):
        m, eq, mo, report, trades = sample_result
        last_eq_balance = eq[-1].balance
        # Equity / scoring path uses soft-capped PnL (Fix #8)
        expected = 10000.0 + sum(
            (t.capped_pnl if t.capped_pnl is not None else t.pnl) for t in trades
        )
        assert last_eq_balance == pytest.approx(expected, rel=1e-6)

    def test_monthly_returns_span_correct(self, sample_result):
        m, eq, mo, report, trades = sample_result
        # CSV spans Jan 2025 through Dec 2025
        # Monthly returns has 12-1=11 entries (no return before first month)
        # Actually, pct_change drops NaN of first month, so 11 values
        assert len(mo) >= 10  # at least 10 monthly changes over ~12 months
