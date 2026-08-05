"""Feature tests — Modules 3 (CUSUM), 4 (moving-block), 5 (DSR/PSR)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.block_bootstrap import moving_block_sample, run_moving_block_bootstrap
from engine.cleaning import clean_raw_rows
from engine.ingestion import ingest_file
from engine.metrics import compute_all_metrics
from engine.monte_carlo import run_monte_carlo
from engine.regime import run_regime_check, _two_sided_cusum
from engine.seed import make_rng
from main import EvaluateConfig, _build_overfitting_estimate, _run_pipeline

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _sample_trades(tmp_path):
    raw = (FIXTURES / "sample_trades.csv").read_bytes()
    rows, _, _, _ = ingest_file(raw, "s.csv")
    trades, report = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
    return trades, report


# --------------------------------------------------------------------------- #
# Module 3 — CUSUM / regime
# --------------------------------------------------------------------------- #


class TestModule3CUSUM:
    def test_flat_series_no_hit(self):
        x = np.zeros(40)
        stat = _two_sided_cusum(x)
        assert stat.threshold_hit is False
        assert stat.n_regime_shifts_detected == 0

    def test_mean_shift_triggers_hit(self):
        rng = np.random.default_rng(0)
        first = rng.normal(0.0, 1.0, size=40)
        second = rng.normal(3.0, 1.0, size=40)  # clear level shift
        R = np.concatenate([first, second])
        check = run_regime_check(R)
        assert check.cusum_full is not None
        assert check.regime_unstable_flag is True
        assert check.mean_return_gap is not None
        assert abs(check.mean_return_gap) > 0

    def test_attached_to_metrics(self, tmp_path):
        trades, _ = _sample_trades(tmp_path)
        metrics, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=10, sr_trials=[0.10]*10)
        assert metrics.behav.regime_check is not None
        assert metrics.behav.regime_check.cusum_full is not None
        assert isinstance(metrics.behav.regime_check.regime_unstable_flag, bool)


# --------------------------------------------------------------------------- #
# Module 4 — moving-block bootstrap
# --------------------------------------------------------------------------- #


class TestModule4MovingBlock:
    def test_sample_length_and_range(self):
        rng, _ = make_rng(7)
        idx = moving_block_sample(50, L=5, rng=rng)
        assert len(idx) == 50
        assert idx.min() >= 0
        assert idx.max() < 50

    def test_deterministic_given_seed(self, tmp_path):
        trades, _ = _sample_trades(tmp_path)
        metrics, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=5, sr_trials=[0.05]*5)
        rng1, _ = make_rng(42)
        rng2, _ = make_rng(42)
        mb1 = run_moving_block_bootstrap(
            trades, rng1, start_balance=10_000.0, iterations=40,
            avg_trades_per_year=metrics.risk_adj.avg_trades_per_year,
            iid_robustness_rate=0.9,
        )
        mb2 = run_moving_block_bootstrap(
            trades, rng2, start_balance=10_000.0, iterations=40,
            avg_trades_per_year=metrics.risk_adj.avg_trades_per_year,
            iid_robustness_rate=0.9,
        )
        assert mb1.block_length == mb2.block_length
        assert mb1.percentiles.total_return == pytest.approx(mb2.percentiles.total_return, abs=1e-12)
        assert mb1.block_vs_iid_rr_gap == pytest.approx(0.9 - mb1.robustness_rate, abs=1e-12)

    def test_pipeline_includes_moving_block(self, tmp_path):
        raw = (FIXTURES / "sample_trades.csv").read_bytes()
        cfg = EvaluateConfig(start_balance=10_000.0, mc_iterations=30, n_trials=25, sr_trials=[0.05]*25)
        card = _run_pipeline(raw, "s.csv", config=cfg, seed="99", strategy_name="t")
        assert card.moving_block is not None
        assert card.moving_block.iterations == 30
        assert card.moving_block.block_length >= 1
        assert card.moving_block.robustness_rate is not None
        assert card.overfitting is not None
        assert card.overfitting.dsr_n_trials == 25
        assert card.overfitting.n_obs == card.metrics.basic.total_trades
        assert card.disclaimer.startswith("Notice:")
        assert card.metrics.risk_adj.deflated_sharpe_ratio is not None
        assert card.metrics.behav.regime_check is not None


# --------------------------------------------------------------------------- #
# Module 5 — required n_trials + DSR ≠ PSR N
# --------------------------------------------------------------------------- #


class TestModule5RequiredNTrials:
    def test_config_requires_n_trials(self):
        with pytest.raises(Exception):
            EvaluateConfig(start_balance=10_000.0)  # missing n_trials

    def test_dsr_not_using_trade_count_as_n(self, tmp_path):
        trades, _ = _sample_trades(tmp_path)
        n_obs = len(trades)
        # Deliberately set n_trials ≠ n_obs
        m, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=7, sr_trials=[0.05]*7)
        assert m.risk_adj.dsr_n_trials == 7
        assert m.risk_adj.dsr_n_trials != n_obs or n_obs == 7
        assert m.basic.total_trades == n_obs
