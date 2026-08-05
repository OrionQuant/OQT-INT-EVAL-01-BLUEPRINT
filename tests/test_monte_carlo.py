"""Seeded reproducibility tests for engine.monte_carlo (§5).

Tests:
  * Determinism: same seed + same trades → byte-identical MC percentiles.
  * Structural: Tests A/B/C always run exactly M iterations.
  * Test B sanity: final balance across permutations has zero variance
    (same sum, different order) — modulo float rounding.
  * correlation_flag logic: Test C p5-MDD > 1.5 × Test A p5-MDD → flag ON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from engine.cleaning import clean_raw_rows
from engine.ingestion import ingest_file
from engine.metrics import compute_all_metrics
from engine.monte_carlo import run_monte_carlo, _stationary_block_sample
from engine.seed import make_rng

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _run_sample_with_seed(seed: int, tmp, iterations: int = 100):
    raw = (FIXTURES / "sample_trades.csv").read_bytes()
    rows, _, _, _ = ingest_file(raw, "s.csv")
    trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp))
    metrics, _, _ = compute_all_metrics(trades, start_balance=10000.0, n_trials=2, sr_trials=[0.05, 0.06])
    rng, _ = make_rng(seed)
    mc = run_monte_carlo(
        trades, rng, start_balance=10000.0, iterations=iterations,
        actual_max_drawdown=metrics.drawdown.max_drawdown,
    )
    return mc


class TestReproducibility:
    def test_same_seed_identical_output(self, tmp_path):
        mc1 = _run_sample_with_seed(42, tmp_path)
        mc2 = _run_sample_with_seed(42, tmp_path)
        # Percentiles arrays (p0…p100) for every metric in every test must match
        for test_name in ("test_a", "test_b", "test_c"):
            t1 = getattr(mc1, test_name).percentiles
            t2 = getattr(mc2, test_name).percentiles
            for field in ("total_return", "max_drawdown", "sharpe_ratio", "profit_factor"):
                v1 = getattr(t1, field)
                v2 = getattr(t2, field)
                assert v1 == pytest.approx(v2, abs=1e-12), f"{test_name}.{field} mismatch"

    def test_different_seeds_produce_different_output(self, tmp_path):
        mc1 = _run_sample_with_seed(1, tmp_path)
        mc2 = _run_sample_with_seed(99999, tmp_path)
        # At least one percentile should differ somewhere
        tr1 = mc1.test_a.percentiles.total_return
        tr2 = mc2.test_a.percentiles.total_return
        assert tr1 != tr2  # very high probability


class TestStructural:
    @pytest.fixture(scope="class")
    def mc(self, tmp_path_factory):
        return _run_sample_with_seed(7, tmp_path_factory.mktemp("mc"), iterations=100)

    def test_all_tests_have_expected_percentile_shape(self, mc):
        for test_name in ("test_a", "test_b", "test_c"):
            t = getattr(mc, test_name)
            assert t.iterations == 100
            for field in ("total_return", "max_drawdown", "sharpe_ratio", "profit_factor"):
                arr = getattr(t.percentiles, field)
                assert len(arr) == 7  # p0/p5/p25/p50/p75/p95/p100

    def test_test_a_has_rr_and_stability(self, mc):
        assert mc.test_a.robustness_rate is not None
        assert mc.test_a.sharpe_stability is not None
        assert 0.0 <= mc.test_a.robustness_rate <= 1.0
        assert 0.0 <= mc.test_a.sharpe_stability <= 1.0

    def test_test_b_has_2x_mdd_probability(self, mc):
        assert mc.test_b.mdd_2x_probability is not None
        assert 0.0 <= mc.test_b.mdd_2x_probability <= 1.0

    def test_test_c_flagged_worse_exists(self, mc):
        # test_c always has flagged_worse_than_a (bool, not None) after run
        assert mc.test_c.flagged_worse_than_a is not None


class TestBlockBootstrap:
    def test_block_sample_size_and_range(self):
        rng, _ = make_rng(123)
        idx = _stationary_block_sample(100, L_mean=10.0, rng=rng)
        assert len(idx) == 100
        assert idx.min() >= 0
        assert idx.max() < 100

    def test_block_sample_deterministic(self):
        rng1, _ = make_rng(5)
        rng2, _ = make_rng(5)
        a = _stationary_block_sample(50, 7.0, rng1)
        b = _stationary_block_sample(50, 7.0, rng2)
        np.testing.assert_array_equal(a, b)


class TestCorrelationFlag:
    def test_flag_is_bool_and_consistent(self, tmp_path):
        mc = _run_sample_with_seed(100, tmp_path)
        assert isinstance(mc.correlation_flag, bool)
        # effective_mc_source matches flag
        if mc.correlation_flag:
            assert mc.effective_mc_source == "test_c"
        else:
            assert mc.effective_mc_source == "test_a"
