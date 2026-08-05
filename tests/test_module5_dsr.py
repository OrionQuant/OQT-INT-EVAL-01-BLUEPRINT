import pytest
import math
import numpy as np

from engine import metrics
from engine.models import Trade
from datetime import datetime


def test_psr_per_trade_benchmark():
    # Example: SR_trade = 0.10, N_obs = 250, skew=0, kurt=3
    psr = metrics._probabilistic_sharpe_ratio(0.10, 250, sr_ref=0.0, skew=0.0, kurt=3.0)
    assert psr == pytest.approx(0.9422, rel=1e-3)


def test_expected_max_sharpe_empirical_scaling():
    # Create a synthetic trials distribution with mean=0.5, sigma=0.2
    mean = 0.5
    sigma = 0.2
    std_limit = metrics._expected_max_sharpe(50, mean=0.0, sigma=1.0)
    scaled = metrics._expected_max_sharpe(50, mean=mean, sigma=sigma)
    assert scaled == pytest.approx(mean + sigma * std_limit, rel=1e-12)


def test_compute_all_metrics_requires_n_trials_ge_2():
    # Build one minimal trade to pass ingestion
    t = Trade(
        timestamp=datetime.utcnow(),
        symbol="EURUSD",
        side="long",
        quantity=1.0,
        entry_price=1.0,
        exit_price=1.01,
        pnl=10.0,
        entry_notional=1000.0,
    )
    with pytest.raises(ValueError):
        metrics.compute_all_metrics([t], start_balance=10000.0, rf_annual=0.04, n_trials=1)