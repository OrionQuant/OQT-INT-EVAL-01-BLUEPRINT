"""Mandatory remediation tests — ingestion, cleaning, EU fixtures, seed strings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.cleaning import clean_raw_rows, _coerce_number, _coerce_timestamp
from engine.ingestion import (
    _parse_number,
    _parse_timestamp_eu,
    ingest_file,
    parse_ctrader_html,
    parse_csv,
)
from engine.metrics import compute_all_metrics
from engine.monte_carlo import run_monte_carlo
from engine.seed import make_rng

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------- #
# Fix #1 — European / US number parsing
# --------------------------------------------------------------------------- #


class TestParseNumber:
    def test_us_thousands(self):
        assert _parse_number("1,000.00") == pytest.approx(1000.0)

    def test_eu_thousands_and_decimal(self):
        assert _parse_number("1.000,00") == pytest.approx(1000.0)
        assert _parse_number("2.000,00") == pytest.approx(2000.0)

    def test_eu_negative_decimal(self):
        assert _parse_number("-50,00") == pytest.approx(-50.0)
        assert _parse_number("−50,00") == pytest.approx(-50.0)

    def test_zero_and_empty(self):
        assert _parse_number("0") == pytest.approx(0.0)
        assert _parse_number("0,00") == pytest.approx(0.0)
        assert _parse_number("") is None
        assert _parse_number(None) is None
        assert _parse_number("-") is None

    def test_plain_decimal(self):
        assert _parse_number("26.50") == pytest.approx(26.50)
        assert _parse_number("26,50") == pytest.approx(26.50)

    def test_coerce_number_delegates(self):
        assert _coerce_number("1.000,25") == pytest.approx(1000.25)
        assert _coerce_number("1,000.25") == pytest.approx(1000.25)
        assert _coerce_number(-3.0) == pytest.approx(-3.0)


# --------------------------------------------------------------------------- #
# Fix #2 — CSV delimiter sniffing
# --------------------------------------------------------------------------- #


class TestCsvDelimiterSniffing:
    def test_comma_separated(self):
        raw = b"timestamp,symbol,side,quantity,entry_price,exit_price,pnl\n" \
              b"2025-01-03 08:00:00,EURUSD,Buy,1.0,1.08,1.083,26.5\n"
        rows = parse_csv(raw, "a.csv")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "EURUSD"

    def test_semicolon_european(self):
        raw = (FIXTURES / "eu_trades_semicolon.csv").read_bytes()
        rows = parse_csv(raw, "eu.csv")
        assert len(rows) == 3
        assert rows[0]["symbol"] == "EURUSD"

    def test_tab_separated(self):
        raw = b"timestamp\tsymbol\tside\tquantity\tentry_price\texit_price\tpnl\n" \
              b"2025-01-03 08:00:00\tEURUSD\tBuy\t1.0\t1.08\t1.083\t26.5\n"
        rows = parse_csv(raw, "t.tsv")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "EURUSD"


# --------------------------------------------------------------------------- #
# Fix #3 — dayfirst timestamp parsing
# --------------------------------------------------------------------------- #


class TestTimestampParsing:
    def test_dd_mm_yyyy(self):
        ts = _parse_timestamp_eu("03.01.2025 08:00:00")
        assert ts is not None
        assert ts.day == 3
        assert ts.month == 1
        assert ts.year == 2025

    def test_coerce_timestamp_dayfirst(self):
        ts = _coerce_timestamp("14.02.2025 10:05:00")
        assert ts is not None
        assert ts.day == 14
        assert ts.month == 2


# --------------------------------------------------------------------------- #
# Fix #4 + EU integration — cTrader HTML without losing first row
# --------------------------------------------------------------------------- #


class TestEuropeanCTraderExport:
    def test_zero_row_loss_and_pnl(self, tmp_path):
        raw = (FIXTURES / "ctrader_eu_sample.html").read_bytes()
        rows = parse_ctrader_html(raw)
        assert len(rows) == 5  # header outside thead/tbody must not drop P1

        first = rows[0]
        assert first["symbol"] == "EURUSD"
        assert first["quantity"] == pytest.approx(1.0)
        assert first["pnl"] == pytest.approx(26.50)
        assert first["commission"] == pytest.approx(-3.0)
        assert first["entry_price"] == pytest.approx(1.08)
        assert first["balance"] == pytest.approx(10026.50)

        # Total Net P&L across all five rows
        total_pnl = sum(float(r["pnl"]) for r in rows)
        assert total_pnl == pytest.approx(26.50 - 33.30 + 66.80 + 13.00 + 96.70)

        # Full pipeline: cleaning must keep all five
        rows2, fmt, _, n = ingest_file(raw, "eu_report.html")
        assert fmt == "ctrader_html"
        assert n == 5
        trades, report = clean_raw_rows(rows2, evaluations_dir=str(tmp_path))
        assert len(trades) == 5
        assert report.rows_rejected_schema == 0
        assert sum(t.pnl for t in trades) == pytest.approx(total_pnl)

    def test_header_outside_tbody_keeps_first_trade(self):
        html = """
        <table class="history">
        <tr><th>Symbol</th><th>Time</th><th>Type</th><th>Volume</th>
            <th>Open Price</th><th>Close Price</th><th>Net P&amp;L</th></tr>
        <tbody>
        <tr><td>EURUSD</td><td>03.01.2025 08:00:00</td><td>Buy</td><td>1,00</td>
            <td>1,08000</td><td>1,08300</td><td>26,50</td></tr>
        <tr><td>GBPUSD</td><td>06.01.2025 09:30:00</td><td>Buy</td><td>1,00</td>
            <td>1,26500</td><td>1,26200</td><td>-33,30</td></tr>
        </tbody>
        </table>
        """
        rows = parse_ctrader_html(html)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "EURUSD"
        assert rows[0]["pnl"] == pytest.approx(26.50)


# --------------------------------------------------------------------------- #
# Fix #5 — deterministic sort tiebreak via input_idx
# --------------------------------------------------------------------------- #


class TestSortReproducibility:
    def test_identical_timestamps_stable_across_runs(self, tmp_path):
        base = {
            "timestamp": "2025-01-03 08:00:00",
            "side": "Buy",
            "quantity": 1.0,
            "entry_price": 1.08,
            "exit_price": 1.083,
            "pnl": 10.0,
            "symbol": "EURUSD",
        }
        rows = [
            {**base, "deal_id": "A", "pnl": 1.0},
            {**base, "deal_id": "B", "pnl": 2.0},
            {**base, "deal_id": "C", "pnl": 3.0},
        ]
        t1, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path / "a"))
        t2, _ = clean_raw_rows(list(reversed(rows)), evaluations_dir=str(tmp_path / "b"))
        # Same input order → same output; reversed input → reversed input_idx order
        assert [t.deal_id for t in t1] == ["A", "B", "C"]
        assert [t.deal_id for t in t2] == ["C", "B", "A"]


# --------------------------------------------------------------------------- #
# Fix #7 — PnL zero / fee sign
# --------------------------------------------------------------------------- #


class TestPnlImputation:
    def test_zero_pnl_not_overwritten_when_naive_also_zero(self, tmp_path):
        rows = [{
            "timestamp": "2025-01-03 08:00:00",
            "side": "Buy",
            "quantity": 1.0,
            "entry_price": 1.08,
            "exit_price": 1.08,  # naive = 0
            "pnl": 0.0,
            "commission": 0.0,
            "swap": 0.0,
            "symbol": "EURUSD",
        }]
        trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        assert len(trades) == 1
        assert trades[0].pnl == pytest.approx(0.0)

    def test_missing_pnl_uses_gross_plus_fees(self, tmp_path):
        rows = [{
            "timestamp": "2025-01-03 08:00:00",
            "side": "Buy",
            "quantity": 1.0,
            "entry_price": 1.08,
            "exit_price": 1.083,
            "gross_pnl": 30.0,
            "commission": -3.0,
            "swap": -0.5,
            "symbol": "EURUSD",
            # pnl intentionally omitted
        }]
        trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        assert trades[0].pnl == pytest.approx(26.5)


# --------------------------------------------------------------------------- #
# Fix #8 — capped_pnl consumed downstream
# --------------------------------------------------------------------------- #


class TestCappedPnlScoringPath:
    def test_metrics_use_capped_when_present(self, tmp_path):
        rows = []
        for i in range(40):
            rows.append({
                "timestamp": f"2025-01-{(i % 28) + 1:02d} 08:00:00",
                "side": "Buy",
                "quantity": 1.0,
                "entry_price": 1.08,
                "exit_price": 1.081,
                "pnl": 10.0,
                "symbol": "EURUSD",
            })
        # One extreme outlier
        rows.append({
            "timestamp": "2025-02-01 08:00:00",
            "side": "Buy",
            "quantity": 1.0,
            "entry_price": 1.08,
            "exit_price": 2.0,
            "pnl": 50_000.0,
            "symbol": "EURUSD",
        })
        trades, report = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        assert report.outliers_tagged >= 1
        outlier = next(t for t in trades if t.is_outlier)
        assert outlier.capped_pnl is not None
        assert outlier.capped_pnl < outlier.pnl

        metrics, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=1)
        # Net PnL on the scoring path must reflect the cap (not the raw 50k spike)
        assert metrics.growth.net_pnl < 50_000.0


# --------------------------------------------------------------------------- #
# Fix #9 — duration_seconds
# --------------------------------------------------------------------------- #


class TestDurationSeconds:
    def test_positive_inter_trade_duration(self, tmp_path):
        rows = [
            {
                "timestamp": "2025-01-03 08:00:00", "side": "Buy", "quantity": 1.0,
                "entry_price": 1.08, "exit_price": 1.083, "pnl": 26.5, "symbol": "EURUSD",
            },
            {
                "timestamp": "2025-01-04 08:00:00", "side": "Buy", "quantity": 1.0,
                "entry_price": 1.08, "exit_price": 1.083, "pnl": 26.5, "symbol": "EURUSD",
            },
        ]
        trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        assert trades[0].duration_seconds == pytest.approx(1.0)
        assert trades[1].duration_seconds == pytest.approx(86400.0)


# --------------------------------------------------------------------------- #
# Fix #10 — seed JSON string round-trip reproducibility
# --------------------------------------------------------------------------- #


class TestSeedStringRoundTrip:
    def test_seed_is_string_and_json_roundtrip_preserves_mc(self, tmp_path):
        raw = (FIXTURES / "sample_trades.csv").read_bytes()
        rows, _, _, _ = ingest_file(raw, "s.csv")
        trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        metrics, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=1)

        # Large seed that would lose precision as a JS Number (> 2^53)
        big = "9007199254740993"  # 2^53 + 1
        rng1, seed_str = make_rng(big)
        assert isinstance(seed_str, str)
        assert seed_str == big

        # Simulate browser JSON round-trip (seed stays a string)
        payload = json.dumps({"seed": seed_str})
        restored = json.loads(payload)["seed"]
        assert restored == big
        assert isinstance(restored, str)

        rng2, seed2 = make_rng(restored)
        assert seed2 == big

        mc1 = run_monte_carlo(
            trades, rng1, start_balance=10_000.0, iterations=50,
            actual_max_drawdown=metrics.drawdown.max_drawdown,
            avg_trades_per_year=metrics.risk_adj.avg_trades_per_year,
        )
        mc2 = run_monte_carlo(
            trades, rng2, start_balance=10_000.0, iterations=50,
            actual_max_drawdown=metrics.drawdown.max_drawdown,
            avg_trades_per_year=metrics.risk_adj.avg_trades_per_year,
        )
        assert mc1.test_a.percentiles.total_return == pytest.approx(
            mc2.test_a.percentiles.total_return, abs=1e-12
        )


# --------------------------------------------------------------------------- #
# DSR / PSR parameter rules
# --------------------------------------------------------------------------- #


class TestDsrPsrParameterRules:
    def test_psr_uses_n_obs_dsr_uses_n_trials(self, tmp_path):
        raw = (FIXTURES / "sample_trades.csv").read_bytes()
        rows, _, _, _ = ingest_file(raw, "s.csv")
        trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path))

        m1, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=1)
        m50, _, _ = compute_all_metrics(trades, start_balance=10_000.0, n_trials=50)
        assert m1.risk_adj.probabilistic_sharpe_ratio is not None
        assert m50.risk_adj.probabilistic_sharpe_ratio is not None
        # Same N_obs → same PSR
        assert m1.risk_adj.probabilistic_sharpe_ratio == pytest.approx(
            m50.risk_adj.probabilistic_sharpe_ratio, abs=1e-12
        )
        assert m50.risk_adj.dsr_n_trials == 50
        assert m50.risk_adj.deflated_sharpe_ratio is not None
        # Higher N_trials raises the DSR reference bar → DSR ≤ PSR
        assert m50.risk_adj.deflated_sharpe_ratio <= m50.risk_adj.probabilistic_sharpe_ratio + 1e-12
        # More trials → DSR no higher than with N_trials=1
        assert m50.risk_adj.deflated_sharpe_ratio <= m1.risk_adj.deflated_sharpe_ratio + 1e-12

    def test_n_trials_required(self, tmp_path):
        raw = (FIXTURES / "sample_trades.csv").read_bytes()
        rows, _, _, _ = ingest_file(raw, "s.csv")
        trades, _ = clean_raw_rows(rows, evaluations_dir=str(tmp_path))
        with pytest.raises(TypeError):
            compute_all_metrics(trades, start_balance=10_000.0)  # noqa: missing n_trials
        with pytest.raises(ValueError):
            compute_all_metrics(trades, start_balance=10_000.0, n_trials=0)
