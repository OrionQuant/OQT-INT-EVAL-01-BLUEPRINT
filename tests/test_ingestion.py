"""Tests for engine.ingestion (cTrader HTML parser + CSV/XLSX fallback)."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.ingestion import (
    ingest_file,
    parse_ctrader_html,
    parse_csv,
    sha256_file,
    _match_header,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_sha256_bytes(self):
        h = sha256_file(b"hello world")
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_header_matching(self):
        assert _match_header("Net P&L") == "pnl"
        assert _match_header("Gross Profit") == "gross_pnl"
        assert _match_header("Position ID") == "position_id"
        assert _match_header("Time") == "timestamp"
        assert _match_header("Volume") == "quantity"
        assert _match_header("something unknown") is None


# --------------------------------------------------------------------------- #
# cTrader HTML parser
# --------------------------------------------------------------------------- #


class TestCTraderHtml:
    def test_parse_sample_html(self):
        html = _read("ctrader_sample.html")
        rows = parse_ctrader_html(html)
        # The fixture has 63 trade rows
        assert isinstance(rows, list)
        assert len(rows) == 63

        # Spot-check the first row (EURUSD Buy 1.00)
        first = rows[0]
        assert first["symbol"] == "EURUSD"
        assert first["side"] == "Buy"
        assert first["quantity"] == pytest.approx(1.0)
        assert first["entry_price"] == pytest.approx(1.08000)
        assert first["exit_price"] == pytest.approx(1.08300)
        assert first["pnl"] == pytest.approx(26.50)
        assert first["gross_pnl"] == pytest.approx(30.00)
        assert first["commission"] == pytest.approx(-3.00)
        assert first["swap"] == pytest.approx(-0.50)
        assert first["deal_id"] == "D1"
        assert first["position_id"] == "P1001"

    def test_parse_html_second_row_sell(self):
        rows = parse_ctrader_html(_read("ctrader_sample.html"))
        # Row 3 is USDJPY Sell (0-indexed row 2)
        usdjpy = rows[2]
        assert usdjpy["side"] == "Sell"
        assert usdjpy["symbol"] == "USDJPY"
        assert usdjpy["pnl"] == pytest.approx(66.80)

    def test_parse_html_last_row(self):
        rows = parse_ctrader_html(_read("ctrader_sample.html"))
        last = rows[-1]
        assert last["symbol"] == "EURUSD"
        assert last["side"] == "Buy"
        assert last["pnl"] == pytest.approx(46.70)
        assert last["balance"] == pytest.approx(12477.50)

    def test_empty_html_returns_empty_list(self):
        rows = parse_ctrader_html(b"<html><body>No tables</body></html>")
        assert rows == []


# --------------------------------------------------------------------------- #
# CSV fallback
# --------------------------------------------------------------------------- #


class TestCsvFallback:
    def test_parse_sample_csv(self):
        csv_bytes = _read("sample_trades.csv")
        rows = parse_csv(csv_bytes, "sample_trades.csv")
        assert len(rows) == 63
        first = rows[0]
        assert first["symbol"] == "EURUSD"
        assert first["side"].lower() == "buy"
        assert float(first["pnl"]) == pytest.approx(26.5)

    def test_alias_column_detection(self):
        # CSV columns are: timestamp,symbol,side,quantity,entry_price,...
        # These map to canonical names via alias detection
        csv_bytes = _read("sample_trades.csv")
        rows = parse_csv(csv_bytes, "sample_trades.csv")
        for r in rows:
            assert "pnl" in r
            assert "timestamp" in r
            assert "symbol" in r
            assert "side" in r


# --------------------------------------------------------------------------- #
# Public entry point (ingest_file)
# --------------------------------------------------------------------------- #


class TestIngestFile:
    def test_ingest_html_detected_format(self):
        raw = _read("ctrader_sample.html")
        rows, fmt, sha, n = ingest_file(raw, "report.html")
        assert fmt == "ctrader_html"
        assert n == 63
        assert len(rows) == 63
        assert isinstance(sha, str) and len(sha) == 64

    def test_ingest_csv_detected_format(self):
        raw = _read("sample_trades.csv")
        rows, fmt, sha, n = ingest_file(raw, "trades.csv")
        assert fmt == "csv"
        assert n == 63
        assert len(rows) == 63

    def test_ingest_reproducible_sha(self):
        raw = _read("ctrader_sample.html")
        _, _, sha1, _ = ingest_file(raw, "a.html")
        _, _, sha2, _ = ingest_file(raw, "b.html")
        assert sha1 == sha2
