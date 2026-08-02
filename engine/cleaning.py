"""Data cleaning pipeline — blueprint §3.3 steps (1)…(9)."""

from __future__ import annotations

import csv as _csv
import os
import uuid
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from dateutil import parser as _dateparser

from .models import CleaningReport, Trade

SYNTHETIC_START_BALANCE = 10_000.0


def _coerce_side(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"buy", "long", "1", "+1", "bull", "bullish"}:
        return "long"
    if s in {"sell", "short", "-1", "bear", "bearish"}:
        return "short"
    return None


def _coerce_number(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace("\u00a0", "").replace(",", "").replace(" ", "").strip()
    if s in {"", "-", "—", "n/a"}:
        return np.nan
    # Strip leading/trailing currency-ish non-numeric characters
    while s and not (s[0].isdigit() or s[0] in "+-."):
        s = s[1:]
    while s and not (s[-1].isdigit() or s[-1] == "."):
        s = s[:-1]
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _coerce_timestamp(raw) -> pd.Timestamp | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(raw)
    else:
        try:
            ts = pd.Timestamp(_dateparser.parse(str(raw)))
        except (ValueError, TypeError, OverflowError):
            # Numeric → Excel serial
            try:
                n = float(raw)
            except (ValueError, TypeError):
                return None
            if 1_000 < n < 100_000:  # plausible Excel serial
                ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=n)
            else:
                # epoch ms
                ts = pd.Timestamp(n, unit="ms") if n > 1e12 else pd.Timestamp(n, unit="s")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _drop_and_remember(rejected: List[dict], row: dict, reason: str) -> None:
    r = dict(row)
    r["_rejection_reason"] = reason
    rejected.append(r)


def _write_rejected_csv(rejected: List[dict], job_id: str, out_dir: str) -> str | None:
    if not rejected:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"_rejected_rows_{job_id}.csv")
    keys: List[str] = []
    for row in rejected:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rejected)
    return path


def clean_raw_rows(
    raw_rows: List[dict],
    *,
    evaluations_dir: str,
    synthetic_start_balance: float = SYNTHETIC_START_BALANCE,
    mad_cap_factor: float = 10.0,
) -> Tuple[List[Trade], CleaningReport]:
    """Apply the 9-step cleaning pipeline from blueprint §3.3.

    Returns ``(trades, cleaning_report)``.
    """
    job_id = uuid.uuid4().hex[:8]
    report = CleaningReport(rows_received=len(raw_rows))
    rejected: List[dict] = []

    # ----------------------------------------------------------------------- #
    # Step 1: Schema validation (required fields present & parsable)
    # ----------------------------------------------------------------------- #
    step1_rows: List[dict] = []
    for row in raw_rows:
        # Skip rows that look like adjustment-only (no symbol or no quantity
        # but a P&L exists) -> remember as adjustment and exclude from trades
        if (not row.get("symbol") and row.get("pnl") is not None) or \
           (str(row.get("symbol", "")).lower().startswith("adjustment")) or \
           (str(row.get("deal_id", "")) == "" and str(row.get("position_id", "")).lower().startswith("adjust")):
            report.adjustments_ignored += 1
            continue

        required = ["timestamp", "side", "quantity", "entry_price", "exit_price"]
        ok = True
        for key in required:
            v = row.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                # Special case: if we have gross_pnl / pnl AND have exit_price
                # but timestamp is missing, still reject — timestamp required.
                _drop_and_remember(rejected, row, f"missing field: {key}")
                report.rows_rejected_schema += 1
                ok = False
                break
        if ok:
            step1_rows.append(row)

    # ----------------------------------------------------------------------- #
    # Step 2: Type coercion
    # ----------------------------------------------------------------------- #
    step2_rows: List[dict] = []
    for row in step1_rows:
        r = dict(row)
        r["timestamp"]   = _coerce_timestamp(r.get("timestamp"))
        r["side"]        = _coerce_side(r.get("side"))
        r["quantity"]    = _coerce_number(r.get("quantity"))
        r["entry_price"] = _coerce_number(r.get("entry_price"))
        r["exit_price"]  = _coerce_number(r.get("exit_price"))
        r["gross_pnl"]   = _coerce_number(r.get("gross_pnl"))
        r["commission"]  = _coerce_number(r.get("commission")) if r.get("commission") is not None else 0.0
        r["swap"]        = _coerce_number(r.get("swap"))       if r.get("swap")       is not None else 0.0
        r["pnl"]         = _coerce_number(r.get("pnl"))
        r["balance"]     = _coerce_number(r.get("balance"))

        # After coercion, any still-missing required fields -> reject
        if r["timestamp"] is None or r["side"] is None or \
           not np.isfinite(r["quantity"]) or not np.isfinite(r["entry_price"]) or \
           not np.isfinite(r["exit_price"]):
            _drop_and_remember(rejected, row, "type coercion failed on required field")
            report.rows_rejected_schema += 1
            continue
        step2_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 3: Deduplicate (exact-hash)
    # ----------------------------------------------------------------------- #
    seen_hashes = set()
    step3_rows: List[dict] = []
    for r in step2_rows:
        h = hash(tuple(sorted((k, (tuple(v) if isinstance(v, (list, tuple)) else v))
                         for k, v in r.items()
                         if k != "_order")))
        if h in seen_hashes:
            report.rows_deduped += 1
            continue
        seen_hashes.add(h)
        step3_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 4: Stable sort by timestamp ascending
    # ----------------------------------------------------------------------- #
    step4_rows = sorted(step3_rows, key=lambda r: (r["timestamp"], id(r)))

    # ----------------------------------------------------------------------- #
    # Step 5: Negative-quantity flip
    # ----------------------------------------------------------------------- #
    step5_rows = []
    for r in step4_rows:
        if r["quantity"] < 0:
            r["quantity"] = abs(r["quantity"])
            r["side"] = "short" if r["side"] == "long" else "long"
        step5_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 6: PnL imputation (gross + commission + swap if Net missing)
    # ----------------------------------------------------------------------- #
    step6_rows = []
    for r in step5_rows:
        direction = +1 if r["side"] == "long" else -1
        naive = direction * r["quantity"] * (r["exit_price"] - r["entry_price"])
        fees = (r["commission"] if np.isfinite(r["commission"]) else 0.0) + \
               (r["swap"]       if np.isfinite(r["swap"])       else 0.0)

        if (not np.isfinite(r["pnl"])) or r["pnl"] == 0 and abs(naive) > 1e-9:
            if np.isfinite(r["gross_pnl"]):
                r["pnl"] = r["gross_pnl"] + fees
            else:
                r["pnl"] = naive - abs(fees) if fees < 0 else naive - fees
        step6_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 7: Zero-duration filter (flag only, count in report)
    # ----------------------------------------------------------------------- #
    # We don't have an open-timestamp column by default; a synthetic offset
    # (1 s) is applied so duration >= 1 s always. Flag a zero-duration row
    # when a deal has both open_time (if provided) and close_time equal.
    step7_rows = []
    for r in step6_rows:
        # Nothing to drop — just keep a counter for reporting if an explicit
        # open-time column is ever present and matches close-time.
        open_t = r.get("open_time") if isinstance(r.get("open_time"), pd.Timestamp) else None
        if open_t is not None and open_t == r["timestamp"]:
            report.rows_flagged_zero_duration += 1
        step7_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 8: Outlier tagging via |x - median| > 10 * MAD (capped copies for
    #         scoring; originals retained for reporting)
    # ----------------------------------------------------------------------- #
    pnls = np.array([r["pnl"] for r in step7_rows], dtype=float)
    if pnls.size == 0:
        step8_rows = step7_rows
    else:
        med = np.median(pnls)
        mad = np.median(np.abs(pnls - med))
        cap = mad_cap_factor * mad if mad > 0 else 1e-9
        step8_rows = []
        for r in step7_rows:
            if abs(r["pnl"] - med) > cap:
                r = dict(r)
                r["is_outlier"] = True
                report.outliers_tagged += 1
                # Soft-cap (for scoring the outlier set won't be used directly
                # by the user; we leave the original pnl intact and tag.)
            step8_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 9: Balance interpolation — reconstruct synthetic balance series
    # ----------------------------------------------------------------------- #
    # Check if we were given a balance column and it's fully populated
    balances = [r["balance"] for r in step8_rows]
    has_balance_col = all(isinstance(b, (int, float)) and np.isfinite(b) for b in balances) and len(balances) > 0

    trades: List[Trade] = []
    running_balance = synthetic_start_balance
    previous_timestamp = None

    for idx, r in enumerate(step8_rows):
        qty = float(r["quantity"])
        ep  = float(r["entry_price"])
        xp  = float(r["exit_price"])
        entry_notional = qty * ep

        if has_balance_col:
            # Trust the provided balance after the trade
            bal_after = float(balances[idx])
        else:
            running_balance = running_balance + float(r["pnl"])
            bal_after = running_balance

        duration_s = 0.0
        if previous_timestamp is not None:
            try:
                delta: timedelta = (r["timestamp"] - previous_timestamp).to_pydatetime() - pd.Timestamp(0).to_pydatetime()
                duration_s = max(0.0, (r["timestamp"] - previous_timestamp).total_seconds())
            except Exception:
                duration_s = 0.0
        else:
            duration_s = 1.0
        if duration_s <= 0.0:
            duration_s = 1.0
            if hasattr(r, "get") and r.get("open_time") and r.get("open_time") == r["timestamp"]:
                report.rows_flagged_zero_duration += 1

        trades.append(Trade(
            position_id=str(r["position_id"]) if r.get("position_id") else None,
            deal_id=str(r["deal_id"])     if r.get("deal_id")     else None,
            timestamp=r["timestamp"],
            symbol=str(r["symbol"]) if r.get("symbol") else "UNKNOWN",
            side=r["side"],
            quantity=qty,
            entry_price=ep,
            exit_price=xp,
            gross_pnl=float(r["gross_pnl"]) if np.isfinite(r.get("gross_pnl")) else None,
            commission=float(r["commission"]) if np.isfinite(r.get("commission")) else 0.0,
            swap=float(r["swap"])           if np.isfinite(r.get("swap"))       else 0.0,
            pnl=float(r["pnl"]),
            balance_after=bal_after,
            entry_notional=entry_notional,
            is_outlier=bool(r.get("is_outlier", False)),
            duration_seconds=float(duration_s),
        ))
        previous_timestamp = r["timestamp"]

    report.rejected_file = _write_rejected_csv(rejected, job_id, evaluations_dir)

    return trades, report
