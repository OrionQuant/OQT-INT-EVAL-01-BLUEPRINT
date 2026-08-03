"""Data cleaning pipeline — blueprint §3.3 steps (1)…(9)."""

from __future__ import annotations

import csv as _csv
import os
import uuid
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
    """Parse a number that may use US (1,000.00) or European (1.000,00) formatting.

    Delegates to the engine-level number parser for consistency with HTML
    ingestion (single source of truth for locale-aware parsing).
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    from .ingestion import _parse_number
    out = _parse_number(x)
    return np.nan if out is None else float(out)


def _coerce_timestamp(raw) -> pd.Timestamp | None:
    """Parse a timestamp with European-format preference (dayfirst=True, Fix #3)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(raw)
    elif isinstance(raw, str):
        # Use the engine-level EU parser for string -> datetime preference
        from .ingestion import _parse_timestamp_eu
        out = _parse_timestamp_eu(raw)
        if out is None:
            return None
        ts = pd.Timestamp(out)
    else:
        try:
            n = float(raw)
        except (ValueError, TypeError):
            return None
        if 1_000 < n < 100_000:
            ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=n)
        else:
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
    # Step 3: Deduplicate (exact-hash), while preserving original input index
    #         so later sort order can be deterministic across runs (Fix #5).
    # ----------------------------------------------------------------------- #
    seen_hashes = set()
    step3_rows: List[dict] = []
    for order, r in enumerate(step2_rows):
        if "_input_idx" not in r:
            r["_input_idx"] = order
        h = hash(
            tuple(
                sorted(
                    (
                        k,
                        (tuple(v) if isinstance(v, (list, tuple)) else v),
                    )
                    for k, v in r.items()
                    if k not in ("_order", "_input_idx")
                )
            )
        )
        if h in seen_hashes:
            report.rows_deduped += 1
            continue
        seen_hashes.add(h)
        step3_rows.append(r)

    # ----------------------------------------------------------------------- #
    # Step 4: Stable deterministic sort by timestamp + original input order
    #         (Fix #5 — use _input_idx, not id(r), for cross-run determinism)
    # ----------------------------------------------------------------------- #
    step4_rows = sorted(step3_rows, key=lambda r: (r["timestamp"], int(r.get("_input_idx", 0))))

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
    # Fix #7:
    #   * Parens around the `r["pnl"] == 0` check so NaN OR (zero with
    #     plausible naive) triggers the branch.
    #   * Consistent fee sign: commission and swap are almost always recorded
    #     as negative values (costs) in broker exports. `fees = commission +
    #     swap` is negative-valued; to go from gross→net we ADD the (negative)
    #     fee row, which is equivalent to subtracting the absolute cost.
    #     Never abs() fees or subtract fees twice.
    # ----------------------------------------------------------------------- #
    step6_rows = []
    for r in step5_rows:
        direction = +1 if r["side"] == "long" else -1
        naive = direction * r["quantity"] * (r["exit_price"] - r["entry_price"])
        commission = float(r["commission"]) if np.isfinite(r.get("commission")) else 0.0
        swap = float(r["swap"]) if np.isfinite(r.get("swap")) else 0.0
        fees = commission + swap  # will typically be negative-valued

        pnl_missing = not np.isfinite(r["pnl"])
        pnl_zero_but_data_available = (
            (np.isfinite(r["pnl"])) and abs(float(r["pnl"])) < 1e-12 and
            (abs(naive) > 1e-9 or (np.isfinite(r.get("gross_pnl")) and abs(float(r["gross_pnl"])) > 1e-9))
        )
        if pnl_missing or pnl_zero_but_data_available:
            if np.isfinite(r.get("gross_pnl")):
                # gross_pnl already excludes cost (or is explicitly gross)
                # net = gross + fees (since fees are stored as negative)
                r["pnl"] = float(r["gross_pnl"]) + fees
            else:
                # naive (raw price × qty) is gross-of-fees; go net by adding fees
                r["pnl"] = naive + fees
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
    # Step 8: Outlier tagging via |x - median| > K * MAD, plus soft capping.
    # Fix #8:
    #   * Every trade row receives a `capped_pnl` value clamped to
    #     [median ∓ K·MAD] regardless of outlier status.
    #   * Rows exceeding the threshold receive `is_outlier=True` + count bump.
    #   * Downstream metrics.py and monte_carlo.py consume capped_pnl when
    #     computing any series the scoring uses for grading. Raw `pnl` stays
    #     attached for audit / display transparency.
    # ----------------------------------------------------------------------- #
    pnls = np.array([r["pnl"] for r in step7_rows], dtype=float)
    if pnls.size == 0:
        step8_rows = step7_rows
    else:
        med = float(np.median(pnls))
        mad = float(np.median(np.abs(pnls - med)))
        cap = mad_cap_factor * mad if mad > 0 else 1e-9
        lower = med - cap
        upper = med + cap
        step8_rows = []
        for raw_r in step7_rows:
            r = dict(raw_r)
            x = float(r["pnl"])
            r["capped_pnl"] = float(max(lower, min(upper, x)))
            if abs(x - med) > cap:
                r["is_outlier"] = True
                report.outliers_tagged += 1
            else:
                r.setdefault("is_outlier", False)
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
            bal_after = float(balances[idx])
        else:
            running_balance = running_balance + float(r["pnl"])
            bal_after = running_balance

        # Fix #9: compute inter-trade interval as a simple total_seconds().
        # The previous code subtracted pd.Timestamp(0) and called to_pydatetime()
        # on a Timestamp (which is already a timestamp delta on the left-hand
        # result), always threw and was swallowed by `except Exception: pass`.
        duration_s = 0.0
        if previous_timestamp is not None:
            try:
                diff = r["timestamp"] - previous_timestamp
                if hasattr(diff, "total_seconds"):
                    duration_s = float(diff.total_seconds())
                else:
                    duration_s = 0.0
            except (TypeError, ValueError, AttributeError, ArithmeticError):
                duration_s = 0.0
        else:
            # First trade: synthetic 1s anchor so we never report 0 for row 1
            duration_s = 1.0
        if duration_s <= 0.0:
            duration_s = 1.0
            if hasattr(r, "get") and r.get("open_time") and r.get("open_time") == r["timestamp"]:
                report.rows_flagged_zero_duration += 1

        capped_pnl = float(r.get("capped_pnl", r["pnl"]))

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
            capped_pnl=capped_pnl,
            balance_after=bal_after,
            entry_notional=entry_notional,
            is_outlier=bool(r.get("is_outlier", False)),
            duration_seconds=float(duration_s),
        ))
        previous_timestamp = r["timestamp"]

    report.rejected_file = _write_rejected_csv(rejected, job_id, evaluations_dir)

    return trades, report
