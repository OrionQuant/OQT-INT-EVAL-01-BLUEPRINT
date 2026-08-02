"""Performance & risk metric calculators — blueprint §4.1 … §4.5.

Given a list of cleaned, ordered Trade objects, computes the full set of
metrics, an equity curve, and monthly returns aggregations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import (
    BasicMetrics,
    BehaviouralMetrics,
    DrawdownMetrics,
    EquityPoint,
    GrowthMetrics,
    Metrics,
    RiskAdjustedMetrics,
    Trade,
)

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _trade_arrays(trades: List[Trade]) -> Dict[str, np.ndarray]:
    """Convert a Trade list to aligned NumPy arrays for vectorised math."""
    n = len(trades)
    pnl = np.zeros(n, dtype=float)
    sides = np.zeros(n, dtype=object)
    qty = np.zeros(n, dtype=float)
    ep = np.zeros(n, dtype=float)
    xp = np.zeros(n, dtype=float)
    entry_notional = np.zeros(n, dtype=float)
    balance_after = np.zeros(n, dtype=float)
    timestamps = np.empty(n, dtype="datetime64[ns]")
    symbols = np.zeros(n, dtype=object)

    for i, t in enumerate(trades):
        pnl[i] = t.pnl
        sides[i] = t.side
        qty[i] = t.quantity
        ep[i] = t.entry_price
        xp[i] = t.exit_price
        entry_notional[i] = t.entry_notional
        balance_after[i] = t.balance_after if t.balance_after is not None else np.nan
        timestamps[i] = pd.Timestamp(t.timestamp).to_datetime64()
        symbols[i] = t.symbol

    return {
        "pnl": pnl,
        "sides": sides,
        "qty": qty,
        "ep": ep,
        "xp": xp,
        "entry_notional": entry_notional,
        "balance_after": balance_after,
        "timestamps": timestamps,
        "symbols": symbols,
    }


def _build_balance_series(arr: dict, start_balance: float) -> np.ndarray:
    """Reconstruct balance_before_i + balance_after_i per trade.

    Uses the provided balance_after column if every value is finite; otherwise
    accumulates PnL from *start_balance*. Returns ``balance_before`` (length N)
    aligned to each trade.
    """
    ba = arr["balance_after"]
    pnl = arr["pnl"]
    if np.all(np.isfinite(ba)) and ba.size > 0:
        balance_before = np.empty_like(ba)
        balance_before[0] = ba[0] - pnl[0]
        balance_before[1:] = ba[:-1]
        return balance_before
    balance_before = np.empty_like(pnl)
    running = start_balance
    for i in range(pnl.size):
        balance_before[i] = running
        running += pnl[i]
    return balance_before


def _per_trade_returns(arr: dict, balance_before: np.ndarray) -> np.ndarray:
    """R_i from blueprint §4 intro.

    If balance series exists: R_i = r_i / balance_before_i; else
    R_i = r_i / entry_notional_i (with a guard against zero notional).
    """
    pnl = arr["pnl"]
    en = arr["entry_notional"]
    # Treat the balance series as real only if all balance_before are sensible
    if np.all(np.isfinite(balance_before)) and np.all(np.abs(balance_before) > EPS):
        return pnl / balance_before
    denom = np.where(np.abs(en) > EPS, en, 1.0)
    return pnl / denom


def _equity_curve_from_pnl(
    timestamps: np.ndarray,
    pnl: np.ndarray,
    start_balance: float,
) -> List[EquityPoint]:
    """Build (timestamp, balance, peak, dd) equity curve points."""
    balance = np.empty(pnl.size + 1, dtype=float)
    balance[0] = start_balance
    balance[1:] = start_balance + np.cumsum(pnl)
    peak = np.maximum.accumulate(balance)
    dd = (balance - peak) / np.where(peak > 0, peak, 1.0)

    points: List[EquityPoint] = []
    # Insert the synthetic starting point at the earliest timestamp - 1s
    if timestamps.size > 0:
        t0 = pd.Timestamp(timestamps[0]) - pd.Timedelta(seconds=1)
        points.append(EquityPoint(t=t0.to_pydatetime(), balance=float(balance[0]),
                                  peak=float(peak[0]), dd=float(dd[0])))
    for i in range(pnl.size):
        ti = pd.Timestamp(timestamps[i]).to_pydatetime()
        points.append(EquityPoint(t=ti, balance=float(balance[i + 1]),
                                  peak=float(peak[i + 1]), dd=float(dd[i + 1])))
    return points


def _monthly_returns(equity: List[EquityPoint]) -> Dict[str, float]:
    if len(equity) < 2:
        return {}
    rows = [(pd.Timestamp(e.t), e.balance) for e in equity]
    df = pd.DataFrame(rows, columns=["t", "b"]).set_index("t")
    month_end = df.resample("ME").last()
    month_end = month_end[month_end["b"].notna()]
    if len(month_end) < 2:
        return {}
    rets = month_end["b"].pct_change().dropna()
    out: Dict[str, float] = {}
    for ts, r in rets.items():
        key = ts.strftime("%Y-%m")
        out[key] = float(r)
    return out


# --------------------------------------------------------------------------- #
# §4.1 Basic counts & Win/Loss
# --------------------------------------------------------------------------- #


def _compute_basic(arr: dict) -> BasicMetrics:
    pnl = arr["pnl"]
    sides = arr["sides"]
    n = pnl.size

    wins_mask = pnl > 0
    losses_mask = pnl < 0
    ties_mask = pnl == 0

    nw = int(np.sum(wins_mask))
    nl = int(np.sum(losses_mask))
    nt = int(np.sum(ties_mask))

    wr = (nw / n) if n > 0 else 0.0
    lr = 1.0 - wr

    aw = float(np.mean(pnl[wins_mask])) if nw > 0 else 0.0
    al = float(np.mean(np.abs(pnl[losses_mask]))) if nl > 0 else 0.0

    sum_wins = float(np.sum(pnl[wins_mask])) if nw > 0 else 0.0
    sum_losses = float(np.sum(np.abs(pnl[losses_mask]))) if nl > 0 else 0.0
    pf = sum_wins / max(EPS, sum_losses)
    pf = min(pf, 50.0)  # cap for scoring

    pr = aw / max(EPS, al)
    expect_per_trade = wr * aw - (1 - wr) * al
    expect_per_risk = expect_per_trade / max(EPS, al)

    long_mask = sides == "long"
    short_mask = sides == "short"
    lc = int(np.sum(long_mask))
    sc = int(np.sum(short_mask))
    lwr = (float(np.sum(pnl[long_mask & wins_mask])) / max(1, lc)) / max(EPS, 1.0)
    # Proper per-side win rate
    l_wins = int(np.sum(long_mask & wins_mask))
    s_wins = int(np.sum(short_mask & wins_mask))
    lwr = (l_wins / lc) if lc > 0 else 0.0
    swr = (s_wins / sc) if sc > 0 else 0.0

    return BasicMetrics(
        total_trades=n,
        win_rate=wr,
        loss_rate=lr,
        win_count=nw,
        loss_count=nl,
        tie_count=nt,
        average_win=aw,
        average_loss=al,
        profit_factor=pf,
        payoff_ratio=pr,
        expectancy_per_trade=expect_per_trade,
        expectancy_per_unit_risk=expect_per_risk,
        long_count=lc,
        short_count=sc,
        long_win_rate=lwr,
        short_win_rate=swr,
    )


# --------------------------------------------------------------------------- #
# §4.2 Returns & Growth
# --------------------------------------------------------------------------- #


def _compute_growth(arr: dict, R: np.ndarray, start_balance: float, end_balance: float) -> GrowthMetrics:
    pnl = arr["pnl"]
    timestamps = arr["timestamps"]
    n = pnl.size

    net_pnl = float(np.sum(pnl))
    tr = (end_balance - start_balance) / max(EPS, start_balance)

    cagr: float | None = None
    cagr_flagged = False
    if n >= 2:
        t_first = pd.Timestamp(timestamps[0])
        t_last = pd.Timestamp(timestamps[-1])
        t_years = (t_last - t_first).days / 365.25
        if t_years >= 7 / 365.25:
            if end_balance > 0 and start_balance > 0:
                cagr = (end_balance / start_balance) ** (1.0 / t_years) - 1.0
        else:
            cagr_flagged = True

    mean_r = float(np.mean(R)) if R.size > 0 else 0.0
    median_r = float(np.median(R)) if R.size > 0 else 0.0

    if R.size >= 3:
        try:
            from scipy.stats import skew as _sskew, kurtosis as _skurt
            skew = float(_sskew(R, bias=False))
            ekurt = float(_skurt(R, fisher=True, bias=False))
        except Exception:
            skew = float(_skew_fallback(R))
            ekurt = float(_kurtosis_fallback(R))
    else:
        skew = 0.0
        ekurt = 0.0

    return GrowthMetrics(
        net_pnl=net_pnl,
        total_return=tr,
        cagr=cagr,
        cagr_flagged_insufficient=cagr_flagged,
        mean_return_per_trade=mean_r,
        median_return_per_trade=median_r,
        skew_returns=skew,
        excess_kurtosis_returns=ekurt,
    )


def _skew_fallback(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < EPS:
        return 0.0
    n = x.size
    return float(np.sum(((x - m) / s) ** 3) * n / ((n - 1) * (n - 2)))


def _kurtosis_fallback(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < EPS:
        return 0.0
    n = x.size
    k = (np.sum(((x - m) / s) ** 4) * n * (n + 1) /
         ((n - 1) * (n - 2) * (n - 3))) - \
        3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float(k)


# --------------------------------------------------------------------------- #
# §4.3 Drawdown
# --------------------------------------------------------------------------- #


def _compute_drawdown(arr: dict, balance_before: np.ndarray, pnl: np.ndarray, start_balance: float) -> DrawdownMetrics:
    # Build equity series B_t with starting point
    equity = np.empty(pnl.size + 1, dtype=float)
    equity[0] = start_balance
    equity[1:] = start_balance + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd_arr = (equity - peak) / np.where(peak > 0, peak, 1.0)

    mdd = float(np.min(dd_arr)) if dd_arr.size else 0.0
    dd_neg = dd_arr[dd_arr < 0]
    avg_dd = float(np.mean(dd_neg)) if dd_neg.size > 0 else 0.0

    # Drawdown duration: longest contiguous run of DD_t < 0, in calendar days
    timestamps = arr["timestamps"]
    max_dur_days = 0.0
    if timestamps.size > 0:
        # Align timestamps to dd_arr (dd_arr[0] is start, dd_arr[1:] after trades)
        all_ts = np.empty(dd_arr.size, dtype="datetime64[ns]")
        all_ts[0] = pd.Timestamp(timestamps[0]) - pd.Timedelta(seconds=1)
        all_ts[1:] = timestamps
        run_start: int | None = None
        for i in range(dd_arr.size):
            if dd_arr[i] < 0:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    dur = (pd.Timestamp(all_ts[i - 1]) - pd.Timestamp(all_ts[run_start])).total_seconds() / 86400.0
                    max_dur_days = max(max_dur_days, dur)
                    run_start = None
        if run_start is not None:
            dur = (pd.Timestamp(all_ts[-1]) - pd.Timestamp(all_ts[run_start])).total_seconds() / 86400.0
            max_dur_days = max(max_dur_days, dur)

    ui = float(np.sqrt(np.mean(dd_arr ** 2))) if dd_arr.size > 0 else 0.0
    net_pnl_total = float(np.sum(pnl))
    rf = net_pnl_total / max(EPS, abs(mdd) * start_balance)

    return DrawdownMetrics(
        max_drawdown=mdd,
        average_drawdown=avg_dd,
        max_drawdown_duration_days=float(max_dur_days),
        ulcer_index=ui,
        recovery_factor=rf,
    )


# --------------------------------------------------------------------------- #
# §4.4 Risk-Adjusted Returns
# --------------------------------------------------------------------------- #


def _compute_risk_adj(
    arr: dict,
    R: np.ndarray,
    mdd: float,
    tr: float,
    cagr: float | None,
    rf_annual: float = 0.04,
) -> RiskAdjustedMetrics:
    timestamps = arr["timestamps"]
    pnl = arr["pnl"]
    n = R.size

    # avg trades per year
    if n >= 2 and timestamps.size >= 2:
        t_first = pd.Timestamp(timestamps[0])
        t_last = pd.Timestamp(timestamps[-1])
        days = max(1.0, (t_last - t_first).days)
        atpy = n * 365.25 / days
    else:
        atpy = n

    annualised = n >= 10
    if annualised and atpy > EPS:
        rf_per_trade = (1.0 + rf_annual) ** (1.0 / atpy) - 1.0
    else:
        rf_per_trade = 0.0

    excess = R - rf_per_trade
    mu = float(np.mean(excess)) if excess.size > 0 else 0.0
    sigma = float(np.std(excess, ddof=1)) if excess.size >= 2 else 0.0
    K = float(np.sqrt(atpy)) if annualised else 1.0

    sr_raw = (mu * K) / sigma if sigma > EPS else 0.0
    # small-sample correction (blueprint §4.4)
    small_sample_corr = False
    if n < 30:
        sr_raw = sr_raw * (n / 30.0) ** 0.5
        small_sample_corr = True

    # downside deviation: min(R_i - rf, 0) std
    downside = np.minimum(excess, 0.0)
    dd_sigma = float(np.std(downside, ddof=1)) if downside.size >= 2 else 0.0
    sor = (mu * K) / dd_sigma if dd_sigma > EPS else 0.0

    calmar: float | None = None
    if cagr is not None and mdd < 0:
        calmar = cagr / abs(mdd)

    # Omega ratio at τ = rf_per_trade
    upside = np.sum(np.maximum(R - rf_per_trade, 0.0))
    downside_vol = np.sum(np.maximum(rf_per_trade - R, 0.0))
    omega = float(upside / max(EPS, downside_vol))

    romad = tr / abs(mdd) if mdd < 0 else 0.0

    if R.size >= 10:
        p5 = float(np.percentile(R, 5))
        p95 = float(np.percentile(R, 95))
    elif R.size >= 2:
        p5 = float(np.percentile(R, 0)) if R.size < 10 else float(np.percentile(R, 5))
        p95 = float(np.percentile(R, 100)) if R.size < 10 else float(np.percentile(R, 95))
    else:
        p5, p95 = 0.0, 0.0

    tail_ratio = abs(p95) / max(EPS, abs(p5))

    var_95 = -float(p5)
    tail = R[R <= p5]
    cvar_95 = -float(np.mean(tail)) if tail.size > 0 else 0.0

    return RiskAdjustedMetrics(
        sharpe_ratio=float(sr_raw),
        sharpe_small_sample_corrected=small_sample_corr,
        sortino_ratio=float(sor),
        calmar_ratio=calmar,
        omega_ratio=omega,
        romad=float(romad),
        tail_ratio=float(tail_ratio),
        var_95=float(var_95),
        cvar_95=float(cvar_95),
        annualised=annualised,
        avg_trades_per_year=float(atpy),
    )


# --------------------------------------------------------------------------- #
# §4.5 Behavioural / Consistency
# --------------------------------------------------------------------------- #


def _compute_behavioural(arr: dict, monthly_returns: Dict[str, float]) -> BehaviouralMetrics:
    pnl = arr["pnl"]
    n = pnl.size

    # Streaks — operate on wins_mask
    wins = (pnl > 0).astype(int)
    losses = (pnl < 0).astype(int)

    def _longest_run(mask: np.ndarray) -> int:
        if mask.size == 0:
            return 0
        best = cur = 0
        for v in mask:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    max_ws = _longest_run(wins)
    max_ls = _longest_run(losses)
    psr = max_ws / max(1, max_ls)

    # Runs-test Z score (Wald–Wolfowitz)
    nw = int(np.sum(wins))
    nl = int(np.sum(losses))
    N_rl = nw + nl
    runs = 1
    if N_rl >= 2:
        last = wins[0] if wins[0] == 1 else (losses[0] * 2)
        for i in range(1, n):
            cur = wins[i] if wins[i] == 1 else (losses[i] * 2)
            if cur != last and cur != 0 and last != 0:
                runs += 1
            if cur != 0:
                last = cur
    if N_rl >= 2 and nw > 0 and nl > 0:
        mu_runs = 1.0 + 2.0 * nw * nl / N_rl
        var_runs = (mu_runs - 1.0) * (mu_runs - 2.0) / (N_rl - 1.0) if N_rl > 1 else 0.0
        z = (runs - mu_runs) / (var_runs ** 0.5) if var_runs > EPS else 0.0
    else:
        z = 0.0

    best_trade = float(np.max(pnl)) if n > 0 else 0.0
    worst_trade = float(np.min(pnl)) if n > 0 else 0.0

    mr_values = list(monthly_returns.values())
    if mr_values:
        pmf = float(np.mean([1.0 if r > 0 else 0.0 for r in mr_values]))
        mrv = float(np.std(mr_values, ddof=1)) if len(mr_values) >= 2 else None
    else:
        pmf = None
        mrv = None

    return BehaviouralMetrics(
        max_win_streak=int(max_ws),
        max_loss_streak=int(max_ls),
        profit_streak_ratio=float(psr),
        runs_z_score=float(z),
        best_trade=best_trade,
        worst_trade=worst_trade,
        profitable_month_fraction=pmf,
        monthly_return_volatility=mrv,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def compute_all_metrics(
    trades: List[Trade],
    *,
    start_balance: float = 10_000.0,
    rf_annual: float = 0.04,
) -> Tuple[Metrics, List[EquityPoint], Dict[str, float]]:
    """Compute every metric group plus equity curve and monthly returns.

    Returns ``(metrics, equity_curve, monthly_returns)``.
    """
    if not trades:
        raise ValueError("No trades provided — cannot compute metrics.")

    arr = _trade_arrays(trades)
    balance_before = _build_balance_series(arr, start_balance)
    R = _per_trade_returns(arr, balance_before)

    pnl = arr["pnl"]
    end_balance = start_balance + float(np.sum(pnl))

    basic = _compute_basic(arr)
    growth = _compute_growth(arr, R, start_balance, end_balance)
    dd = _compute_drawdown(arr, balance_before, pnl, start_balance)
    ra = _compute_risk_adj(arr, R, dd.max_drawdown, growth.total_return, growth.cagr, rf_annual=rf_annual)

    equity = _equity_curve_from_pnl(arr["timestamps"], pnl, start_balance)
    monthly = _monthly_returns(equity)

    behav = _compute_behavioural(arr, monthly)

    metrics = Metrics(basic=basic, growth=growth, drawdown=dd, risk_adj=ra, behav=behav)
    return metrics, equity, monthly
