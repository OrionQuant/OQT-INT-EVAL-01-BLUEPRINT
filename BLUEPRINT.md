# Strategy Evaluation Tool — Technical Blueprint
**Document:** OQT-INT-EVAL-01-BLUEPRINT  
**Version:** 1.0  
**Date:** 2026-08-02

---

## 1. Architecture Choice & Justification

**Chosen: Option B — FastAPI backend + minimal HTML frontend**

**Justification (1-paragraph):**
Option B is selected because it cleanly separates the pure evaluation engine (Python) from presentation (HTML/JS), making the engine trivially unit-testable and reusable as a library or CLI. FastAPI adds zero overhead for typed request/response contracts, auto-generates an OpenAPI spec for debugging, and serving static HTML is a single line of code. The HTML frontend uses Chart.js for interactive visualisations — lighter than shipping a Tkinter app bundle, easier to demo in a browser, and the same chart objects can be exported to PNG via a single backend endpoint if needed. Reproducibility is enforced by seeding NumPy's RNG at the API boundary.

---

## 2. Project Structure

```
nst/
├── BLUEPRINT.md                    ← This document
├── README.md                       ← Technical write-up (formulas, assumptions)
├── requirements.txt
├── main.py                         ← FastAPI entry point + static page serve
├── seed.py                         ← Reproducibility helpers
│
├── engine/
│   ├── __init__.py
│   ├── models.py                   ← Pydantic schemas (Trade, Metrics, Scorecard, etc.)
│   ├── ingestion.py                ← cTrader report.html parser (BeautifulSoup) + CSV/XLSX fallback
│   ├── cleaning.py                 ← Outlier removal, deduplication, validation
│   ├── metrics.py                  ← Performance & risk metric calculators
│   ├── monte_carlo.py              ← MC simulation engine
│   ├── scoring.py                  ← Weighted scoring model (0–100)
│   └── storage.py                  ← JSON-file save/load/compare evaluations
│
├── static/
│   ├── index.html                  ← Single-page UI (upload, charts, score)
│   ├── app.js                      ← Upload logic, chart rendering, compare view
│   └── styles.css                  ← Minimal styling
│
├── tests/
│   ├── test_ingestion.py           ← cTrader HTML parser tests
│   ├── test_metrics.py             ← Known-value tests for every metric
│   ├── test_monte_carlo.py         ← Seeded reproducibility tests
│   ├── test_scoring.py             ← Weight-sum + knockout + boundary tests
│   └── fixtures/
│       ├── ctrader_sample.html     ← Saved cTrader report snapshot (primary fixture)
│       └── sample_trades.csv       ← Synthetic known-truth trade set (CSV fallback)
│
└── evaluations/                    ← Persisted scorecards (created on first save)
    └── .gitkeep
```

### Module Dependency DAG
```
ingestion → cleaning → metrics → scoring → storage
                   ↓           ↓
              monte_carlo   (visualisation endpoints in main.py)
```
No cycles. `storage` depends on `models` only.

---

## 3. Data Ingestion & Cleaning Pipeline

### 3.1 Supported Input Formats
- **Primary (MANDATORY):** `cTrader report.html` — single-file HTML statement exported from cTrader Desktop / Web ("History" tab → Export → HTML). Parsed with `lxml + BeautifulSoup4` from the `tbody tr` rows inside the History `<table>`.
- **Secondary (fallback):** CSV (UTF-8, comma-delimited, BOM-tolerant) and Excel `.xlsx` (first sheet). Useful once the HTML has been flattened externally.
- **Stretch (not P0):** MetaTrader MT4/MT5 `.htm` / `.html` account statements (table-based layout, separate fields for Order/Deal). Addressed if time permits in Phase P7.

### 3.2 cTrader HTML Table → Canonical Field Mapping
cTrader HTML reports expose a well-known History table schema. The parser maps each `<th>` header to a canonical field by header text (case-insensitive, NBSP-trimmed):

| cTrader HTML Header | Canonical Field | Notes |
|---|---|---|
| `Position` / `Position ID` | `position_id` | String — multiple Deals share one Position (hedging accounts). |
| `Deal` / `Deal ID` | `deal_id` | Unique per row (primary key for dedup). |
| `Time` / `Closing Time` | `timestamp` (close time) | Parsed via `dateutil.parser`; cTrader localised format. Open time sourced from `Open Time` column if present; otherwise `timestamp` minus a synthetic 1s offset (see §3.3 step 7). |
| `Symbol` / `Instrument` | `symbol` | e.g. `EURUSD`, `XAUUSD`, `BTCUSD`. |
| `Type` / `Direction` | `side` | Normalised: `Buy`→`long`, `Sell`→`short`. For in/out position rows we keep the **closing** row only; opening fills without P&L are dropped in §3.3 step 1. |
| `Volume` / `Size` / `Lots` | `quantity` | Lots → units conversion only if a `Symbol Contract Size` summary cell exists at the top of the report; otherwise leave as lots and rely on price × volume for notional. |
| `Open Price` | `entry_price` | float |
| `Close Price` | `exit_price` | float |
| `Gross P&L` / `Gross Profit` | `gross_pnl` | Before fees. |
| `Commission` | `commission` | Usually ≤ 0 (negative = charge). |
| `Swap` / `Rollover` | `swap` | Interest carry; ≤0 for long carry-negative pairs. |
| `Net P&L` / `Profit` / `Net Profit` | `pnl` | Final closed P&L. **Preferred value when present.** If absent, `pnl = gross_pnl + commission + swap` (note signs: commission/swap are typically negative). |
| `Balance` (rare, summary-only) | `balance` | cTrader only shows balance in the footer; per-row balance is reconstructed in §3.3 step 9. |
| *(any bonus/adjustment row)* | `adjustment` | Isolated from trade list; added to final Net PnL sanity check, excluded from metric calculations. |

**Fallback — Column Alias Detection (for CSV/XLSX only):** If the input is not HTML, fall back to a canonical-field → alias-set matcher (case-insensitive, whitespace-trimmed):
`timestamp ← time/date/datetime/open_time/close_time` · `symbol ← pair/instrument/asset/ticker` · `side ← direction/action/type (buy/sell|long/short|1/-1)` · `quantity ← qty/size/amount/units` · `entry_price ← open/price_in/buy_price/entry` · `exit_price ← close/price_out/sell_price/exit` · `pnl ← profit/profit_loss/pl/realized_pnl` · `fees ← commission/cost/slippage` · `balance ← equity/account_value`.

### 3.3 Cleaning Steps (applied in order)
1. **Schema validation** — drop rows missing required fields (`timestamp`, `side`, `quantity`, `entry_price`, `exit_price`). Flag count in report.
2. **Type coercion** — numeric fields → float; timestamps → `datetime64[ns, UTC]`; side → canonical `long`/`short`.
3. **Dedup** — exact duplicate rows dropped (hash of all columns). Count reported.
4. **Sort** — ascending by `timestamp`. If tie, preserve input order (stable sort).
5. **Negative-quantity flip** — if `quantity < 0`, flip sign and invert `side`.
6. **PnL imputation** — if `pnl` missing:
   ```
   direction = +1 if side==long else -1
   pnl = direction * quantity * (exit_price - entry_price) - fees
   ```
7. **Zero-duration filter** — trades with `entry_time == exit_time` flagged (counted, not dropped — user can see in report).
8. **Outlier cap** — PnL values beyond `median ± 10 * MAD` are **not** dropped but tagged as outliers; metrics are computed both ways (capped/uncapped) and the cap-set is used for scoring (defends against one-off data-entry errors). MAD = median absolute deviation.
9. **Balance interpolation** — if `balance` column absent, build synthetic balance starting from 10,000 units and accumulate PnL trade-by-trade.

### 3.4 Validation Failures
Any row failing steps 1–3 is written to `./evaluations/_rejected_rows_{job_id}.csv` and a warning is surfaced. The job proceeds with the remainder.

---

## 4. Performance & Risk Metrics

All metrics are computed on the **cleaned, ordered trade list**. Let:
- `N` = number of trades
- `r_i` = PnL of trade `i`
- `R_i` = **return** of trade `i`. If balance series exists: `R_i = r_i / balance_before_i`; else `R_i = r_i / entry_notional_i` (entry_notional = quantity × entry_price).
- `B_t` = balance/equity at step `t` (after `t` trades)
- `B_0` = starting balance (actual or synthetic 10,000)

### 4.1 Basic Counts & Win/Loss
| Metric | Formula | Notes |
|---|---|---|
| Total Trades | `N` | |
| Win Rate `WR` | `(# wins) / N` where win = `r_i > 0` | Ties (`r_i==0`) excluded from numerator, kept in denominator. |
| Loss Rate | `1 - WR` | |
| Win Count / Loss Count | | |
| Average Win `AW` | `mean(r_i for wins)` | |
| Average Loss `AL` | `mean(abs(r_i) for losses)` | |
| Profit Factor `PF` | `sum(wins) / max(epsilon, sum(abs(losses)))` | `epsilon = 1e-9` to avoid div-by-zero. Capped at 50 for scoring. |
| Payoff Ratio `PR` | `AW / max(epsilon, AL)` | |
| Expectancy per Trade | `WR * AW - (1-WR) * AL` | Currency units. |
| Expectancy per $ Risked | `(WR * AW - (1-WR) * AL) / max(epsilon, AL)` | Dimensionless. |
| Long/Short Split | count and WR by side | |

### 4.2 Returns & Growth
| Metric | Formula | Notes |
|---|---|---|
| Net PnL | `Σ r_i` | |
| Total Return `TR` | `(B_N - B_0) / B_0` | Decimal, not %.
| CAGR | `(B_N / B_0)^(1/T_years) - 1` | `T_years = (last_ts - first_ts).days / 365.25`. If T < 7 days, fall back to TR (flag in report). |
| Mean Return per Trade | `mean(R_i)` | |
| Median Return per Trade | `median(R_i)` | Robust to outliers. |
| Skew of Returns | `skew(R_i)` | Positive skew preferred. |
| Kurtosis of Returns | `kurtosis(R_i)` | Excess kurtosis > 0 → fat tails → flag. |

### 4.3 Drawdown
Define running peak `P_t = max(B_0 … B_t)`; drawdown `DD_t = (B_t - P_t) / P_t`.
| Metric | Formula |
|---|---|
| Max Drawdown `MDD` | `min(DD_t)` (negative number; reported as magnitude in UI) |
| Average Drawdown | `mean(DD_t for DD_t < 0)` |
| Drawdown Duration (max) | Longest contiguous run of `DD_t < 0`, in calendar days |
| Ulcer Index `UI` | `sqrt(mean(DD_t^2))` |
| Recovery Factor | `Net PnL / max(epsilon, abs(MDD * B_0))` |

> ⚠ **DOCUMENTED LIMITATION — CLOSED-TRADE BALANCE DRAWDOWN** (Non-Negotiable, Brief §7 & Data-Honesty clause)
> Balance-based drawdown computed above operates **exclusively on closed trade P&L** (either as reported in `Net P&L` of the cTrader report, or reconstructed from `gross_pnl + commission + swap`). It **does not** capture intra-trade floating losses, unrealized equity swings, margin-equity drawdowns during open positions, or the worst-tick equity trough that occurred *between* trade open and close. **Consequence:** for strategies that (a) hold losing trades open for extended periods before closing for a smaller realised loss, (b) average-in across multiple overlapping open positions in the same basket, or (c) run with tight stop-losses that are rarely hit *on close*, the §4.3 drawdown numbers will **understate the true peak-to-trough risk**. Capital-preservation and behavioural metrics should therefore be read with this bias in mind; the §6 scoring model compensates for the optimism by using conservative (reverse-sigmoid) MDD normalisation and a hard 35 % MDD knockout in §6.3.

### 4.4 Risk-Adjusted Returns
Assumptions for ratios that need a "risk-free rate" `rf`:
- Default `rf = 0.04` annualised (4%). Overridable in request.
- Trade-level conversion: `rf_per_trade = (1 + rf)^(1/avg_trades_per_year) - 1`.
- If < 10 trades, skip annualisation entirely (return "insufficient data" and use raw excess return).

| Metric | Formula | Notes |
|---|---|---|
| Sharpe Ratio `SR` | `(mean(R_i - rf_per_trade) * K) / std(R_i - rf_per_trade)` | `K = sqrt(avg_trades_per_year)` for annualisation. If `N < 30`, divide SR by `sqrt(30/N)` correction (small-sample bias). |
| Sortino Ratio `SoR` | `(mean(R_i - rf_per_trade) * K) / downside_dev` | `downside_dev = std(min(R_i - rf_per_trade, 0))` |
| Calmar Ratio | `CAGR / abs(MDD)` | |
| Omega Ratio | `(Σ max(R_i - τ, 0)) / (Σ max(τ - R_i, 0))` | Threshold `τ = rf_per_trade`. Measures upside volume vs downside volume. |
| Profit / Max-DD (RoMaD) | `TR / abs(MDD)` | Simpler alternative to Calmar. |
| Tail Ratio | `abs(percentile(R_i, 95)) / abs(percentile(R_i, 5))` | >1 means right tail fatter than left. Good. |
| VaR (95%, historical) | `-percentile(R_i, 5)` | Worst 5% loss, per-trade. |
| CVaR/ES (95%) | `-mean(R_i where R_i ≤ percentile(R_i, 5))` | Expected shortfall. |

### 4.5 Behavioural / Consistency
| Metric | Formula |
|---|---|
| Win Streak (max) | Longest consecutive winning run |
| Loss Streak (max) | Longest consecutive losing run |
| Profit Streak Ratio | `max_win_streak / max(1, max_loss_streak)` |
| Z-Score (runs test) | `(Runs - μ) / σ` where `μ = 1 + 2·Nw·Nl/N`, `σ² = (μ-1)(μ-2)/(N-1)`. | Tests for serial correlation of wins/losses. \|Z\| < 1.96 → no evidence of autocorrelation (good). |
| Best Trade / Worst Trade | `max(r_i)`, `min(r_i)` |
| % Profitable Months | (requires grouping by month) |
| Monthly Return Volatility | `std(monthly_returns)` |

---

## 5. Monte Carlo Robustness Analysis

### 5.1 Purpose
Answer: *"How sensitive is the reported performance to the ordering and subset of trades we happened to get?"*

### 5.2 Three Complementary Tests
**All three are seeded with `numpy.random.default_rng(seed)` and produce identical output on same input + seed.**

#### Test A — Bootstrap Resampling (with replacement)
- `M = 1000` iterations.
- Each iteration: sample `N` trades **with replacement** from the cleaned list → synthetic equity curve.
- Compute per-iteration: `TR_i`, `MDD_i`, `SR_i`, `PF_i`.
- Outputs:
  - Distribution histograms + 5/25/50/75/95 percentiles for each metric.
  - **Robustness Rate (RR):** `# iterations where TR > 0 / M` — probability strategy is profitable on resampled history.
  - **Sharpe Stability:** `# iterations where SR > 1.0 / M`.

> ⚠ **DOCUMENTED CAVEAT — TRADE INDEPENDENCE ASSUMPTION** (Brief §8, non-negotiable)
> Test A performs **naive per-trade resampling with replacement** and therefore assumes trade returns are **independent and identically distributed (i.i.d.)**. This assumption is violated for any strategy that:
> 1.  holds **multiple overlapping open positions at the same time** (e.g. basket strategies, portfolio rebalancing, multi-symbol market-neutral books), because shuffling destroys cross-symbol return correlation realised at the same timestamp; or
> 2.  pyramid / scale-in / scale-out of one position via multiple partial deals (Deal IDs sharing a Position ID), because resampling can break a single position's entry/exit composition and create synthetic position sizes that never existed in reality.
>
> In either case, Test A will produce **artificially optimistic risk distributions** (thinner left tail of MDD, inflated Robustness Rate) because diversification benefit is sampled as-if noise-free. **Mitigation & prescribed reading:** always read Test A distributions alongside **Test C (Block Bootstrap, Politis–White stationary bootstrap with mean block length L = √N)**, which preserves short-range temporal and cross-correlation structure by resampling contiguous trade blocks. If Test C's 5th-percentile MDD is >1.5× worse than Test A's, this is flagged in the scorecard as `"correlation_flag": true`, and the MC Robustness Rate used in §6 scoring is drawn **from Test C, not Test A**, to avoid penalising the user on the optimistic estimate.

#### Test B — Shuffled Order (permutation, no replacement)
- `M = 1000` iterations.
- Each iteration: random permutation of the same N trades → equity curve.
- Purpose: isolate **path-dependency / compounding effects**. Same trades, different order.
- Outputs:
  - Distribution of `MDD` (ordering matters a lot for drawdown).
  - Distribution of final balance (should be *exactly* the same every iteration in theory — if not, we have a bug → sanity check).
  - **Drawdown Sensitivity:** `P(MDD_iteration > 2 × actual_MDD)` — probability unlucky ordering doubles your worst drawdown.

#### Test C — Block Bootstrap (preserves serial correlation)
- Same as Test A but resample contiguous **blocks** of mean length `L = sqrt(N)` (Politis–White stationary bootstrap).
- Defends against the runs-test concern: if wins cluster, i.i.d. bootstrap overestimates diversity.
- Outputs same percentiles as Test A. If Test C percentiles are **much worse** than Test A, the strategy has unfavourable serial correlation → flagged.

### 5.3 Reproducibility Contract
```python
# seed.py
import numpy as _np

def make_rng(seed: int | None) -> tuple[_np.random.Generator, int]:
    if seed is None:
        seed = _np.random.SeedSequence().entropy  # ← logged in result
    return _np.random.default_rng(seed), seed
```
The returned `seed` is **always** stored in the scorecard JSON. The frontend shows it and lets the user paste a seed to re-run.

---

## 6. Weighted Scoring Model (0–100)

### 6.1 Design Principles
... (full document continues in repo)
