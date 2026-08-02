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
| Total Return `TR` | `(B_N - B_0) / B_0` | Decimal, not %. |
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
> Balance-based drawdown computed above operates **exclusively on closed trade P&L** (either as reported in `Net P&L` of the cTrader report, or reconstructed from `gross_pnl + commission + swap`). It **does not** capture intra-trade floating losses, unrealized equity swings, margin-equity drawdowns during open positions, stop-tolerance violations, or the worst-tick equity trough that occurred *between* trade open and close. **Consequence:** for strategies that (a) hold losing trades open for extended periods before closing for a smaller realised loss, (b) average-in across multiple overlapping open positions in the same basket, or (c) run with tight stop-losses that are rarely hit *on close*, the §4.3 drawdown numbers will **understate the true peak-to-trough risk**. Capital-preservation and behavioural metrics should therefore be read with this bias in mind; the §6 scoring model compensates for the optimism by using conservative (reverse-sigmoid) MDD normalisation and a hard 35 % MDD knockout in §6.3.

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
- Each metric is **normalised to [0, 10] sub-scores** using a bounded sigmoid or piecewise-linear function (no arbitrary hard cutoffs).
- Weights sum to 1. Final score = `Σ w_k · subscore_k`.
- **Penalties** (drawdown, kurtosis, loss-streak) are *multiplicative* on top of the weighted sum — so a great PF cannot paper over catastrophic drawdown.
- All thresholds are documented and overridable in a `ScoringConfig` JSON.

### 6.2 Category Weights
| Category | Weight `w_cat` | Rationale |
|---|---|---|
| **1. Profitability** | 0.25 | Without positive expectancy nothing else matters. |
| **2. Risk-Adjusted Return** | 0.25 | Sharpe/Sortino/Calmar — the industry standard quality axis. |
| **3. Drawdown & Capital Preservation** | 0.20 | Survive first, optimise second. |
| **4. Consistency & Robustness** | 0.15 | MC robustness rate, runs test, month-to-month stability. |
| **5. Behavioural Sanity** | 0.10 | Fat tails, worst loss, skewness — things that blow up strategies in real markets. |
| **6. Statistical Sufficiency** | 0.05 | N ≥ 30? Outlier fraction? Penalise tiny samples. |
| **Sum** | **1.00** | |

### 6.3 Sub-Score Normalisation — Details

Each metric maps `raw_value → subscore ∈ [0, 10]` via sigmoid:
```
subscore(x) = 10 / (1 + exp(-k · (x - x0)))
```
where `x0` = inflection point (score = 5), `k` = steepness. For "bad is higher" metrics (MDD, kurtosis), use `10 - subscore(x)`.

#### Category 1: Profitability (w=0.25)
| Metric | Weight within cat | Function params |
|---|---|---|
| Profit Factor PF | 0.35 | x0=1.5, k=2.5. PF≥3 → ≥9; PF≤1 → ≤2 |
| Win Rate WR | 0.20 | x0=0.50, k=6. WR≥0.6→≥9; WR≤0.35→≤2 |
| Payoff Ratio PR | 0.20 | x0=1.5, k=2.0. PR≥3→≥9 |
| Expectancy / $ risked | 0.25 | x0=0.20, k=4. ≥0.5→≥9; ≤0→0 |

#### Category 2: Risk-Adjusted Return (w=0.25)
| Metric | Weight within cat | Function params |
|---|---|---|
| Sharpe Ratio | 0.40 | x0=1.0, k=2.2. SR≥2.0→≥9; SR≤0→0 |
| Sortino Ratio | 0.30 | x0=1.5, k=1.8. SoR≥3→≥9 |
| Calmar Ratio | 0.30 | x0=1.0, k=2.5. ≥3→≥9; negative→0 |

#### Category 3: Drawdown & Capital Preservation (w=0.20)
| Metric | Weight within cat | Function params |
|---|---|---|
| Max Drawdown | 0.45 | reverse sigmoid: x0=-0.20, k=15. (MDD=−0.10 → ≥9; MDD=−0.40→≤2) |
| Ulcer Index | 0.25 | reverse sigmoid: x0=0.08, k=40. UI≤0.03→≥9 |
| Recovery Factor | 0.30 | x0=2.0, k=2.0. ≥5→≥9; ≤0.5→≤1 |

#### Category 4: Consistency & Robustness (w=0.15)
| Metric | Weight within cat | Function params |
|---|---|---|
| MC Robustness Rate (**Test C when correlation_flag; else Test A** per §5.2 Test-A caveat) | 0.35 | linear piecewise: 0%→0, 50%→4, 80%→7, 95%→10 |
| MC Sharpe Stability (same Test selection rule) | 0.25 | same piecewise as above |
| Runs-test Z (abs) | 0.20 | reverse sigmoid: x0=1.96, k=3. \|Z\|<1→≥9; \|Z\|>3→≤2 |
| % Profitable Months | 0.20 | x0=0.60, k=6. ≥0.80→≥9 |

#### Category 5: Behavioural Sanity (w=0.10)
| Metric | Weight within cat | Function params |
|---|---|---|
| Skew of returns | 0.35 | x0=0.5, k=2.5. skew≥1→≥9; skew<−0.5→≤2 |
| Excess Kurtosis | 0.25 | reverse sigmoid: x0=3, k=1.5. ≤0→9; ≥8→≤2 |
| Tail Ratio | 0.20 | x0=1.0, k=4. ≥1.5→9; ≤0.7→≤2 |
| Max Loss Streak | 0.20 | reverse piecewise: ≤3→10, 6→6, 10→2, >15→0 |

#### Category 6: Statistical Sufficiency (w=0.05)
| Metric | Weight within cat |
|---|---|
| Trade count N | 0.60 | N<10→0; 10→2; 30→5; 100→8; ≥500→10 |
| Outlier fraction (capped / total) | 0.40 | reverse: ≤1%→10; 5%→6; 15%→2; >25%→0 |

### 6.4 Hard Knockout Rules (INSTANT FAIL — evaluated BEFORE weighted sub-scores)
Brief §9 explicitly distinguishes **soft weighted parameters** from **hard knockout criteria**. If any of the following conditions is true, the scoring function short-circuits immediately with `final_score = 0`, `status = "KNOCKED_OUT"`, and a machine-readable `reason` string. No sub-scores, no penalty multiplication, no score inflation — the strategy is uninvestable at the stated thresholds.

```python
# scoring.py — knockouts evaluated first, no further scoring performed
def apply_knockouts(metrics: Metrics) -> KnockoutResult | None:
    N       = metrics.total_trades
    PF      = metrics.profit_factor
    MDD     = abs(metrics.max_drawdown)       # magnitude, positive
    reasons = []

    if N < 30:
        reasons.append(f"Insufficient sample size: N={N} < 30 trades required for any statistical confidence.")
    if PF < 1.0:
        reasons.append(f"Profit Factor {PF:.3f} < 1.0 — strategy is unprofitable before costs; no further scoring warranted.")
    if MDD > 0.35:
        reasons.append(f"Maximum Drawdown magnitude {MDD*100:.1f}% exceeds 35% — catastrophic capital loss inadmissible per risk mandate.")

    if reasons:
        return KnockoutResult(
            final_score=0,
            status="KNOCKED_OUT",
            reason=" | ".join(reasons),
            category_scores=None,
            penalties_applied=None,
        )
    return None
```

| Knockout | Threshold | Rationale |
|---|---|---|
| `N < 30` trades | ≥ 30 closed trades | CLT + small-sample Sharpe correction unreliable below this; strategy track record too short to grade. |
| `Profit Factor < 1.0` | ≥ 1.0 | Gross profit ≤ gross losses → zero or negative expectancy; no weighting scheme should mask a losing strategy. |
| `|Max Drawdown| > 35 %` | ≤ 35 % | Breaches the 1/3-capital-loss risk floor; recovery requires >54 % gain just to break even; immediate fail per mandate. |

If knocked out, the scorecard still contains all computed `metrics`, `monte_carlo` distributions, and `equity_curve` data for transparency — only the `scoring` block reports knockout. This allows a user to **still compare** a knocked-out strategy against live ones (compare-view labels it with a red KNOCKED_OUT pill).

### 6.5 Multiplicative Penalties (applied AFTER weighted sum — only if NOT knocked out)
Penalty factor `P = product(p_j)`, each `p_j ∈ [0.5, 1.0]`.

| Condition | Penalty `p_j` |
|---|---|
| `MDD > 40%` (magnitude) | `p = max(0.5, 1 - 2·(MDD − 0.40))` — extra drawdown pain beyond the 35 % knockout floor |
| `CVaR_95 > 10%` per trade | `p = max(0.6, 1 - 4·(CVaR − 0.10))` |
| `MC Robustness Rate < 50%` | `p = max(0.5, RR / 0.50)` — strategy not robust |
| `N < 50` (non-knockout, soft-discount only since N≥30 already) | `p = max(0.85, N/50)` — sample still borderline; gentle discount |

**Final Score (0–100):**
```
raw = Σ w_cat · Σ (w_metric_within · subscore_metric)      # ∈ [0, 10]
score = round(clamp(0, 100, 10 * raw * P))
```

### 6.6 Score Interpretation Labels
| Range | Label |
|---|---|
| 90–100 | Exceptional |
| 75–89 | Strong |
| 60–74 | Solid |
| 45–59 | Marginal |
| 30–44 | Weak |
| 0–29 | Poor |

---

## 7. Visualisation Plan

All charts are rendered in the frontend with **Chart.js** (MIT licensed, < 200KB). Backend endpoints also return image-exportable PNG via `matplotlib` for report generation.

### 7.1 Primary Plots
1. **Equity Curve (area chart)** — Actual equity + running peak line + MC 5/95 percentile band (Test A).
2. **Drawdown Profile (area, inverted)** — `DD_t` over time, with MDD line highlighted.
3. **Monthly Return Heatmap** — Months × years, colour-coded by return %.
4. **Trade PnL Distribution (histogram)** — Binned `R_i`, +vertical lines for VaR_95 and CVaR_95.
5. **Monte Carlo Distributions (4 subplots)** — TR, MDD, SR, PF from Test A (violin + percentiles).
6. **Win/Loss Scatter** — Per-trade return vs time, colour-coded by side (long/short).
7. **Rolling Sharpe** — 30-trade rolling window (or less if N<100), shows stability.
8. **Score Breakdown (horizontal bars)** — 6 category sub-scores, 0–10 scale; final big number on top.

### 7.2 Compare View
When 2+ evaluations are selected:
- Overlaid equity curves (coloured per strategy).
- Table: all metrics side-by-side, with Δ column vs the selected baseline.
- Radar chart: 6-category sub-scores for each strategy (normalised 0–10).

---

## 8. Save / Compare Strategy Evaluations

### 8.1 Storage Format
One JSON file per evaluation in `./evaluations/{id}.json`. `id` is a short slug: `{strategy_name}-{YYYYMMDD}-{hex4}`.

Schema (top-level keys):
```json
{
  "id": "strat-x-20260802-a1f3",
  "name": "Strategy X",
  "created_at": "ISO-8601",
  "seed": 12345,
  "input_file": {"name": "x.csv", "sha256": "…", "rows": 500},
  "cleaning_report": {"rejected": 2, "deduped": 5, "outliers": 3},
  "metrics": { /* every metric from §4 */ },
  "monte_carlo": {
    "test_a": {"iterations": 1000, "percentiles": {"TR":[…], "MDD":[…], …}, "rr": 0.91, "sr_stability": 0.78},
    "test_b": {"iterations": 1000, "mdd_2x_probability": 0.04},
    "test_c": {"iterations": 1000, "percentiles": {…}, "flagged_worse_than_a": true}
  },
  "correlation_flag": false,
  "scoring": {
    "status": "SCORED",
    "knockout_reason": null,
    "category_scores": {"profitability": 8.4, "risk_adj": 7.1, "drawdown": 6.2, "robustness": 7.4, "sanity": 8.0, "sufficiency": 5.0},
    "penalties": {"mdd_40pct": 1.00, "cvar_10pct": 1.00, "mc_rr": 1.00, "n_50": 1.00, "penalty_product": 1.00},
    "raw_score": 7.62,
    "final_score": 74,
    "label": "Solid"
  },
  "equity_curve": [{"t": ISO, "balance": float, "peak": float, "dd": float}, …],
  "monthly_returns": {"2025-01": 0.042, …},
  "version": "1.0"
}
```

### 8.2 Storage API
| Endpoint | Method | Description |
|---|---|---|
| `/evaluations` | GET | List all saved evaluations (id, name, created, score, N, label). |
| `/evaluations/{id}` | GET | Full JSON + chart data. |
| `/evaluations` | POST | Save current run → returns id. |
| `/evaluations/{id}` | DELETE | Delete. |
| `/evaluations/compare?ids=a,b,c` | GET | Merged metrics table + radar-chart data + baseline Δ. |

### 8.3 Backend (FastAPI) Key Endpoints
```
POST   /evaluate           multipart: file (.html / .htm / .csv / .xlsx),
                                    config (json, optional),
                                    seed (int, optional)
                                    strategy_name (string, optional)
                             → full scorecard + chart data + scoring.status
                                 ("SCORED" | "KNOCKED_OUT")

GET    /evaluations        → list (includes `status` + `label` + `final_score`)

POST   /evaluations        {body: scorecard} → save

GET    /evaluations/{id}   → load

GET    /evaluations/compare?ids=... → side-by-side metrics + Δ + KNOCKED_OUT pills

GET    /schema             → Pydantic JSON of every metric + scoring config (for the write-up)
```

---

## 9. Reproducibility End-to-End Checklist
- [x] `seed` parameter accepted on `/evaluate`; when absent, entropy-generated and stored.
- [x] `numpy.random.default_rng(seed)` — **single RNG instance** passed through MC functions.
- [x] Pandas operations use `kind="stable"` sorts.
- [x] Input file SHA-256 stored in scorecard.
- [x] `version` field in scorecard tracks engine schema.
- [x] `pytest -k reproducible` seeds 42, runs evaluate twice, asserts byte-equal JSON output.

---

## 10. Day-2 Sketch Deliverable Checklist
- [x] Architecture justification (§1)
- [x] Project tree (§2)
- [x] Data cleaning spec (§3)
- [x] Metric list with formulas (§4)
- [x] MC methodology (§5)
- [x] Scoring weights + normalisation (§6)
- [x] Chart list (§7)
- [x] Storage + compare schema (§8)

---

## 11. Implementation Milestones (2-week plan)
| Phase | Days | Deliverables |
|---|---|---|
| **P0 — Engine Core** | 1–3 | `ingestion.py` (cTrader HTML BeautifulSoup parser + CSV/XLSX fallback), `cleaning.py`, `metrics.py` passing all known-value tests. `tests/fixtures/ctrader_sample.html` (primary fixture, saved real cTrader export) + `sample_trades.csv` (secondary). |
| **P1 — Monte Carlo** | 4 | `monte_carlo.py` Tests A/B/C, correlation_flag output. Seed reproducibility test green (`pytest -k reproducible`). |
| **P2 — Scoring** | 5 | `scoring.py` §6.4 knockout rules + §6.5 penalties + §6.6 labels. Boundary + weight-sum + 3 knockout-condition unit tests. |
| **P3 — API + Storage** | 6 | FastAPI endpoints, JSON storage CRUD, `/evaluate` accepts `.html` primary + CSV/XLSX fallback. |
| **P4 — Frontend V1** | 7–8 | Upload form, score big-number, KNOCKED_OUT banner pill, equity+drawdown charts. |
| **P5 — All Charts + Compare** | 9–10 | Remaining 6 charts, compare view, radar chart, correlation-flag warning banner. |
| **P6 — Polish + Write-Up** | 11–12 | README with every formula. Edge-case handling. CSV export of rejected rows. Knockout reason rendering. MT4/MT5 `.htm` stretch ingestion if time. |
| **P7 — Buffer / Demo** | 13–14 | Dogfood with 3+ real cTrader report.html exports. Fix bugs. Demo prep. |

Checkpoint EOD Day 2: P0 substantially complete + this blueprint.

---

## 12. Consolidated Documented Limitations & Non-Negotiable Data-Honesty Caveats

This section exists so that every caveat raised by the assignment brief lives in one place (also repeated inline next to the relevant metric/section for visibility). The evaluation engine has **no hidden assumptions**.

### 12.1 Balance-Series / Closed-Trade Drawdown (Brief §7 + Data-Honesty clause)
*Also repeated inline at §4.3.*
Balance-based drawdown (`MDD`, `Average DD`, `Ulcer Index`, `Drawdown Duration`) is computed **solely from realised closed-trade P&L as reported in the cTrader History (Net P&L, or Gross + Commission + Swap)**. It does **not** include: (a) intra-trade floating / unrealised losses between deal open and close ticks; (b) worst-equity-trough during overlapping open positions; (c) margin-call or stop-hunt excursions that resolved at a better close price; (d) rollover equity gaps between session gaps. **Result:** MDD numbers are systematically **optimistic**, more so for strategies that hold losers open for long periods, run tight stop-losses rarely hit on *close*, or average-in to basket positions. The §6 scoring model compensates by using a conservative reverse-sigmoid on MDD and a hard knockout at `|MDD| > 35 %`. Users for whom open-equity risk matters must cross-check against a raw tick / equity-curve export — this tool intentionally does not fabricate tick data where none exists in the report.

### 12.2 Naive Bootstrap i.i.d. Assumption (Brief §8 — "the assumption you must not gloss over")
*Also repeated inline at §5.2 Test A.*
Test A (naive per-trade bootstrap with replacement) treats each trade return as i.i.d. This is **false** for:
- Basket strategies / portfolio books where N deals open simultaneously at the same timestamp and share a common market shock;
- Scaling in/out (multiple Deal IDs under a single Position ID), because resampling can break the position's entry/exit composition.

**Consequence for Test A output:** Robustness Rate (RR) and Sharpe Stability will be **artificially inflated**, and MDD's 5th-percentile worst case will be **thinner-tailed than reality**, because the bootstrap discards cross-correlation and is thus able to "sample its way out of" co-moving bad periods. **Prescribed remedy** (already wired into the score): (1) always read Test A with Test C (Block Bootstrap, stationary Politis–White, `L = √N` mean block length); (2) if Test C's MDD p5 is > 1.5× Test A's MDD p5 → `correlation_flag = true` in the scorecard; (3) §6 Category-4 scoring **uses Test C's RR & Sharpe Stability, not Test A's**, whenever the flag is on.

### 12.3 Synthetic Balance Interpolation (§3.3 step 9)
cTrader HTML reports do not include a per-row `Balance` column; balance is reconstructed sequentially by accumulating Net P&L onto a synthetic starting balance of `B_0 = 10 000` units or, if the cTrader footer contains a starting/ending "Balance" summary cell, that number is preferred and the synthetic series rescaled to match. This means:
- `R_i` return-per-trade uses either rescaled balance or `entry_notional = quantity × entry_price` as the denominator (§4 introductory definitions);
- Percentage-based metrics (`TR`, `CAGR`, `% DD`) are denominated in this synthetic base and therefore accurate **relative to each other, but not to a specific real account size unless the user overrides `B_0` in the request `config`**.

### 12.4 Fee / Swap / Commission Sign Convention (§3.2 cTrader mapping)
cTrader reports emit Commission and Swap as **negative numbers for charges**. PnL imputation therefore uses `pnl = gross_pnl + commission + swap` (algebraic sum, not `− commission`). Any row where the sign convention is reversed (unusual) will produce outliers and be caught by §3.3 step 8 (MAD cap), with the count reported in `cleaning_report`.

### 12.5 Hard Knockouts vs Soft Weighting (Brief §9)
To eliminate the risk that "good numbers elsewhere mask a fatal flaw," §6.4 introduces **three non-negotiable instant-fail knockout rules evaluated before any sub-scores or penalties**. Weighted soft parameters (§6.2 + §6.3) operate only for strategies that have passed the knockout gate. This distinction between admissible and inadmissible strategies is explicit in every scorecard via `scoring.status ∈ {"SCORED", "KNOCKED_OUT"}` and rendered in the UI as a coloured pill.

