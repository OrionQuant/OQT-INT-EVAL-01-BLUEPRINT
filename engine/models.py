"""Pydantic models for the evaluation engine."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DISCLAIMER_TEXT = (
    "Notice: True Walk-Forward Validation and Purged Cross-Validation require strategy engine "
    "integration and underlying bar/tick data. The overfitting risk reported here is a bounded "
    "estimate derived exclusively from static trade-history analysis."
)


# --------------------------------------------------------------------------- #
# Ingestion / Trade level
# --------------------------------------------------------------------------- #

class Trade(BaseModel):
    """A single closed trade (or deal row from a cTrader report)."""

    position_id: Optional[str] = None
    deal_id: Optional[str] = None
    timestamp: datetime
    symbol: str
    side: Literal["long", "short"]
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    gross_pnl: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    pnl: float
    capped_pnl: Optional[float] = None
    balance_after: Optional[float] = None
    entry_notional: float
    is_outlier: bool = False
    duration_seconds: float = 0.0


class CleaningReport(BaseModel):
    rows_received: int = 0
    rows_rejected_schema: int = 0
    rows_deduped: int = 0
    rows_flagged_zero_duration: int = 0
    outliers_tagged: int = 0
    adjustments_ignored: int = 0
    rejected_file: Optional[str] = None


# --------------------------------------------------------------------------- #
# Performance & Risk Metrics
# --------------------------------------------------------------------------- #

class BasicMetrics(BaseModel):
    total_trades: int
    win_rate: float
    loss_rate: float
    win_count: int
    loss_count: int
    tie_count: int
    average_win: float
    average_loss: float
    profit_factor: float
    payoff_ratio: float
    expectancy_per_trade: float
    expectancy_per_unit_risk: float
    long_count: int
    short_count: int
    long_win_rate: float
    short_win_rate: float


class GrowthMetrics(BaseModel):
    net_pnl: float
    total_return: float
    cagr: Optional[float] = None
    cagr_flagged_insufficient: bool = False
    mean_return_per_trade: float
    median_return_per_trade: float
    skew_returns: float
    excess_kurtosis_returns: float


class DrawdownMetrics(BaseModel):
    max_drawdown: float  # signed, negative
    average_drawdown: float
    max_drawdown_duration_days: float
    ulcer_index: float
    recovery_factor: float


class RiskAdjustedMetrics(BaseModel):
    sharpe_ratio: float
    sharpe_small_sample_corrected: bool = False
    sortino_ratio: float
    calmar_ratio: Optional[float] = None
    omega_ratio: float
    romad: float
    tail_ratio: float
    var_95: float
    cvar_95: float
    annualised: bool
    avg_trades_per_year: float
    # Module 5: Probabilistic Sharpe Ratio — N_obs = T. Higher is better.
    probabilistic_sharpe_ratio: Optional[float] = None
    # Module 5: Deflated Sharpe Ratio — N_trials is user-provided. Lower than PSR.
    deflated_sharpe_ratio: Optional[float] = None
    psr_reference_sharpe: float = 0.0
    dsr_n_trials: Optional[int] = None


# --------------------------------------------------------------------------- #
# Module 3: CUSUM + Regime shift checks on realized return series
# (Report-analysis layer — operates on closed trades only; no bar data).
# --------------------------------------------------------------------------- #

class CUSUMStat(BaseModel):
    """Two-sided CUSUM run on per-trade returns R_i."""

    cumsum_pos_peak: float = 0.0    # max value of CUSUM+
    cumsum_neg_peak: float = 0.0    # min value of CUSUM− (negative magnitude)
    threshold_hit: bool = False     # True if either side crossed threshold
    n_regime_shifts_detected: int = 0  # cumulative crossings of ±threshold
    threshold_used: float = 0.0


class RegimeCheck(BaseModel):
    """Module 3 output: CUSUM + subsample stability checks."""

    cusum_first_half: Optional[CUSUMStat] = None
    cusum_second_half: Optional[CUSUMStat] = None
    cusum_full: Optional[CUSUMStat] = None
    # R_first_half_mean - R_second_half_mean
    mean_return_gap: Optional[float] = None
    # Sharpe-first-half minus Sharpe-second-half (per-trade raw, not annual)
    sharpe_half_gap: Optional[float] = None
    # Boolean: flagged as unstable if gap > threshold or CUSUM hit
    regime_unstable_flag: bool = False


class BehaviouralMetrics(BaseModel):
    max_win_streak: int
    max_loss_streak: int
    profit_streak_ratio: float
    runs_z_score: float
    best_trade: float
    worst_trade: float
    profitable_month_fraction: Optional[float] = None
    monthly_return_volatility: Optional[float] = None
    # Module 3 attaches here:
    regime_check: Optional[RegimeCheck] = None


class Metrics(BaseModel):
    basic: BasicMetrics
    growth: GrowthMetrics
    drawdown: DrawdownMetrics
    risk_adj: RiskAdjustedMetrics
    behav: BehaviouralMetrics


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #

class MCPercentiles(BaseModel):
    total_return: List[float]   # p0/p5/p25/p50/p75/p95/p100
    max_drawdown: List[float]
    sharpe_ratio: List[float]
    profit_factor: List[float]


class MCTestResult(BaseModel):
    iterations: int
    percentiles: MCPercentiles
    robustness_rate: Optional[float] = None   # P(TR > 0)
    sharpe_stability: Optional[float] = None  # P(SR > 1.0)
    mdd_2x_probability: Optional[float] = None
    flagged_worse_than_a: Optional[bool] = None


class MonteCarloResult(BaseModel):
    test_a: MCTestResult
    test_b: MCTestResult
    test_c: MCTestResult
    correlation_flag: bool = False
    effective_mc_source: Literal["test_a", "test_c"] = "test_a"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

ScoringStatus = Literal["SCORED", "KNOCKED_OUT"]


class CategoryScores(BaseModel):
    profitability: float
    risk_adj: float
    drawdown: float
    robustness: float
    sanity: float
    sufficiency: float


class PenaltiesApplied(BaseModel):
    mdd_40pct: float = 1.0
    cvar_10pct: float = 1.0
    mc_rr: float = 1.0
    n_50: float = 1.0
    penalty_product: float = 1.0


class ScoringResult(BaseModel):
    status: ScoringStatus = "SCORED"
    knockout_reason: Optional[str] = None
    category_scores: Optional[CategoryScores] = None
    penalties: PenaltiesApplied = Field(default_factory=PenaltiesApplied)
    raw_score: Optional[float] = None        # 0..10 pre-penalty weighted
    final_score: int                         # 0..100
    label: str                               # Exceptional / ... / Poor / KNOCKED_OUT


# --------------------------------------------------------------------------- #
# Equity curve + monthly returns (for charts + storage)
# --------------------------------------------------------------------------- #

class EquityPoint(BaseModel):
    t: datetime
    balance: float
    peak: float
    dd: float   # drawdown, <=0


# --------------------------------------------------------------------------- #
# Top-level scorecard (ingestion -> cleaning -> metrics -> MC -> scoring)
# --------------------------------------------------------------------------- #

class InputFileInfo(BaseModel):
    name: str
    sha256: str
    rows: int
    format: Literal["ctrader_html", "csv", "xlsx", "unknown"]


class Scorecard(BaseModel):
    id: Optional[str] = None
    name: str = "unnamed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    seed: Optional[str] = None
    input_file: InputFileInfo
    cleaning_report: CleaningReport
    metrics: Metrics
    monte_carlo: MonteCarloResult
    correlation_flag: bool = False
    scoring: ScoringResult
    equity_curve: List[EquityPoint]
    monthly_returns: Dict[str, float] = Field(default_factory=dict)
    # Report-analysis layer — explicit disclaimer for every dashboard/report.
    disclaimer: str = DISCLAIMER_TEXT
    version: str = "1.1"
