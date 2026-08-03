"""FastAPI entry point — blueprint §8.3.

Endpoints
---------
POST  /evaluate            Multipart: file + config + seed + strategy_name
GET   /evaluations         List saved scorecards (id/name/score/N/label).
POST  /evaluations         Save a scorecard body → returns id.
GET   /evaluations/{id}    Load full scorecard JSON.
DEL   /evaluations/{id}    Delete.
GET   /evaluations/compare?ids=a,b,c   Compare 2+ scorecards.
GET   /schema              Pydantic schemas (for the write-up).
GET   /                    Static index.html (single-page UI).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_BASE_DIR_SELF = Path(__file__).resolve().parent
_DEPS_DIR = _BASE_DIR_SELF / "_deps"
if _DEPS_DIR.exists() and str(_DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPS_DIR))
if str(_BASE_DIR_SELF) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR_SELF))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import storage
from engine.cleaning import clean_raw_rows
from engine.ingestion import ingest_file
from engine.metrics import compute_all_metrics
from engine.monte_carlo import run_monte_carlo
from engine.models import InputFileInfo, Scorecard
from engine.scoring import score_strategy
from engine.seed import make_rng

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVAL_DIR = BASE_DIR / "evaluations"

STATIC_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="OrionQuant Strategy Evaluation Tool", version="1.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Pydantic request body wrappers
# --------------------------------------------------------------------------- #


class SaveScorecardRequest(BaseModel):
    scorecard: Dict


class EvaluateConfig(BaseModel):
    start_balance: float = 10_000.0
    rf_annual: float = 0.04
    mc_iterations: int = 1000
    # DSR deflation parameter — total backtests / parameter variants attempted.
    # Never inferred from trade count (N_obs). Optional; when omitted DSR is null.
    n_trials: Optional[int] = None


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #


def _run_pipeline(
    raw_bytes: bytes,
    filename: str,
    *,
    config: EvaluateConfig,
    seed: Optional[str],
    strategy_name: str,
) -> Scorecard:
    # 1. Ingest
    rows, fmt, sha, row_count = ingest_file(raw_bytes, filename)
    input_info = InputFileInfo(name=filename, sha256=sha, rows=row_count, format=fmt)

    # 2. Clean
    trades, cleaning_report = clean_raw_rows(
        rows, evaluations_dir=str(EVAL_DIR),
        synthetic_start_balance=config.start_balance,
    )
    if not trades:
        raise HTTPException(status_code=400, detail="No valid trade rows after cleaning.")

    # 3. Metrics + equity curve + monthly returns
    #    N_obs (PSR) = len(trades); N_trials (DSR) = config.n_trials (user).
    metrics, equity, monthly = compute_all_metrics(
        trades,
        start_balance=config.start_balance,
        rf_annual=config.rf_annual,
        n_trials=config.n_trials,
    )

    # 4. Monte Carlo (seeded RNG — seed always stored as string)
    seed_int: Optional[int] = None
    if seed is not None and str(seed).strip() != "":
        # Accept decimal string or legacy integer form from the form field
        try:
            seed_int = int(str(seed).strip(), 10)
        except ValueError:
            raise HTTPException(status_code=400, detail="seed must be a decimal integer string")
    rng, actual_seed = make_rng(seed_int)
    mc = run_monte_carlo(
        trades, rng,
        start_balance=config.start_balance,
        iterations=config.mc_iterations,
        actual_max_drawdown=metrics.drawdown.max_drawdown,
        avg_trades_per_year=metrics.risk_adj.avg_trades_per_year,
        rf_annual=config.rf_annual,
    )

    # 5. Scoring
    scoring = score_strategy(
        metrics, mc,
        trades_count=cleaning_report.rows_received,
        outliers_tagged=cleaning_report.outliers_tagged,
    )

    card = Scorecard(
        name=strategy_name or filename,
        seed=actual_seed,  # always a string (Fix #10)
        input_file=input_info,
        cleaning_report=cleaning_report,
        metrics=metrics,
        monte_carlo=mc,
        correlation_flag=mc.correlation_flag,
        scoring=scoring,
        equity_curve=equity,
        monthly_returns=monthly,
    )
    return card


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/")
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return {"ok": True, "message": "Static index.html missing — add it under ./static/"}


@app.post("/evaluate")
async def evaluate(
    file: UploadFile = File(...),
    config: str = Form("{}"),
    seed: Optional[str] = Form(None),
    strategy_name: str = Form(""),
):
    raw = await file.read()
    try:
        cfg_dict = json.loads(config) if config else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config must be valid JSON")
    cfg = EvaluateConfig(**cfg_dict)
    card = _run_pipeline(
        raw, file.filename or "upload",
        config=cfg, seed=seed, strategy_name=strategy_name,
    )
    return JSONResponse(card.model_dump(mode="json"))


@app.get("/evaluations")
def list_evaluations():
    return JSONResponse(storage.list_scorecards(str(EVAL_DIR)))


@app.post("/evaluations")
def save_evaluation(body: SaveScorecardRequest):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(Scorecard)
    try:
        card = adapter.validate_python(body.scorecard)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid scorecard: {e}")
    sid = storage.save_scorecard(card, evaluations_dir=str(EVAL_DIR))
    return {"id": sid}


@app.get("/evaluations/{eval_id}")
def get_evaluation(eval_id: str):
    try:
        card = storage.load_scorecard(str(EVAL_DIR), eval_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(card.model_dump(mode="json"))


@app.delete("/evaluations/{eval_id}")
def delete_evaluation(eval_id: str):
    ok = storage.delete_scorecard(str(EVAL_DIR), eval_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": True}


@app.get("/evaluations/compare")
def compare_evaluations(ids: str):
    id_list = [i for i in ids.split(",") if i]
    if len(id_list) < 1:
        raise HTTPException(status_code=400, detail="Provide ?ids=a,b,c")
    try:
        result = storage.compare_scorecards(str(EVAL_DIR), id_list)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return JSONResponse(result)


@app.get("/schema")
def get_schema():
    from engine.models import (
        Metrics, Scorecard, ScoringResult, MonteCarloResult,
    )
    return {
        "Metrics": Metrics.model_json_schema(),
        "Scorecard": Scorecard.model_json_schema(),
        "ScoringResult": ScoringResult.model_json_schema(),
        "MonteCarloResult": MonteCarloResult.model_json_schema(),
        "ScoringConfig": EvaluateConfig.model_json_schema(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
