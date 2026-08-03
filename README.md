
# OrionQuant – Strategy Evaluation Tool (OQT-INT-EVAL-01)

Strategy-agnostic engine: ingest a cTrader `report.html` (or CSV/XLSX fallback) and get a
0–100 **Scorecard** over 30+ performance / risk metrics plus 3 Monte Carlo robustness tests.

Full design → **[BLUEPRINT.md](BLUEPRINT.md)**.

---

## Quick start

```powershell
# 1. Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run
python main.py
# → http://localhost:8000    (UI)
# → http://localhost:8000/docs  (Swagger)

# 3. Test
pytest tests/ -v          # 52/52 passing
```

## Primary workflow

1. **cTrader → History → Export → HTML** (`report.html`) — this is the *primary* ingestion path.
2. Upload at `http://localhost:8000/` → get Score / label (Exceptional / Strong / Good /
   Acceptable / Poor / Inadequate / KNOCKED_OUT) + 8 charts + full metrics table.
3. Save runs → tick 2+ in the **Saved / Compare** tab → overlaid equity curves, radar, Δ% table.

## Hard knockouts (→ 0 / red pill)

- `N < 30` trades
- `Profit Factor < 1.0`
- `|Max Drawdown| > 35%`

## Scoring weights (6 cats, clamp 0–100)

Profitability 25% · Risk-Adj 25% · Drawdown 20% · Robustness (MC) 15% · Behavioural 10% · Sufficiency 5%

See BLUEPRINT §4–§6 for formulas, sigmoid/piecewise sub-scores, and the documented
closed-trade MDD and Monte-Carlo i.i.d. caveats.

## Project layout

```
NST/
├── BLUEPRINT.md               Technical write-up + formulas + assumptions
├── main.py                    FastAPI entry (13 routes)
├── requirements.txt
├── conftest.py                pytest _deps sys.path bootstrap
├── engine/
│   ├── ingestion.py           cTrader HTML (BeautifulSoup) + CSV/XLSX alias fallback
│   ├── cleaning.py            schema → dedup → MAD outlier → balance interpolation
│   ├── metrics.py             Basic / Growth / Risk-Adj / Drawdown / Behavioural (30+)
│   ├── monte_carlo.py         Tests A i.i.d. · B permute · C block-bootstrap + corr flag
│   ├── scoring.py             Knockouts → weighted categories → multiplicative penalties
│   ├── storage.py             save/load/list/compare evaluations (per-JSON files)
│   ├── models.py              All Pydantic schemas (GET /schema)
│   └── seed.py                Reproducible RNG (RandomSequence)
├── static/                    index.html · app.js · styles.css  (Chart.js UI)
├── tests/
│   └── fixtures/              ctrader_sample.html · sample_trades.csv
└── evaluations/               Saved run JSONs (gitignored except .gitkeep)
```
#
