"""Persisted evaluation storage — blueprint §8.

One JSON file per evaluation in ``./evaluations/{id}.json``. The ``id`` is a
short slug: ``{strategy_name}-{YYYYMMDD}-{hex4}``.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import TypeAdapter

from .models import Scorecard

_SCORECARD_ADAPTER = TypeAdapter(Scorecard)


def _slugify(name: str) -> str:
    """Make a filename-safe slug from a strategy name."""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "strategy"


def _make_id(name: str, created_at: datetime | None = None) -> str:
    created_at = created_at or datetime.utcnow()
    date = created_at.strftime("%Y%m%d")
    hex4 = secrets.token_hex(2)
    return f"{_slugify(name)}-{date}-{hex4}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _path_for(evaluations_dir: str, eval_id: str) -> str:
    return os.path.join(evaluations_dir, f"{eval_id}.json")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def save_scorecard(
    scorecard: Scorecard,
    *,
    evaluations_dir: str,
    force_id: Optional[str] = None,
) -> str:
    """Persist a scorecard. Returns the id (slug)."""
    _ensure_dir(evaluations_dir)
    if force_id:
        scorecard.id = force_id
    elif scorecard.id is None:
        scorecard.id = _make_id(scorecard.name, scorecard.created_at)
    path = _path_for(evaluations_dir, scorecard.id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(scorecard.model_dump_json(indent=2))
    return scorecard.id


def load_scorecard(evaluations_dir: str, eval_id: str) -> Scorecard:
    path = _path_for(evaluations_dir, eval_id)
    if not os.path.exists(path):
        raise KeyError(f"Evaluation not found: {eval_id}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _SCORECARD_ADAPTER.validate_python(data)


def delete_scorecard(evaluations_dir: str, eval_id: str) -> bool:
    path = _path_for(evaluations_dir, eval_id)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def list_scorecards(evaluations_dir: str) -> List[Dict]:
    """Return a lightweight index row per evaluation:
    {id, name, created_at, final_score, N, label, status}.
    """
    _ensure_dir(evaluations_dir)
    results: List[Dict] = []
    for p in Path(evaluations_dir).glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "id": data.get("id") or p.stem,
                "name": data.get("name", "unnamed"),
                "created_at": data.get("created_at"),
                "final_score": data.get("scoring", {}).get("final_score", 0),
                "total_trades": data.get("metrics", {}).get("basic", {}).get("total_trades", 0),
                "label": data.get("scoring", {}).get("label", ""),
                "status": data.get("scoring", {}).get("status", ""),
            })
        except Exception:
            continue
    results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return results


def compare_scorecards(evaluations_dir: str, ids: List[str]) -> Dict:
    """Return merged metrics table + per-metric Δ vs the first (baseline) id.

    Schema::

        {"baseline_id": str, "rows": [{metric: str, baseline: v, by_id: {id: v}, delta: {id: v_pct}}]}
    """
    if not ids:
        return {"baseline_id": None, "rows": [], "scorecards": {}}

    cards: Dict[str, Scorecard] = {}
    for i in ids:
        cards[i] = load_scorecard(evaluations_dir, i)
    baseline_id = ids[0]

    # Flatten metric names → values.
    def _flatten(card: Scorecard) -> Dict[str, float | str | int]:
        out: Dict[str, float | str | int] = {}
        for grp_name in ("basic", "growth", "drawdown", "risk_adj", "behav"):
            grp = getattr(card.metrics, grp_name)
            for k, v in grp.model_dump().items():
                if isinstance(v, (int, float, str, bool)):
                    out[f"{grp_name}.{k}"] = v
        if card.scoring.category_scores:
            for k, v in card.scoring.category_scores.model_dump().items():
                out[f"cat.{k}"] = v
        out["final_score"] = card.scoring.final_score
        out["raw_score"] = card.scoring.raw_score or 0.0
        return out

    flat_baseline = _flatten(cards[baseline_id])
    metric_names = sorted(flat_baseline.keys())
    rows = []
    for m in metric_names:
        bv = flat_baseline[m]
        by_id = {}
        delta = {}
        for i, card in cards.items():
            f = _flatten(card)
            v = f.get(m)
            by_id[i] = v
            if i != baseline_id and isinstance(bv, (int, float)) and isinstance(v, (int, float)):
                if abs(bv) > 1e-9:
                    delta[i] = (v - bv) / abs(bv) * 100.0
                else:
                    delta[i] = None
        rows.append({
            "metric": m,
            "baseline": bv,
            "by_id": by_id,
            "delta_pct": delta,
        })
    return {
        "baseline_id": baseline_id,
        "rows": rows,
        "scorecards": {i: c.model_dump() for i, c in cards.items()},
    }
