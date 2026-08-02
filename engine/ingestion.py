"""Data ingestion.

Reads trade history from:
- Primary: cTrader report.html (BeautifulSoup4 parser on the History table)
- Fallback: CSV (UTF-8) or Excel (.xlsx first sheet) with aliased columns

Returns a list of raw dictionaries (one per trade), before cleaning.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from .models import CleaningReport

FileFormat = Literal["ctrader_html", "csv", "xlsx", "unknown"]  # noqa: F821


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sha256_file(path_or_bytes: str | os.PathLike | bytes) -> str:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return hashlib.sha256(bytes(path_or_bytes)).hexdigest()
    h = hashlib.sha256()
    with open(path_or_bytes, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_format_from_name(name: str) -> str:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext in {"html", "htm"}:
        return "ctrader_html"
    if ext == "csv":
        return "csv"
    if ext in {"xlsx", "xlsm"}:
        return "xlsx"
    return "unknown"


# --------------------------------------------------------------------------- #
# cTrader HTML parser
# --------------------------------------------------------------------------- #

# Map: canonical field -> list of accepted header strings (case-insensitive,
# whitespace/non-breaking-space normalised)
CTRADER_HEADER_ALIASES: Dict[str, List[str]] = {
    "position_id":  ["position", "position id", "positionid"],
    "deal_id":      ["deal", "deal id", "dealid", "order id", "orderid"],
    "symbol":       ["symbol", "instrument", "pair", "asset", "ticker"],
    "timestamp":    ["time", "close time", "closing time", "date", "datetime", "exit time"],
    "side":         ["type", "direction", "side", "action"],
    "quantity":     ["volume", "size", "lots", "qty", "amount", "units"],
    "entry_price":  ["open price", "open", "entry", "price in", "buy price"],
    "exit_price":   ["close price", "close", "exit", "price out", "sell price"],
    "gross_pnl":    ["gross p&l", "gross pl", "gross profit", "gross"],
    "commission":   ["commission", "comm"],
    "swap":         ["swap", "rollover"],
    "pnl":          ["net p&l", "net pl", "net profit", "profit", "p&l", "pl", "realized pnl"],
    "balance":      ["balance"],
}


def _norm(s: str) -> str:
    if s is None:
        return ""
    return (
        s.replace("\u00a0", " ")
         .strip()
         .lower()
         .replace("  ", " ")
    )


def _match_header(raw_header: str) -> str | None:
    key = _norm(raw_header)
    if not key:
        return None
    for canonical, aliases in CTRADER_HEADER_ALIASES.items():
        if key in {_norm(a) for a in aliases}:
            return canonical
    return None


def _parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    t = (
        str(s).replace("\u00a0", "")
              .replace(" ", "")
              .replace(",", "")
              .strip()
    )
    if t in {"", "-", "—"}:
        return None
    # Strip trailing currency sign if any
    while t and not (t[0].isdigit() or t[0] in "+-." ):
        t = t[1:]
    while t and not t[-1].isdigit() and t[-1] != ".":
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def parse_ctrader_html(raw_html: bytes | str) -> List[Dict]:
    """Parse a cTrader report.html body into a list of trade rows (dicts).

    Returns one dict per deal row in the History tbody; fields are matched via
    header aliases. Numbers are left as strings / floats; type coercion and
    validation happen in cleaning.py.
    """
    if isinstance(raw_html, bytes):
        raw_html = raw_html.decode("utf-8", errors="replace")

    soup = BeautifulSoup(raw_html, "lxml")

    # Prefer a <table class="history"> (as in our fixture), else the largest
    # table in the document by row count.
    tables = soup.find_all("table")
    if not tables:
        return []

    def _score(tbl) -> int:
        # class="history" => highest score
        cls = " ".join(tbl.get("class", []))
        if "history" in cls.lower():
            return 10_000
        return len(tbl.find_all("tr"))

    table = max(tables, key=_score)

    header_row = table.find("tr")
    if header_row is None:
        return []

    header_cells = header_row.find_all(["th", "td"])
    col_map: Dict[int, str] = {}
    for idx, cell in enumerate(header_cells):
        canonical = _match_header(cell.get_text())
        if canonical:
            col_map[idx] = canonical

    rows: List[Dict] = []
    tbody = table.find("tbody")
    has_separate_thead = tbody is not None and header_row.find_parent("thead") is not None
    tbody = tbody or table
    body_rows = tbody.find_all("tr")
    if not has_separate_thead:
        body_rows = body_rows[1:]
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        record: Dict = {}
        for idx, canonical in col_map.items():
            if idx >= len(cells):
                continue
            text = cells[idx].get_text(" ", strip=True)
            if canonical in {"position_id", "deal_id", "symbol", "timestamp", "side"}:
                record[canonical] = text
            else:
                record[canonical] = _parse_number(text)
        rows.append(record)

    return rows


# --------------------------------------------------------------------------- #
# CSV / XLSX fallback
# --------------------------------------------------------------------------- #

# Alias map for non-HTML ingestion (blueprint §3.2 fallback)
CSV_ALIASES: Dict[str, List[str]] = {
    "timestamp":   ["timestamp", "time", "date", "datetime", "open_time", "close_time", "exit_time"],
    "symbol":      ["symbol", "pair", "instrument", "asset", "ticker"],
    "side":        ["side", "direction", "action", "type"],
    "quantity":    ["quantity", "qty", "size", "amount", "units", "volume", "lots"],
    "entry_price": ["entry_price", "open", "price_in", "buy_price", "entry"],
    "exit_price":  ["exit_price", "close", "price_out", "sell_price", "exit"],
    "pnl":         ["pnl", "profit", "profit_loss", "pl", "realized_pnl", "net_pnl"],
    "gross_pnl":   ["gross_pnl", "gross_profit", "gross_pl", "gross"],
    "commission":  ["commission", "fees", "cost"],
    "swap":        ["swap", "rollover"],
    "balance":     ["balance", "equity", "account_value"],
    "position_id": ["position_id", "positionid", "position"],
    "deal_id":     ["deal_id", "dealid", "deal", "order_id", "orderid"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    rename: Dict[str, str] = {}
    for col in df.columns:
        c = str(col).strip().lower()
        for canonical, aliases in CSV_ALIASES.items():
            if c in [a.lower() for a in aliases]:
                rename[col] = canonical
                break
    df = df.rename(columns=rename)
    return df


def parse_csv(raw_bytes: bytes, name: str = "input.csv") -> List[Dict]:
    # Try UTF-8 with BOM tolerance, else latin-1 fallback
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text))
    df = _normalise_columns(df)
    records: List[Dict] = []
    for rec in df.to_dict(orient="records"):
        clean: Dict = {}
        for k, v in rec.items():
            if isinstance(v, float) and pd.isna(v):
                continue
            clean[k] = v
        records.append(clean)
    return records


def parse_xlsx(raw_bytes: bytes) -> List[Dict]:
    df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
    df = _normalise_columns(df)
    records: List[Dict] = []
    for rec in df.to_dict(orient="records"):
        clean: Dict = {}
        for k, v in rec.items():
            if isinstance(v, float) and pd.isna(v):
                continue
            clean[k] = v
        records.append(clean)
    return records


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def ingest_file(
    raw_bytes: bytes,
    filename: str,
) -> Tuple[List[Dict], str, str, int]:
    """Ingest one report file.

    Returns ``(rows, format_name, sha256_hex, row_count)``.
    """
    fmt = _detect_format_from_name(filename)
    sha = sha256_file(raw_bytes)

    if fmt == "ctrader_html":
        rows = parse_ctrader_html(raw_bytes)
    elif fmt == "csv":
        rows = parse_csv(raw_bytes, filename)
    elif fmt == "xlsx":
        rows = parse_xlsx(raw_bytes)
    else:
        # Heuristic: try HTML if it starts with <, else CSV
        head = raw_bytes.lstrip()[:6].lower()
        if head.startswith(b"<"):
            fmt = "ctrader_html"
            rows = parse_ctrader_html(raw_bytes)
        else:
            fmt = "csv"
            rows = parse_csv(raw_bytes, filename)

    return rows, fmt, sha, len(rows)
