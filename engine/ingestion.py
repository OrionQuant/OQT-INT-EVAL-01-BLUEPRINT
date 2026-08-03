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
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as _dateparser

from .models import CleaningReport

FileFormat = Literal["ctrader_html", "csv", "xlsx", "unknown"]  # noqa: F821

DISCLAIMER_TEXT = (
    "Notice: True Walk-Forward Validation and Purged Cross-Validation require strategy engine "
    "integration and underlying bar/tick data. The overfitting risk reported here is a bounded "
    "estimate derived exclusively from static trade-history analysis."
)


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
    """Parse a number with either US (1,000.00) or European (1.000,00) formatting.

    Rules:
      * If both '.' and ',' are present → the rightmost one is the decimal
        separator; the other is stripped as thousand separator.
      * If exactly one of '.' or ',' appears and it appears more than once →
        treat it as a thousands separator (e.g. "1.000.000", "1,000,000").
      * Otherwise: the single separator, if present, is assumed to be decimal
        unless the context has 3 digits after it (heuristic to prefer thousands
        for ambiguous "2.000" which looks like 2 thousand in EU locales).
    """
    if s is None:
        return None
    t = (
        str(s).replace("\u00a0", "")
              .replace(" ", "")
              .strip()
    )
    if t in {"", "-", "—", "−"}:
        return None
    sign = 1.0
    while t and t[0] in "+-−":
        if t[0] in ("-", "−"):
            sign *= -1.0
        t = t[1:]
    # Strip trailing currency / non-numeric characters
    while t and not (t[-1].isdigit() or t[-1] in ".,"):
        t = t[:-1]
    # Strip leading currency-ish characters
    while t and not (t[0].isdigit() or t[0] in ".,"):
        t = t[1:]
    if not t:
        return None

    has_dot = "." in t
    has_com = "," in t

    if has_dot and has_com:
        # The rightmost one is decimal
        if t.rfind(",") > t.rfind("."):
            # EU format: 1.000,25 → strip '.' → replace ',' with '.'
            num_str = t.replace(".", "").replace(",", ".")
        else:
            # US format: 1,000.25 → strip ','
            num_str = t.replace(",", "")
    elif has_com:
        if t.count(",") > 1:
            # All commas are thousands separators
            num_str = t.replace(",", "")
        else:
            # Single comma: check if there are exactly 3 digits after it
            idx = t.rfind(",")
            digits_after = len(t) - idx - 1
            if digits_after == 3 and (len(t) - idx - 1 == 3 and idx > 0):
                # Ambiguous: prefer thousands separator (e.g. "2.000" → 2000)
                num_str = t.replace(",", "")
            else:
                # decimal comma
                num_str = t.replace(",", ".")
    elif has_dot:
        if t.count(".") > 1:
            num_str = t.replace(".", "")
        else:
            idx = t.rfind(".")
            digits_after = len(t) - idx - 1
            if digits_after == 3 and idx > 0:
                # Heuristic: ambiguous single dot with 3 trailing digits → thousands
                num_str = t.replace(".", "")
            else:
                num_str = t
    else:
        num_str = t

    try:
        return sign * float(num_str)
    except ValueError:
        return None


def _parse_timestamp_eu(raw: str | None) -> object:
    """Parse a timestamp string with European-format preference (dayfirst=True).

    Unambiguous ISO-like strings (``YYYY-MM-DD…``) are parsed *without*
    dayfirst so ``2025-01-03`` stays January 3, not March 1. European
    dotted / slash forms (``DD.MM.YYYY``, ``DD/MM/YYYY``) use dayfirst=True.
    Returns a timezone-aware datetime (UTC-normalised) or None on failure.
    """
    import re
    import pandas as pd
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # ISO / machine format: leading 4-digit year → never apply dayfirst
    iso_like = bool(re.match(r"^\d{4}[-/]", s))
    dayfirst = not iso_like

    try:
        dt = _dateparser.parse(s, dayfirst=dayfirst)
    except (ValueError, TypeError, OverflowError):
        try:
            dt = _dateparser.parse(s, dayfirst=not dayfirst)
        except (ValueError, TypeError, OverflowError):
            return None
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def parse_ctrader_html(raw_html: bytes | str) -> List[Dict]:
    """Parse a cTrader report.html body into a list of trade rows (dicts).

    Returns one dict per deal row in the History table; fields are matched via
    header aliases. Numbers are left as strings / floats; type coercion and
    validation happen in cleaning.py.

    Header search order (robust to real cTrader exports):
      1. <thead> <tr> if present
      2. Else, first <tr> of <tbody> (if one exists)
      3. Else, first <tr> directly under <table>
    All remaining <tr> cells are treated as data rows regardless of wrapper.
    """
    if isinstance(raw_html, bytes):
        raw_html = raw_html.decode("utf-8", errors="replace")

    soup = BeautifulSoup(raw_html, "lxml")

    # Prefer a <table class="history">, else the largest table by row count
    tables = soup.find_all("table")
    if not tables:
        return []

    def _score(tbl) -> int:
        cls = " ".join(tbl.get("class", []))
        if "history" in cls.lower():
            return 10_000
        return len(tbl.find_all("tr"))

    table = max(tables, key=_score)

    # --- Find the header row with flexible location ---
    thead = table.find("thead")
    tbody = table.find("tbody")

    header_row = None
    if thead is not None:
        header_row = thead.find("tr")
    if header_row is None and tbody is not None:
        candidate = tbody.find("tr")
        # Only accept <tbody>'s first row as header if it looks like one
        # (most cells are non-numeric or use <th> tags)
        if candidate is not None:
            cells = candidate.find_all(["th", "td"])
            n_th = sum(1 for c in cells if c.name == "th")
            joined = " ".join(c.get_text(" ", strip=True) for c in cells).lower()
            if n_th >= 1 or any(kw in joined for kw in ("symbol", "time", "price", "profit", "balance", "volume", "position", "deal")):
                header_row = candidate
    if header_row is None:
        # Fallback: first top-level <tr> directly under <table>
        for tr in table.find_all("tr", recursive=False):
            header_row = tr
            break
        if header_row is None:
            header_row = table.find("tr")
    if header_row is None:
        return []

    header_cells = header_row.find_all(["th", "td"])
    col_map: Dict[int, str] = {}
    for idx, cell in enumerate(header_cells):
        canonical = _match_header(cell.get_text())
        if canonical:
            col_map[idx] = canonical

    # --- Collect body rows: all <tr> under table except the header itself ---
    all_trs = table.find_all("tr")
    body_rows = [tr for tr in all_trs if tr is not header_row]
    # Also drop rows that appear inside <thead> (in case there were extra)
    body_rows = [tr for tr in body_rows if tr.find_parent("thead") is None]
    # If tbody exists, prefer tbody trs only (for data) to avoid summary footer rows
    if tbody is not None:
        tbody_trs = [tr for tr in tbody.find_all("tr") if tr is not header_row]
        if tbody_trs:
            body_rows = tbody_trs

    rows: List[Dict] = []
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        # Skip rows that look like a repeated header (e.g. symbol header strings in data cells)
        joined = " ".join(c.get_text(" ", strip=True) for c in cells).lower()
        if any(kw in joined for kw in ("symbol", "position", "deal id", "deal")) and \
           any(kw in joined for kw in ("time", "close", "profit", "price")):
            if len(rows) == 0:  # only skip if we've not seen any real data yet
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
    text: str | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return []

    # --- Fix #2: Sniff delimiter from the first ~32KB ---
    dialect: csv.Dialect | None = None
    try:
        sample = "\n".join(text.splitlines()[:100])
        if len(sample) > 32768:
            sample = sample[:32768]
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
    except (csv.Error, StopIteration, ValueError):
        dialect = None
    if dialect is None:
        # Fallback: count delimiter occurrences across first 20 lines
        header_line = text.splitlines()[0] if text.splitlines() else ""
        counts = {d: header_line.count(d) for d in (",", ";", "\t")}
        best = max(counts, key=lambda d: (counts[d], d != ","))
        if counts[best] == 0:
            sep = ","
        else:
            sep = best
    else:
        sep = dialect.delimiter

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    except (pd.errors.ParserError, csv.Error, ValueError, UnicodeDecodeError):
        try:
            df = pd.read_csv(io.StringIO(text), sep=";", engine="python")
        except (pd.errors.ParserError, csv.Error, ValueError, UnicodeDecodeError):
            df = pd.read_csv(io.StringIO(text), sep=",", engine="python")
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
