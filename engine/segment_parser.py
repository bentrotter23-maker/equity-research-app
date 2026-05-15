"""
Revenue segment parser — pulls structured breakdown from SEC EDGAR 10-K filings.

Strategy:
  1. Get the latest 10-K accession number via the submissions API.
  2. Fetch FilingSummary.xml — an index of every rendered XBRL "R-file"
     (small HTML table fragments the EDGAR viewer uses, ~10-50 KB each).
  3. Find R-files whose ShortName contains revenue/segment/disaggregat keywords.
  4. Parse the specific EDGAR R-file table structure.

EDGAR R-file revenue table format:
  Multi-level columns: [title row, date row]
  Rows alternate between "Disaggregation of Revenue [Line Items]" headers (NaN)
  and "Total revenues" rows with actual dollar values.
  The segment name appears 2 rows above each "Total revenues" row.
"""

import re
import time
import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "equity-research-app bentrotter23@gmail.com"}
BASE = "https://www.sec.gov"

SEGMENT_KEYWORDS = [
    "disaggregat", "revenue", "segment", "subscription",
    "geographic", "product line",
]

# Prefer "Details" R-files — these have the actual numbers, not just labels
PREFER_DETAIL = True


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — get latest 10-K accession
# ─────────────────────────────────────────────────────────────────────────────

def _get_latest_10k(cik: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (accession_no_dashes, period) for the most recent 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    forms      = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    # The correct key is "reportDate" (not "periodOfReport")
    periods    = filings.get("reportDate", filings.get("periodOfReport", [""]*len(forms)))

    for i, form in enumerate(forms):
        if form == "10-K":
            return accessions[i].replace("-", ""), periods[i]

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — find revenue R-files from FilingSummary.xml
# ─────────────────────────────────────────────────────────────────────────────

def _find_revenue_r_files(cik: str, accn: str) -> List[Tuple[str, str]]:
    """
    Return list of (short_name, r_filename) matching revenue/segment keywords.
    Uses the integer CIK (no leading zeros) in the URL path — EDGAR requires this.
    """
    cik_int = int(cik)
    url = f"{BASE}/Archives/edgar/data/{cik_int}/{accn}/FilingSummary.xml"
    resp = requests.get(url, headers=HEADERS, timeout=12)
    if resp.status_code != 200:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    matches = []
    for report in root.iter("Report"):
        short_name = (report.findtext("ShortName") or "").lower()
        html_file  = report.findtext("HtmlFileName") or ""
        if html_file and any(kw in short_name for kw in SEGMENT_KEYWORDS):
            matches.append((short_name, html_file))

    # Prefer "Details" files — they have real numbers, not just narrative/labels
    if PREFER_DETAIL:
        detail = [(n, f) for n, f in matches if "detail" in n]
        if detail:
            return detail

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — parse EDGAR R-file into a clean DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _extract_date_cols(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    """Find column indices and labels that look like fiscal period dates."""
    if isinstance(df.columns, pd.MultiIndex):
        col_labels = [str(c[1]) for c in df.columns]
    else:
        col_labels = [str(c) for c in df.columns]

    date_re = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d\d)", re.I)
    idxs, names = [], []
    for i, label in enumerate(col_labels):
        if date_re.search(label):
            idxs.append(i)
            names.append(label.strip())
    return idxs, names


def _clean_val(val) -> Optional[float]:
    """Convert an EDGAR cell value to float (handles '$41,525', '(123)', NaN)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).replace(",", "").replace("$", "").replace("\xa0", "").strip()
    s = re.sub(r"^\((.+)\)$", r"-\1", s)   # (123) → -123
    try:
        return float(s)
    except ValueError:
        return None


def _parse_r_file(cik: str, accn: str, r_file: str) -> Optional[pd.DataFrame]:
    """
    Download one EDGAR R-file and parse it into a Segment × Period DataFrame.

    EDGAR revenue detail R-files follow this row pattern:
        Row N:   segment name (e.g. "Americas")
        Row N+1: "Disaggregation of Revenue [Line Items]"  (NaN data)
        Row N+2: "Total revenues"   ← actual dollar values here

    We collect every "Total revenues" row and find the segment name 2 rows above.
    """
    cik_int = int(cik)
    url = f"{BASE}/Archives/edgar/data/{cik_int}/{accn}/{r_file}"
    resp = requests.get(url, headers=HEADERS, timeout=12)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return None

    for table in tables:
        try:
            raw = pd.read_html(str(table), header=[0, 1])
        except Exception:
            continue
        if not raw:
            continue

        df = raw[0]
        date_idxs, date_names = _extract_date_cols(df)
        if not date_idxs:
            continue

        # Verify this table has revenue-related content
        flat_text = df.to_string().lower()
        if "revenue" not in flat_text:
            continue

        # Walk rows to extract (segment_name → {date: value})
        segments: Dict[str, Dict[str, Optional[float]]] = {}
        for i in range(len(df)):
            label = str(df.iloc[i, 0]).strip()
            if re.search(r"total revenue", label, re.I):
                # Hunt for the segment name in the rows above
                seg_name = None
                for j in range(i - 1, max(i - 4, -1), -1):
                    candidate = str(df.iloc[j, 0]).strip()
                    skip = re.search(
                        r"disaggregation|line items|NaN|^$", candidate, re.I
                    )
                    if not skip and candidate.lower() not in ("nan", ""):
                        seg_name = candidate
                        break
                if seg_name is None:
                    seg_name = "Total Revenue"

                values = {
                    date_names[k]: _clean_val(df.iloc[i, date_idxs[k]])
                    for k in range(len(date_idxs))
                }
                # Only add if at least one real number
                if any(v is not None for v in values.values()):
                    segments[seg_name] = values

        if segments:
            result = pd.DataFrame(segments).T
            result.index.name = "Segment"
            # Convert all columns to numeric
            for col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
            # Drop all-NaN columns (sometimes EDGAR adds a blank qualifier column)
            result = result.dropna(axis=1, how="all")
            # Shorten column headers: "Jan. 31, 2026" → keep as-is (readable)
            # but strip long titles if multi-level leaked through
            result.columns = [
                re.sub(r"^.*(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\..*)", r"\1", c).strip()
                if re.search(r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec", c) else c
                for c in result.columns
            ]
            return result

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_segments(cik: str) -> Dict:
    """
    Fetch revenue segment breakdown from the latest 10-K.

    Returns:
        {
          "period":  "2026-01-31",
          "tables":  [df, ...],   # one DataFrame per matched R-file
          "error":   None or str,
        }

    DataFrame index = segment names.
    DataFrame columns = period date strings (e.g. "Jan. 31, 2026").
    Values in millions USD (as reported in the filing).
    """
    try:
        accn, period = _get_latest_10k(cik)
        if not accn:
            return {"period": None, "tables": [], "error": "No 10-K filing found."}

        time.sleep(0.15)
        r_files = _find_revenue_r_files(cik, accn)
        if not r_files:
            return {"period": period, "tables": [],
                    "error": "No revenue R-files found in FilingSummary.xml."}

        tables = []
        seen   = set()
        for short_name, r_file in r_files[:8]:
            time.sleep(0.1)
            df = _parse_r_file(cik, accn, r_file)
            if df is not None and not df.empty:
                key = (df.shape, tuple(df.index[:3]))
                if key not in seen:
                    df.attrs["source"] = short_name
                    tables.append(df)
                    seen.add(key)

        if not tables:
            return {"period": period, "tables": [],
                    "error": "Revenue R-files found but no structured tables extracted."}

        return {"period": period, "tables": tables, "error": None}

    except Exception as e:
        return {"period": None, "tables": [], "error": str(e)}
