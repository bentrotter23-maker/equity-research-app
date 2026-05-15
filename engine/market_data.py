"""
Market data engine — live stock prices, enterprise value, and peer comps.
Uses yfinance for real-time and trailing-twelve-months data.

All monetary values in millions USD unless noted.
"""

import warnings
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# Default SaaS / enterprise software peer universe
SAAS_PEERS = [
    "CRM", "NOW", "MSFT", "ADBE", "INTU", "WDAY",
    "ZS",  "CRWD", "NET",  "SNOW", "ORCL", "SAP",
]

# Industry benchmark ranges (SaaS / enterprise software) from public market data
INDUSTRY_BENCHMARKS = {
    "EV/Revenue":  {"low": 4,  "median": 9,  "high": 18},
    "EV/EBITDA":   {"low": 20, "median": 35, "high": 60},
    "EV/EBIT":     {"low": 25, "median": 45, "high": 80},
    "P/E":         {"low": 30, "median": 55, "high": 100},
}


def _safe_get(info: dict, *keys, default=None):
    """Try multiple keys, return the first non-None value."""
    for key in keys:
        v = info.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def get_stock_quote(ticker: str) -> dict:
    """
    Fetch current price, market cap, and enterprise value for a single ticker.

    Returns:
        {
          "ticker":         str,
          "name":           str,
          "price":          float,   # USD
          "market_cap":     float,   # millions USD
          "enterprise_value": float, # millions USD
          "ev_revenue":     float | None,
          "ev_ebitda":      float | None,
          "ev_ebit":        float | None,
          "pe":             float | None,
          "revenue_ttm":    float | None,  # millions USD
          "ebitda_ttm":     float | None,  # millions USD
          "error":          str | None,
        }
    """
    if not _YF_AVAILABLE:
        return {"ticker": ticker, "error": "yfinance not installed"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)
            info = t.info or {}

        price = _safe_get(info, "currentPrice", "regularMarketPrice", "previousClose", default=None)
        market_cap_raw = _safe_get(info, "marketCap", default=None)
        ev_raw         = _safe_get(info, "enterpriseValue", default=None)
        revenue_raw    = _safe_get(info, "totalRevenue", default=None)
        ebitda_raw     = _safe_get(info, "ebitda", default=None)
        ebit_raw       = _safe_get(info, "ebit", default=None)

        # Convert to millions
        market_cap = market_cap_raw / 1e6 if market_cap_raw else None
        ev         = ev_raw         / 1e6 if ev_raw         else None
        revenue    = revenue_raw    / 1e6 if revenue_raw    else None
        ebitda     = ebitda_raw     / 1e6 if ebitda_raw     else None
        ebit       = ebit_raw       / 1e6 if ebit_raw       else None

        # Multiples
        ev_rev   = ev / revenue if ev and revenue and revenue > 0 else None
        ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else None
        ev_ebit  = ev / ebit    if ev and ebit   and ebit   > 0 else None
        pe       = _safe_get(info, "trailingPE", "forwardPE", default=None)

        return {
            "ticker":           ticker.upper(),
            "name":             _safe_get(info, "shortName", "longName", default=ticker),
            "price":            price,
            "market_cap":       market_cap,
            "enterprise_value": ev,
            "ev_revenue":       round(ev_rev,   2) if ev_rev   else None,
            "ev_ebitda":        round(ev_ebitda, 2) if ev_ebitda else None,
            "ev_ebit":          round(ev_ebit,  2) if ev_ebit  else None,
            "pe":               round(pe, 2)        if pe       else None,
            "revenue_ttm":      revenue,
            "ebitda_ttm":       ebitda,
            "error":            None,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_peer_comps(subject_ticker: str, peers=None) -> pd.DataFrame:
    """
    Build a peer comps table including the subject ticker.

    Args:
        subject_ticker: The company being analyzed.
        peers: Optional custom list of peer tickers; defaults to SAAS_PEERS.

    Returns:
        DataFrame with columns:
            Company | Ticker | Price | Market Cap ($B) | EV ($B) |
            EV/Revenue | EV/EBITDA | EV/EBIT | P/E
        Rows sorted by EV/Revenue descending.
        Subject ticker row highlighted via df.attrs["subject"].
    """
    if peers is None:
        peers = SAAS_PEERS

    # Ensure subject is in the list
    tickers = list({subject_ticker.upper()} | {p.upper() for p in peers})

    rows = []
    for tk in tickers:
        q = get_stock_quote(tk)
        if q.get("error"):
            continue
        rows.append({
            "Ticker":       tk,
            "Company":      q["name"],
            "Price":        q["price"],
            "Market Cap":   round(q["market_cap"]       / 1_000, 2) if q["market_cap"]       else None,  # $B
            "EV":           round(q["enterprise_value"] / 1_000, 2) if q["enterprise_value"] else None,  # $B
            "EV/Revenue":   q["ev_revenue"],
            "EV/EBITDA":    q["ev_ebitda"],
            "EV/EBIT":      q["ev_ebit"],
            "P/E":          q["pe"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("EV/Revenue", ascending=False, na_position="last").reset_index(drop=True)
    df.attrs["subject"] = subject_ticker.upper()
    return df
