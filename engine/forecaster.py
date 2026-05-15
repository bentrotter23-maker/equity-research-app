"""
Forward forecast engine — projects NOPLAT, Invested Capital, ROIC, and FCF
based on user-supplied assumptions.

Framework (from Applied Equity Valuation / McKinsey):

    Revenue_t  = Revenue_{t-1} × (1 + growth_t)
    NOPLAT_t   = Revenue_t × NOPBT_margin_t × (1 − tax_rate)

    Reinvestment Rate = g / ROIC          (McKinsey identity)
    ΔIC_t      = NOPLAT_t × reinvestment_rate_t
    FCF_t      = NOPLAT_t − ΔIC_t  =  NOPLAT_t × (1 − g/ROIC)

    IC_t       = IC_{t-1} + ΔIC_t
    ROIC_t     = NOPLAT_t / IC_{t-1}

The user supplies:
  • Per-year revenue growth rates (e.g. [20%, 18%, 15%, 12%, 10%, 8%, 6%, 5%, 5%, 5%])
  • NOPBT margin trajectory: start_margin → end_margin (linear interpolation)
  • Operating tax rate
  • WACC (for Economic Profit and sanity check)
  • Terminal growth rate (for a simple terminal value)

All outputs are in the same $ units as historical data (millions).
"""

import pandas as pd
import numpy as np
from typing import List


def build_forecast(
    base_revenue: float,
    base_ic: float,
    base_noplat: float,
    growth_rates: List[float],         # per-year, e.g. [0.20, 0.18, ...]
    nopbt_margin_start: float,         # decimal, e.g. 0.25
    nopbt_margin_end: float,           # decimal, e.g. 0.35
    tax_rate: float = 0.22,
    wacc: float = 0.09,
    terminal_growth: float = 0.03,
    base_year: int = 2025,
) -> pd.DataFrame:
    """
    Project financials for len(growth_rates) years beyond base_year.

    Returns a DataFrame indexed by forecast year with columns:
        Revenue, NOPBT Margin, Operating EBIT, NOPLAT, Beginning IC, ROIC,
        Reinvestment Rate, ΔIC, FCF, Economic Profit, PV(FCF)
    """
    n = len(growth_rates)
    years = [base_year + i + 1 for i in range(n)]

    # Linearly interpolate NOPBT margin from start → end over the forecast horizon
    margins = np.linspace(nopbt_margin_start, nopbt_margin_end, n)

    rows = []
    rev     = base_revenue
    ic      = base_ic
    noplat  = base_noplat

    for i, (yr, g, margin) in enumerate(zip(years, growth_rates, margins)):
        prev_ic    = ic
        rev        = rev * (1 + g)
        op_ebit    = rev * margin
        noplat     = op_ebit * (1 - tax_rate)

        # ROIC from beginning IC
        roic = noplat / prev_ic if prev_ic != 0 else np.nan

        # Reinvestment rate = g / ROIC (McKinsey identity)
        # Capped at [0, 1] to avoid nonsensical values when ROIC is very high or g < 0
        if roic and not np.isnan(roic) and roic > 0:
            reinvest_rate = np.clip(g / roic, 0.0, 1.0)
        else:
            reinvest_rate = 0.5  # fallback: assume 50% reinvestment

        delta_ic = noplat * reinvest_rate
        fcf      = noplat - delta_ic
        ic       = prev_ic + delta_ic

        ep = (roic - wacc) * prev_ic if not np.isnan(roic) else np.nan

        # Present value of FCF discounted at WACC
        pv_fcf = fcf / ((1 + wacc) ** (i + 1))

        rows.append({
            "Year":              yr,
            "Revenue":           round(rev, 1),
            "Revenue Growth":    g,
            "NOPBT Margin":      round(margin, 4),
            "Operating EBIT":    round(op_ebit, 1),
            "NOPLAT":            round(noplat, 1),
            "Beginning IC":      round(prev_ic, 1),
            "ROIC":              roic,
            "Reinvestment Rate": reinvest_rate,
            "ΔIC":               round(delta_ic, 1),
            "FCF":               round(fcf, 1),
            "Economic Profit":   round(ep, 1) if not np.isnan(ep) else np.nan,
            "PV(FCF)":           round(pv_fcf, 1),
        })

    df = pd.DataFrame(rows).set_index("Year")

    # Simple terminal value (Gordon Growth on terminal FCF)
    terminal_fcf = df["FCF"].iloc[-1] * (1 + terminal_growth)
    terminal_tv  = terminal_fcf / (wacc - terminal_growth) if wacc > terminal_growth else np.nan
    pv_tv        = terminal_tv / ((1 + wacc) ** n)          if wacc > terminal_growth else np.nan

    df.attrs["pv_fcf_sum"]   = df["PV(FCF)"].sum()
    df.attrs["terminal_tv"]  = terminal_tv
    df.attrs["pv_tv"]        = pv_tv
    df.attrs["implied_value"] = df["PV(FCF)"].sum() + (pv_tv or 0)

    return df


def default_growth_schedule(latest_growth: float, terminal_growth: float = 0.03, years: int = 10) -> List[float]:
    """
    Build a smoothly decelerating growth schedule from latest observed growth
    down to terminal growth over `years` years.
    Useful as the starting assumption before the user adjusts.
    """
    # Exponential decay toward terminal
    schedule = []
    for i in range(years):
        t = i / (years - 1)
        g = latest_growth * (1 - t) + terminal_growth * t
        schedule.append(round(g, 4))
    return schedule
