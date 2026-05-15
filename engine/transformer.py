"""
Financial transformation engine.

Replicates the Applied Equity Valuation framework:
  1. NOPLAT  — strip to operating profit, tax-adjust
  2. Invested Capital — OWC + Fixed Capital + Intangible Capital
  3. ROIC / FCF — value driver analysis

All inputs are DataFrames with fiscal year as the index (integers).
All monetary values in millions USD.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Operating Tax Rate
# ─────────────────────────────────────────────────────────────────────────────

def estimate_operating_tax_rate(pretax: pd.Series, tax: pd.Series) -> float:
    """
    Derive a normalized operating tax rate from historical P&L data.

    Why normalize?  Reported tax rates swing wildly year-to-year due to
    deferred tax timing, valuation allowance releases, and one-time items.
    We take the median of 'clean' years (rate between 5% and 45%) so our
    NOPLAT calculation uses a stable, economically meaningful rate.

    Falls back to 22% (US statutory rate) if insufficient clean data.
    """
    rates = []
    for fy in pretax.index:
        pre = pretax.get(fy)
        t = tax.get(fy)
        if pre and pre > 0 and pd.notna(t):
            rate = t / pre
            if 0.05 <= rate <= 0.45:
                rates.append(rate)
    return float(np.median(rates)) if len(rates) >= 2 else 0.22


# ─────────────────────────────────────────────────────────────────────────────
# NOPLAT
# ─────────────────────────────────────────────────────────────────────────────

def compute_noplat(is_df: pd.DataFrame, tax_rate: float) -> pd.DataFrame:
    """
    Compute NOPLAT (Net Operating Profit Less Adjusted Taxes).

    Framework from class:
    ┌─────────────────────────────────────────────────────────┐
    │  Revenue                                                │
    │  − COGS (includes D&A if bundled)                       │
    │  − R&D                                                  │
    │  − SG&A                                                 │
    │  ═══════════════════════════                            │
    │  Operating EBIT  (also called NOPBT — pre-tax)          │
    │  − Estimated Operating Taxes  (EBIT × tax_rate)         │
    │  ═══════════════════════════                            │
    │  NOPLAT                                                 │
    └─────────────────────────────────────────────────────────┘

    We prefer the reported Operating Income (EBIT) if available because
    it captures any line items we might not reconstruct perfectly from
    individual components.
    """
    df = pd.DataFrame(index=is_df.index)

    df["Revenue"]   = is_df["Revenue"]
    df["COGS"]      = is_df["COGS"]
    df["R&D"]       = is_df["R&D"]
    df["SG&A"]      = is_df["SG&A"]
    df["D&A"]       = is_df["D&A"]

    # Build-up Operating EBIT from components
    total_opex = (
        df["COGS"].fillna(0)
        + df["R&D"].fillna(0)
        + df["SG&A"].fillna(0)
    )
    computed_ebit = df["Revenue"] - total_opex

    # Prefer the reported figure when it exists and is materially different
    reported = is_df.get("Operating Income (EBIT)", pd.Series(dtype=float))
    df["Operating EBIT"] = reported.combine_first(computed_ebit)

    df["Operating Tax Rate"]        = tax_rate
    df["Estimated Operating Taxes"] = df["Operating EBIT"] * tax_rate
    df["NOPLAT"]                    = df["Operating EBIT"] - df["Estimated Operating Taxes"]

    # Margins (as decimals — format as % in the UI)
    df["NOPBT Margin"]  = df["Operating EBIT"] / df["Revenue"]
    df["NOPLAT Margin"] = df["NOPLAT"] / df["Revenue"]
    df["Gross Margin"]  = is_df["Gross Profit"] / is_df["Revenue"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Invested Capital
# ─────────────────────────────────────────────────────────────────────────────

def compute_invested_capital(bs_df: pd.DataFrame, revenue: pd.Series) -> pd.DataFrame:
    """
    Compute Invested Capital (IC) — the capital actually deployed to generate NOPLAT.

    Three buckets (matching the class framework):

    ① Operating Working Capital (OWC)
        + Operating Cash  (proxy: 13.5% of revenue — the portion needed to run ops)
        + Accounts Receivable
        − Accounts Payable & Accrued Liabilities   (non-interest-bearing)
        − Deferred Revenue (current)               (customers pre-pay → free funding)
        − Operating Lease Liability (current)

    ② Fixed Capital (FC)
        + PP&E (net)
        + Operating Lease ROU Asset   (leased assets are real economic assets)
        − Other Noncurrent Liabilities (non-debt operating obligations)

    ③ Intangible Capital (IntC)
        + Goodwill
        + Acquired Intangibles (net)

    Key insight for SaaS companies:
    OWC is typically NEGATIVE because deferred revenue (advance payments from
    customers) far exceeds receivables. This means customers effectively fund
    operations — a structural cash advantage relative to other industries.
    """
    df = pd.DataFrame(index=bs_df.index)

    # ① Operating Working Capital
    OPERATING_CASH_PCT = 0.135  # standard proxy; see McKinsey Valuation textbook
    df["Operating Cash"]       = revenue * OPERATING_CASH_PCT
    df["Accounts Receivable"]  = bs_df["Accounts Receivable"].fillna(0)

    df["Accounts Payable"]     = bs_df["Accounts Payable"].fillna(0)
    df["Accrued Liabilities"]  = bs_df["Accrued Liabilities"].fillna(0)
    df["Deferred Rev (Curr)"]  = bs_df["Deferred Revenue (Current)"].fillna(0)
    df["OL Liab (Curr)"]       = bs_df["OL Liability (Current)"].fillna(0)

    df["Operating Current Assets"] = df["Operating Cash"] + df["Accounts Receivable"]
    df["Operating Current Liabilities"] = (
        df["Accounts Payable"]
        + df["Accrued Liabilities"]
        + df["Deferred Rev (Curr)"]
        + df["OL Liab (Curr)"]
    )
    df["Operating Working Capital"] = df["Operating Current Assets"] - df["Operating Current Liabilities"]

    # ② Fixed Capital
    df["PP&E Net"]               = bs_df["PP&E (Net)"].fillna(0)
    df["Operating Lease ROU"]    = bs_df["Operating Lease ROU Asset"].fillna(0)
    df["Other NCL"]              = bs_df["Other Noncurrent Liabilities"].fillna(0)

    df["Total Fixed Capital"] = df["PP&E Net"] + df["Operating Lease ROU"] - df["Other NCL"]

    # ③ Intangible Capital
    df["Goodwill"]        = bs_df["Goodwill"].fillna(0)
    df["Intangibles Net"] = bs_df["Intangibles (Net)"].fillna(0)
    df["Total Intangible Capital"] = df["Goodwill"] + df["Intangibles Net"]

    # Total
    df["Total Invested Capital"] = (
        df["Operating Working Capital"]
        + df["Total Fixed Capital"]
        + df["Total Intangible Capital"]
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ROIC & FCF
# ─────────────────────────────────────────────────────────────────────────────

def compute_roic_fcf(noplat_df: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ROIC, FCF, and Economic Profit.

    ROIC = NOPLAT_t / Beginning IC_{t-1}
        Why beginning IC? Because that's the capital base that was available
        to generate this year's NOPLAT.

    FCF = NOPLAT − ΔIC
        Not all NOPLAT becomes cash — some must be reinvested to sustain/grow
        the business. FCF is what remains for investors.

    IC Turnover = Revenue / Beginning IC
        Efficiency measure — how many $ of revenue per $ of IC deployed.

    ROIC Decomposition:
        ROIC ≈ NOPBT Margin × IC Turnover × (1 − Tax Rate)
        This reveals whether ROIC is driven by profitability or efficiency.
    """
    df = pd.DataFrame(index=noplat_df.index)

    df["Revenue"]          = noplat_df["Revenue"]
    df["NOPLAT"]           = noplat_df["NOPLAT"]
    df["Operating EBIT"]   = noplat_df["Operating EBIT"]
    df["Total IC"]         = ic_df["Total Invested Capital"]
    df["OWC"]              = ic_df["Operating Working Capital"]
    df["Fixed Capital"]    = ic_df["Total Fixed Capital"]
    df["Intangible Capital"] = ic_df["Total Intangible Capital"]

    # Beginning IC (prior year end = this year's start)
    df["Beginning IC"] = df["Total IC"].shift(1)

    # ROIC
    df["ROIC"] = df["NOPLAT"] / df["Beginning IC"]

    # Pre-tax ROIC
    df["Pre-Tax ROIC"] = df["Operating EBIT"] / df["Beginning IC"]

    # FCF
    df["ΔIC"]  = df["Total IC"] - df["Beginning IC"]
    df["FCF"]  = df["NOPLAT"] - df["ΔIC"]

    # IC Turnover
    df["IC Turnover"] = df["Revenue"] / df["Beginning IC"]

    # Margins
    df["NOPBT Margin"]  = noplat_df["NOPBT Margin"]
    df["NOPLAT Margin"] = noplat_df["NOPLAT Margin"]

    # Incremental ROIC (ΔNOPLAT / ΔIC) — how productive is new investment?
    df["ΔNOPLAT"]        = df["NOPLAT"].diff()
    df["Incremental ROIC"] = df["ΔNOPLAT"] / df["ΔIC"].abs().replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Common-Size Statements
# ─────────────────────────────────────────────────────────────────────────────

def common_size_is(is_df: pd.DataFrame) -> pd.DataFrame:
    """Divide every IS line item by Revenue. Revenue row = 1.0."""
    cs = is_df.copy()
    for col in cs.columns:
        cs[col] = cs[col] / is_df["Revenue"]
    cs["Revenue"] = 1.0
    return cs


def common_size_bs(bs_df: pd.DataFrame) -> pd.DataFrame:
    """Divide every BS item by Total Assets."""
    cs = bs_df.copy()
    for col in cs.columns:
        cs[col] = cs[col] / bs_df["Total Assets"]
    return cs


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_saas_metrics(
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    noplat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SaaS-specific unit economics and efficiency metrics.

    All metrics derive from already-fetched IS/BS/CF data — no new API calls.

    Columns returned:
        Revenue Growth %       — YoY revenue growth
        Gross Margin %         — Gross Profit / Revenue
        COGS %                 — COGS / Revenue
        R&D %                  — R&D / Revenue
        SG&A %                 — SG&A / Revenue
        SBC %                  — Stock-Based Comp / Revenue (dilution intensity)
        FCF Margin %           — Free Cash Flow / Revenue
        Capex Intensity %      — CapEx / Revenue
        DSO                    — Days Sales Outstanding (AR / Revenue × 365)
        DPO                    — Days Payable Outstanding (AP / COGS × 365)
        CCC                    — Cash Conversion Cycle (DSO − DPO; negative is SaaS strength)
        Deferred Rev Total     — Current + Noncurrent deferred revenue
        Deferred Rev % Rev     — Total deferred revenue / Revenue
        Deferred Rev Growth %  — YoY growth of total deferred revenue
        Rule of 40             — Revenue Growth % + FCF Margin %
        Tangible IC            — IC minus Goodwill and Intangibles (excludes M&A inflation)
        Quality ROIC           — NOPLAT / Beginning Tangible IC
        Goodwill % of IC       — Goodwill / Total IC (M&A intensity)
        Operating Leverage     — EBIT Growth % / Revenue Growth % (>1 = scaling efficiently)
        IC Growth %            — YoY change in Total Invested Capital
    """
    df = pd.DataFrame(index=is_df.index)

    rev  = is_df["Revenue"]
    cogs = is_df["COGS"].fillna(0)
    gp   = is_df["Gross Profit"].fillna(rev - cogs)

    # ── P&L ratios ────────────────────────────────────────────────────
    df["Revenue Growth %"]   = rev.pct_change() * 100
    df["Gross Margin %"]     = gp / rev * 100
    df["COGS %"]             = cogs / rev * 100
    df["R&D %"]              = is_df["R&D"].fillna(0) / rev * 100
    df["SG&A %"]             = is_df["SG&A"].fillna(0) / rev * 100
    df["SBC %"]              = is_df["Stock-Based Compensation"].fillna(0) / rev * 100

    # ── Cash efficiency ────────────────────────────────────────────────
    fcf  = cf_df["Free Cash Flow"]
    capex = cf_df["Capital Expenditures"].abs()
    df["FCF Margin %"]       = fcf / rev * 100
    df["Capex Intensity %"]  = capex / rev * 100

    # ── Working capital days ───────────────────────────────────────────
    ar   = bs_df["Accounts Receivable"].fillna(0)
    ap   = bs_df["Accounts Payable"].fillna(0)
    # Protect DPO: some SaaS cos report near-zero COGS (100% gross margin)
    dpo_denom = cogs.replace(0, np.nan)
    df["DSO"] = ar / rev * 365
    df["DPO"] = ap / dpo_denom * 365
    df["CCC"] = df["DSO"] - df["DPO"]

    # ── Deferred revenue (forward bookings indicator) ──────────────────
    dr_curr   = bs_df["Deferred Revenue (Current)"].fillna(0)
    dr_nc     = bs_df["Deferred Revenue (Noncurrent)"].fillna(0)
    dr_total  = dr_curr + dr_nc
    df["Deferred Rev Total"]    = dr_total
    df["Deferred Rev % Rev"]    = dr_total / rev * 100
    df["Deferred Rev Growth %"] = dr_total.pct_change() * 100

    # ── Rule of 40 ────────────────────────────────────────────────────
    df["Rule of 40"] = df["Revenue Growth %"] + df["FCF Margin %"]

    # ── ROIC quality (tangible IC only, strips M&A goodwill) ──────────
    gw       = ic_df["Goodwill"].fillna(0)
    intang   = ic_df["Intangibles Net"].fillna(0)
    total_ic = ic_df["Total Invested Capital"]
    tang_ic  = total_ic - gw - intang

    # Quality ROIC is only meaningful when Tangible IC is positive and material.
    # Heavy acquirers (e.g. CRM) can have Goodwill+Intangibles > Total IC,
    # making Tangible IC negative — set those to NaN so they show as N/M.
    tang_ic_safe = tang_ic.where(tang_ic > total_ic.abs() * 0.01, np.nan)

    df["Tangible IC"]       = tang_ic
    df["Quality ROIC"]      = noplat_df["NOPLAT"] / tang_ic_safe.shift(1) * 100
    df["Goodwill % of IC"]  = gw / total_ic.replace(0, np.nan) * 100

    # ── Operating leverage ────────────────────────────────────────────
    ebit_g = noplat_df["Operating EBIT"].pct_change()
    rev_g  = rev.pct_change().replace(0, np.nan)
    df["Operating Leverage"] = ebit_g / rev_g

    # ── IC growth ─────────────────────────────────────────────────────
    df["IC Growth %"] = total_ic.pct_change() * 100

    # ── ROIC decomposition attribution ───────────────────────────────
    # ΔROIC ≈ (Δmargin × prior IC turnover) + (Δturnover × current margin)
    margin  = noplat_df["NOPBT Margin"]
    turnover = ic_df["Total Invested Capital"].shift(1)
    turnover_ratio = rev / ic_df["Total Invested Capital"].shift(1)
    df["Margin Contribution to ROIC"] = margin.diff() * turnover_ratio.shift(1) * (1 - 0.22)
    df["Turnover Contribution to ROIC"] = turnover_ratio.diff() * margin.shift(1) * (1 - 0.22)

    return df


def run_analysis(ticker: str, raw_data: dict) -> dict:
    """
    Run the complete valuation framework pipeline on raw SEC data.

    Steps:
      1. Estimate operating tax rate
      2. Compute NOPLAT
      3. Compute Invested Capital
      4. Compute ROIC / FCF
      5. Build common-size statements
      6. Compute SaaS unit economics metrics
    """
    is_df = raw_data["is"]
    bs_df = raw_data["bs"]
    cf_df = raw_data["cf"]

    tax_rate = estimate_operating_tax_rate(is_df["Pretax Income"], is_df["Income Tax Expense"])

    noplat_df = compute_noplat(is_df, tax_rate)
    ic_df     = compute_invested_capital(bs_df, is_df["Revenue"])
    roic_df   = compute_roic_fcf(noplat_df, ic_df)
    saas_df   = compute_saas_metrics(is_df, bs_df, cf_df, ic_df, noplat_df)

    cs_is = common_size_is(is_df)
    cs_bs = common_size_bs(bs_df)

    return {
        "ticker":       ticker,
        "company":      raw_data["company"],
        "cik":          raw_data.get("cik", ""),
        "fiscal_years": raw_data["fiscal_years"],
        "is":           is_df,
        "bs":           bs_df,
        "cf":           cf_df,
        "cs_is":        cs_is,
        "cs_bs":        cs_bs,
        "noplat":       noplat_df,
        "ic":           ic_df,
        "roic":         roic_df,
        "saas":         saas_df,
        "tax_rate":     tax_rate,
    }
