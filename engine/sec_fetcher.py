"""
SEC EDGAR data fetcher.
Pulls 10-K annual financial data via the EDGAR XBRL API (data.sec.gov).
All monetary values returned in millions USD.
"""

import requests
import pandas as pd
import time
from typing import Optional, Dict, List

HEADERS = {"User-Agent": "equity-research-app bentrotter23@gmail.com"}
BASE_URL = "https://data.sec.gov"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# Each key maps to a priority-ordered list of XBRL concept names.
# We try each in order and use the first one that has data.
CONCEPTS: Dict[str, List[str]] = {
    # ── Income Statement ─────────────────────────────────────────────
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "rnd": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
    ],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ],
    "depreciation": [
        "Depreciation",
    ],
    "amortization_intangibles": [
        "AmortizationOfIntangibleAssets",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax": [
        "IncomeTaxExpenseBenefit",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    # ── Balance Sheet ─────────────────────────────────────────────────
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
    ],
    "total_current_assets": [
        "AssetsCurrent",
    ],
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
    ],
    "operating_lease_rou": [
        "OperatingLeaseRightOfUseAsset",
    ],
    "goodwill": [
        "Goodwill",
    ],
    "intangibles_net": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "total_assets": [
        "Assets",
    ],
    "accounts_payable": [
        "AccountsPayableCurrent",
    ],
    "accrued_liabilities": [
        "AccruedLiabilitiesCurrent",
        "EmployeeRelatedLiabilitiesCurrent",
        "OtherLiabilitiesCurrent",
    ],
    "deferred_revenue_current": [
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueCurrent",
    ],
    "total_current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "deferred_revenue_noncurrent": [
        "ContractWithCustomerLiabilityNoncurrent",
        "DeferredRevenueNoncurrent",
    ],
    "operating_lease_liability_current": [
        "OperatingLeaseLiabilityCurrent",
    ],
    "operating_lease_liability_noncurrent": [
        "OperatingLeaseLiabilityNoncurrent",
    ],
    "other_noncurrent_liabilities": [
        "OtherLiabilitiesNoncurrent",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "additional_paid_in_capital": [
        "AdditionalPaidInCapital",
        "AdditionalPaidInCapitalCommonStock",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
    ],
    # ── Cash Flow ─────────────────────────────────────────────────────
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "cfi": [
        "NetCashProvidedByUsedInInvestingActivities",
    ],
    "cff": [
        "NetCashProvidedByUsedInFinancingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
    ],
    "stock_repurchase": [
        "PaymentsForRepurchaseOfCommonStock",
    ],
    "dividends_paid": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ],
}


def get_cik(ticker: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    resp = requests.get(TICKER_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    ticker_upper = ticker.upper()
    for entry in resp.json().values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_company_name(cik: str) -> str:
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json().get("name", "Unknown")


def _extract_annual(facts: dict, concept: str) -> Dict[int, float]:
    """
    Pull annual 10-K values for a single XBRL concept.
    Returns {fiscal_year: value_in_millions}.
    Deduplicates by fiscal year, keeping the most recently filed value.
    """
    us_gaap = facts.get("us-gaap", {})
    if concept not in us_gaap:
        return {}

    usd_entries = us_gaap[concept].get("units", {}).get("USD", [])
    annual = [e for e in usd_entries if e.get("form") in ("10-K", "10-K/A") and e.get("fp") == "FY"]

    by_fy: Dict[int, dict] = {}
    for e in annual:
        fy = e.get("fy")
        if fy and (fy not in by_fy or e["filed"] > by_fy[fy]["filed"]):
            by_fy[fy] = e

    return {fy: e["val"] / 1_000_000 for fy, e in by_fy.items()}


def _resolve(facts: dict, field: str) -> Dict[int, float]:
    """
    Merge data from ALL concepts for a field across years.

    Why merge instead of first-match?
    Many companies switch XBRL concepts over time. Salesforce, for example,
    reported revenue under 'Revenues' through FY2017, then switched to
    'RevenueFromContractWithCustomerExcludingAssessedTax' starting FY2019
    when ASC 606 was adopted. A first-match strategy only returns 7 years;
    merging returns the full 10+ year history.

    Conflict resolution: higher-priority concepts (earlier in the list)
    overwrite lower-priority ones for the same fiscal year.
    """
    merged: Dict[int, float] = {}
    # Iterate reversed so the highest-priority concept (index 0) wins conflicts
    for concept in reversed(CONCEPTS.get(field, [])):
        merged.update(_extract_annual(facts, concept))
    return merged


def fetch_financials(ticker: str, years: int = 10) -> dict:
    """
    Fetch and structure 10-K financial statements from SEC EDGAR.

    Returns a dict with keys:
        company, cik, ticker, fiscal_years, is, bs, cf
    All monetary values in millions USD.
    """
    cik = get_cik(ticker)
    if not cik:
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR. Check the symbol and try again.")

    company_name = get_company_name(cik)
    time.sleep(0.15)  # respect EDGAR rate limits

    facts_url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(facts_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    facts = resp.json().get("facts", {})

    # Resolve all fields
    raw: Dict[str, Dict[int, float]] = {field: _resolve(facts, field) for field in CONCEPTS}

    # Determine fiscal years from revenue (most reliable anchor)
    if not raw["revenue"]:
        raise ValueError(f"No revenue data found for {ticker}. The company may not file XBRL data.")

    all_fy = sorted(raw["revenue"].keys())
    recent_fy = all_fy[-years:] if len(all_fy) >= years else all_fy

    def series(field: str) -> pd.Series:
        return pd.Series({fy: raw[field].get(fy) for fy in recent_fy}, dtype=float)

    # ── Income Statement ──────────────────────────────────────────────
    is_df = pd.DataFrame({
        "Revenue":                  series("revenue"),
        "COGS":                     series("cogs"),
        "Gross Profit":             series("gross_profit"),
        "R&D":                      series("rnd"),
        "SG&A":                     series("sga"),
        "D&A":                      series("da"),
        "Depreciation":             series("depreciation"),
        "Amortization of Intangibles": series("amortization_intangibles"),
        "Operating Income (EBIT)":  series("operating_income"),
        "Interest Expense":         series("interest_expense"),
        "Pretax Income":            series("pretax_income"),
        "Income Tax Expense":       series("tax"),
        "Net Income":               series("net_income"),
        "Stock-Based Compensation": series("sbc"),
    })
    # Derive Gross Profit if the company doesn't report it directly
    if is_df["Gross Profit"].isna().all():
        is_df["Gross Profit"] = is_df["Revenue"] - is_df["COGS"]

    # ── Balance Sheet ─────────────────────────────────────────────────
    bs_df = pd.DataFrame({
        "Cash & Equivalents":                   series("cash"),
        "Short-Term Investments":               series("short_term_investments"),
        "Accounts Receivable":                  series("accounts_receivable"),
        "Total Current Assets":                 series("total_current_assets"),
        "PP&E (Net)":                           series("ppe_net"),
        "Operating Lease ROU Asset":            series("operating_lease_rou"),
        "Goodwill":                             series("goodwill"),
        "Intangibles (Net)":                    series("intangibles_net"),
        "Total Assets":                         series("total_assets"),
        "Accounts Payable":                     series("accounts_payable"),
        "Accrued Liabilities":                  series("accrued_liabilities"),
        "Deferred Revenue (Current)":           series("deferred_revenue_current"),
        "OL Liability (Current)":               series("operating_lease_liability_current"),
        "Total Current Liabilities":            series("total_current_liabilities"),
        "Long-Term Debt":                       series("long_term_debt"),
        "Deferred Revenue (Noncurrent)":        series("deferred_revenue_noncurrent"),
        "OL Liability (Noncurrent)":            series("operating_lease_liability_noncurrent"),
        "Other Noncurrent Liabilities":         series("other_noncurrent_liabilities"),
        "Total Liabilities":                    series("total_liabilities"),
        "Retained Earnings":                    series("retained_earnings"),
        "Additional Paid-In Capital":           series("additional_paid_in_capital"),
        "Total Equity":                         series("total_equity"),
    })

    # ── Cash Flow ─────────────────────────────────────────────────────
    cf_df = pd.DataFrame({
        "Operating Cash Flow":  series("cfo"),
        "Capital Expenditures": series("capex"),
        "Acquisitions":         series("acquisitions"),
        "Investing Cash Flow":  series("cfi"),
        "Stock Repurchases":    series("stock_repurchase"),
        "Dividends Paid":       series("dividends_paid"),
        "Financing Cash Flow":  series("cff"),
        "D&A (CFS)":            series("da"),
        "SBC (CFS)":            series("sbc"),
    })
    # FCF = CFO - CapEx (CapEx is reported as positive outflow in XBRL)
    cf_df["Free Cash Flow"] = cf_df["Operating Cash Flow"] - cf_df["Capital Expenditures"].abs()

    return {
        "company":      company_name,
        "cik":          cik,
        "ticker":       ticker.upper(),
        "fiscal_years": recent_fy,
        "is":           is_df,
        "bs":           bs_df,
        "cf":           cf_df,
    }
