"""
Equity Research Application
4-Phase intentional flow: Discover → Analyze → Investigate → Thesis
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.sec_fetcher    import fetch_financials
from engine.transformer    import run_analysis
from engine.segment_parser import fetch_segments
from engine.forecaster     import build_forecast, default_growth_schedule
from engine.market_data    import get_stock_quote, get_peer_comps, INDUSTRY_BENCHMARKS
from db.thesis             import ThesisTracker

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sequoia-inspired CSS theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #F2EDE3 !important;
    color: #0A2015 !important;
}

/* ── Background enforcement ── */
[data-testid="stApp"], .stApp, section.main, .main,
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stBottom"], [data-testid="stDecoration"],
.block-container, .main .block-container {
    background-color: #F2EDE3 !important;
}
.main .block-container { padding-top: 2rem; max-width: 1300px; }

/* ── Body text — targeted only, avoids dark-bg elements ── */
.stMarkdown p, .stMarkdown li, .stMarkdown strong, .stMarkdown em,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: #0A2015 !important;
    opacity: 1 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #0A2015 !important; }
[data-testid="stSidebar"] * { color: #E8E0D0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #B8D4A0 !important; }
[data-testid="stSidebar"] hr  { border-color: #235A42 !important; }

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #0A2015 !important;
    letter-spacing: -0.01em !important;
}
h1 { font-size: 2.4rem !important; font-weight: 500 !important; }
h2 { font-size: 1.7rem !important; font-weight: 500 !important; }
h3 { font-size: 1.2rem !important; font-weight: 500 !important; }

/* ── Buttons — green at rest, darker on hover ── */
.stButton > button {
    background-color: #235A42 !important;
    color: #F2EDE3 !important;
    border: none !important;
    border-radius: 5px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background-color: #0A2015 !important; }
/* Ensure button text stays cream regardless of global overrides */
.stButton > button p,
.stButton > button span { color: #F2EDE3 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #E8E2D8 !important;
    border-radius: 7px !important;
    padding: 4px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #3A5A34 !important;
    border-radius: 5px !important;
    padding: 0.4rem 1rem !important;
    border: none !important;
    letter-spacing: 0.01em !important;
}
.stTabs [aria-selected="true"] {
    background-color: #0A2015 !important;
    color: #F2EDE3 !important;
}
/* Ensure selected tab label text stays cream */
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div { color: #F2EDE3 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #FAFAF7 !important;
    border: 1px solid #D5CEC0 !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 3px rgba(10,32,21,0.06) !important;
}
[data-testid="stMetricLabel"] {
    color: #527055 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #0A2015 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem !important;
    font-weight: 500 !important;
}

/* ── Inputs ── */
.stTextInput > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextArea textarea,
.stNumberInput input {
    background-color: #FAFAF7 !important;
    border: 1px solid #C5BDB0 !important;
    border-radius: 5px !important;
    color: #0A2015 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Widget labels (number inputs, selects, toggles, text areas) ── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stTextInput label p, .stTextArea label p,
.stNumberInput label p, .stSelectbox label p,
.stToggle label p {
    color: #0A2015 !important;
    opacity: 1 !important;
}

/* ── Sliders ── */
.stSlider > div > div > div { background-color: #235A42 !important; }

/* ── Dataframes ── */
.stDataFrame { border-radius: 7px !important; overflow: hidden !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background-color: #FAFAF7 !important;
    border: 1px solid #D5CEC0 !important;
    border-radius: 7px !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: #0A2015 !important; }

/* ── Phase pill (dark bg + light text — uses inline color, not overridden) ── */
.phase-pill {
    display: inline-block;
    background-color: #0A2015;
    color: #B8D4A0 !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 3px 13px;
    border-radius: 20px;
    margin-bottom: 0.6rem;
}

/* ── Divider ── */
.eq-divider {
    border: none;
    border-top: 1px solid #CECCBC;
    margin: 1.5rem 0;
}

/* ── Info/alert boxes ── */
[data-testid="stInfo"] {
    background-color: #E8F0E8 !important;
    border-left: 3px solid #235A42 !important;
    border-radius: 6px !important;
}
[data-testid="stInfo"] p { color: #0A2015 !important; }
[data-testid="stWarning"] {
    background-color: #FEF8EC !important;
    border-left: 3px solid #B8860B !important;
}
[data-testid="stWarning"] p { color: #5A4000 !important; }

/* ── Radio / checkbox labels ── */
.stRadio label, .stCheckbox label,
.stRadio p, .stCheckbox p {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0A2015 !important;
    opacity: 1 !important;
}

/* ── Caption / small text ── */
.stCaption p, small {
    color: #6B7C6A !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "phase":            1,
        "ticker":           "",
        "goals":            [],
        "analysis":         None,
        "raw_data":         None,
        "segments":         None,
        "quote":            None,
        "phase3_answers":   {},
        "thesis_submitted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_financials(ticker: str) -> dict:
    return fetch_financials(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_segments(cik: str) -> dict:
    return fetch_segments(cik)

@st.cache_data(ttl=300, show_spinner=False)
def _load_quote(ticker: str) -> dict:
    return get_stock_quote(ticker)

@st.cache_data(ttl=300, show_spinner=False)
def _load_comps(ticker: str) -> pd.DataFrame:
    return get_peer_comps(ticker)

# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_usd(v, decimals=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.1f}T"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}B"
    return f"${v:,.{decimals}f}M"

def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.{decimals}f}%"

def fmt_mult(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}x"

def md_safe(s: str) -> str:
    """
    Escape '$' so Streamlit/markdown doesn't treat dollar amounts as LaTeX math.
    Use whenever embedding fmt_usd() output inside a markdown string.
    """
    return s.replace("$", r"\$") if isinstance(s, str) else s


# ─────────────────────────────────────────────────────────────────────────────
# Accounting-book table styler
# ─────────────────────────────────────────────────────────────────────────────
#
# Bold rows = subtotals / totals
# Top border = section divider above the row
# Double border = grand-total row

def style_statement(
    fmt_df: pd.DataFrame,
    bold_rows: set,
    top_border_rows: set = None,
    double_total_rows: set = None,
):
    """
    Apply accounting-book formatting:
      • bold for subtotals/totals
      • subtle top border above each subtotal
      • thicker double border for grand totals (Net Income, Total Assets, etc.)
    """
    top_border_rows   = top_border_rows   or set()
    double_total_rows = double_total_rows or set()

    def row_style(row):
        styles = []
        for _ in row:
            s = "font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem;"
            if row.name in bold_rows:
                s += " font-weight: 600; color: #0A2015;"
                s += " background-color: #ECE7DA;"
            else:
                s += " color: #2A3D2E;"
            if row.name in top_border_rows or row.name in bold_rows:
                s += " border-top: 1px solid #8A9282;"
            if row.name in double_total_rows:
                s += " border-top: 2px solid #0A2015; border-bottom: 2px double #0A2015;"
            styles.append(s)
        return styles

    def index_style(idx):
        styles = []
        for v in idx:
            s = "font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem;"
            if v in bold_rows:
                s += " font-weight: 600; color: #0A2015; background-color: #ECE7DA;"
            else:
                s += " color: #2A3D2E; padding-left: 14px;"
            if v in top_border_rows or v in bold_rows:
                s += " border-top: 1px solid #8A9282;"
            if v in double_total_rows:
                s += " border-top: 2px solid #0A2015; border-bottom: 2px double #0A2015;"
            styles.append(s)
        return styles

    styler = (
        fmt_df.style
        .apply(row_style, axis=1)
        .apply_index(index_style, axis=0)
        .set_table_styles([
            {"selector": "thead th",
             "props": "background-color: #0A2015 !important; color: #F2EDE3 !important; "
                      "font-family: 'Space Grotesk', sans-serif !important; "
                      "font-weight: 500; font-size: 0.82rem; "
                      "text-transform: uppercase; letter-spacing: 0.05em; "
                      "padding: 8px 12px; border: none;"},
            {"selector": "tbody td",
             "props": "padding: 6px 12px; color: #2A3D2E !important; "
                      "background-color: #FAFAF7 !important;"},
            {"selector": "tbody tr:nth-child(even) td",
             "props": "background-color: #F5F0E8 !important;"},
            {"selector": "tbody th",
             "props": "padding: 6px 12px; text-align: left; color: #2A3D2E !important; "
                      "background-color: #FAFAF7 !important;"},
            {"selector": "tbody tr:nth-child(even) th",
             "props": "background-color: #F5F0E8 !important;"},
            {"selector": "table",
             "props": "border-collapse: collapse; width: 100%; background-color: #FAFAF7 !important;"},
        ])
    )
    return styler

# ─────────────────────────────────────────────────────────────────────────────
# Plot theme
# ─────────────────────────────────────────────────────────────────────────────

GREEN        = "#0A2015"
MID_GREEN    = "#235A42"
LIGHT_GREEN  = "#6AAF90"
AMBER        = "#B8860B"
CREAM        = "#F2EDE3"

# NOTE: xaxis/yaxis are intentionally excluded from PLOTLY_LAYOUT.
# Passing them here AND as explicit kwargs in update_layout() causes
# "multiple values for keyword argument" TypeError. Apply axis styles
# via fig.update_xaxes() / fig.update_yaxes() separately.
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CREAM,
    plot_bgcolor="#FAFAF7",
    font=dict(family="Space Grotesk", color=GREEN),
    legend=dict(bgcolor=CREAM, bordercolor="#CECCBC", borderwidth=1,
                font=dict(family="Space Grotesk", size=12)),
    colorway=[GREEN, MID_GREEN, AMBER, LIGHT_GREEN, "#8B6914", "#9DC09A"],
)
# When a chart needs a title, apply it explicitly via fig.update_layout(title=dict(text=..., font=...))

# Axis styling applied separately via update_xaxes / update_yaxes
_AX = dict(gridcolor="#E8E4DC", linecolor="#C8C0B0",
           tickfont=dict(family="Space Grotesk", color=GREEN),
           title_font=dict(family="Space Grotesk", color=GREEN))

# ─────────────────────────────────────────────────────────────────────────────
# Phase progress bar
# ─────────────────────────────────────────────────────────────────────────────

def _phase_bar(current: int):
    labels = ["Discover", "Analyze", "Investigate", "Thesis"]
    cols   = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, labels)):
        n = i + 1
        if n < current:
            color, icon = MID_GREEN, "✓"
        elif n == current:
            color, icon = GREEN, str(n)
        else:
            color, icon = "#C8C4BC", str(n)
        col.markdown(f"""
        <div style="text-align:center">
            <div style="width:32px;height:32px;border-radius:50%;background:{color};
                        color:#F2EDE3;display:inline-flex;align-items:center;
                        justify-content:center;font-family:'Space Grotesk',sans-serif;
                        font-weight:600;font-size:0.85rem;margin-bottom:4px">{icon}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.68rem;
                        font-weight:500;color:{'#0A2015' if n<=current else '#A0A0A0'};
                        text-transform:uppercase;letter-spacing:0.08em">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr class='eq-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Phase 3 question generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_questions(a: dict, goals: list) -> list:
    """Return 5 company-specific investigative questions based on actual data."""
    roic_df   = a["roic"]
    noplat_df = a["noplat"]
    is_df     = a["is"]
    company   = a["company"]
    ticker    = a["ticker"]

    questions = []

    # Q1 — Revenue trajectory
    rev = is_df["Revenue"].dropna()
    if len(rev) >= 3:
        latest_g = (rev.iloc[-1] / rev.iloc[-2] - 1) * 100
        prior_g  = (rev.iloc[-2] / rev.iloc[-3] - 1) * 100
        delta_g  = latest_g - prior_g
        trend    = ("decelerating" if delta_g < -3
                    else "accelerating" if delta_g > 3 else "stable")
        questions.append({
            "label": "Revenue Trajectory",
            "q": (
                f"{company}'s revenue growth is {trend} — "
                f"most recently {latest_g:.1f}% vs {prior_g:.1f}% the prior year. "
                f"What is the primary driver of this "
                f"{'slowdown' if trend=='decelerating' else 'acceleration' if trend=='accelerating' else 'consistency'}, "
                f"and does your research suggest this trend will continue, stabilize, or reverse over the next 3 years?"
            ),
        })

    # Q2 — Margin structure
    nopbt = noplat_df["NOPBT Margin"].dropna()
    if len(nopbt) >= 2:
        margin_chg = (nopbt.iloc[-1] - nopbt.iloc[0]) * 100
        direction  = ("expanded" if margin_chg > 2
                      else "compressed" if margin_chg < -2 else "been roughly flat")
        questions.append({
            "label": "Margin Structure",
            "q": (
                f"{ticker}'s NOPBT margin has {direction} by {abs(margin_chg):.1f} ppts "
                f"over the past {len(nopbt)} years "
                f"(from {nopbt.iloc[0]*100:.1f}% to {nopbt.iloc[-1]*100:.1f}%). "
                f"Is this margin level structurally defensible given the competitive environment? "
                f"What is your target NOPBT margin in 5 years, and why?"
            ),
        })

    # Q3 — ROIC & capital allocation
    roic = roic_df["ROIC"].dropna()
    if len(roic) >= 2 and a["ic"]["Total Invested Capital"].iloc[-1] != 0:
        latest_roic = roic.iloc[-1] * 100
        ic_intang   = (
            a["ic"]["Total Intangible Capital"].iloc[-1]
            / a["ic"]["Total Invested Capital"].iloc[-1] * 100
        )
        questions.append({
            "label": "ROIC and Capital Deployment",
            "q": (
                f"{ticker} is generating a {latest_roic:.1f}% ROIC, "
                f"with {ic_intang:.0f}% of invested capital in intangible assets "
                f"(goodwill + acquired intangibles). "
                f"Given {'the heavy M&A footprint' if ic_intang > 50 else 'the asset-light model'}, "
                f"how do you assess management's capital allocation discipline? "
                f"Is incremental ROIC on recent investments trending up or down — and what's driving it?"
            ),
        })

    # Q4 — Goal-aware moat / business model question
    if "Revenue Decomposition" in goals:
        questions.append({
            "label": "Revenue Mix and Moat",
            "q": (
                f"Looking at {company}'s revenue decomposition by segment and geography: "
                f"which segment or region is the highest-quality growth driver, and why? "
                f"What evidence from the 10-K footnotes (contracts, switching costs, NRR) "
                f"supports the durability of that revenue?"
            ),
        })
    elif "ROIC Analysis" in goals:
        questions.append({
            "label": "Competitive Moat Drivers",
            "q": (
                f"ROIC above WACC indicates {company} creates economic value. "
                f"Identify the 2-3 specific competitive advantages (pricing power, network effects, "
                f"cost structure, switching costs) that sustain returns above cost of capital. "
                f"What would have to be true for those advantages to erode?"
            ),
        })
    else:
        questions.append({
            "label": "Competitive Moat",
            "q": (
                f"What is {company}'s primary source of competitive advantage, "
                f"and how does it show up in the financial data "
                f"(gross margin, revenue per employee, customer retention)? "
                f"What is the single biggest risk to that moat?"
            ),
        })

    # Q5 — Variant perception
    questions.append({
        "label": "Variant Perception",
        "q": (
            f"What does your analysis reveal that is NOT yet reflected in the consensus view of {ticker}? "
            f"State your differentiated perspective on one of: "
            f"(a) the revenue growth trajectory, "
            f"(b) the margin expansion opportunity, or "
            f"(c) the capital allocation direction. "
            f"If the market is pricing {ticker} correctly, why would you still own it — or not?"
        ),
    })

    return questions[:5]

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Equity Research")
    st.markdown("---")

    if st.session_state.analysis:
        a = st.session_state.analysis
        st.markdown(f"**{a['company']}**")
        st.markdown(f"`{a['ticker']}` · Phase {st.session_state.phase} of 4")
        st.markdown("---")
        if st.session_state.goals:
            st.markdown("**Research Goals**")
            for g in st.session_state.goals:
                st.markdown(f"• {g}")
        st.markdown("---")

    if st.session_state.phase > 1:
        if st.button("← New Company", use_container_width=True):
            for k in ["phase", "ticker", "goals", "analysis", "raw_data",
                      "segments", "quote", "phase3_answers", "thesis_submitted"]:
                del st.session_state[k]
            _init_state()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#7A9B7A'>Data: SEC EDGAR XBRL API"
        "<br>Prices: Yahoo Finance</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  PHASE 1: DISCOVER  ════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.phase == 1:
    _phase_bar(1)

    st.markdown("<div class='phase-pill'>Phase 1 — Discover</div>", unsafe_allow_html=True)
    st.title("Who are you researching?")
    st.markdown(
        "Enter a ticker and define your research goals. "
        "The app tailors analysis and investigative questions to your focus."
    )

    col_a, col_b = st.columns([1, 1.5], gap="large")

    with col_a:
        st.markdown("#### Company")
        ticker_input = st.text_input(
            "Ticker symbol",
            value=st.session_state.ticker,
            placeholder="e.g. CRM, NOW, MSFT",
            label_visibility="collapsed",
        ).strip().upper()

    with col_b:
        st.markdown("#### Research Focus  *(select all that apply)*")
        GOAL_OPTIONS = [
            "Revenue Trajectory",
            "Margin Analysis",
            "ROIC Analysis",
            "Revenue Decomposition",
            "Forecast and Valuation",
            "Competitive Positioning",
        ]
        selected_goals = []
        g_cols = st.columns(2)
        for i, goal in enumerate(GOAL_OPTIONS):
            with g_cols[i % 2]:
                checked = goal in st.session_state.goals
                if st.checkbox(goal, value=checked, key=f"goal_{i}"):
                    selected_goals.append(goal)

    st.markdown("")
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        go_btn = st.button("Begin Analysis →", use_container_width=True)

    if go_btn:
        if not ticker_input:
            st.error("Please enter a ticker symbol.")
        elif not selected_goals:
            st.warning("Select at least one research focus to continue.")
        else:
            with st.spinner(f"Loading {ticker_input} from SEC EDGAR…"):
                try:
                    raw      = _load_financials(ticker_input)
                    analysis = run_analysis(ticker_input, raw)
                    st.session_state.raw_data  = raw
                    st.session_state.analysis  = analysis
                    st.session_state.ticker    = ticker_input
                    st.session_state.goals     = selected_goals
                    st.session_state.phase     = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load {ticker_input}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  PHASE 2: ANALYZE  ═════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.phase == 2:
    _phase_bar(2)
    a = st.session_state.analysis

    # Auto-refresh analysis if it's missing newer keys (e.g. cached from a previous code version)
    if "saas" not in a or "cik" not in a:
        try:
            a = run_analysis(st.session_state.ticker, st.session_state.raw_data)
            st.session_state.analysis = a
        except Exception as _e:
            st.error(f"Could not refresh analysis: {_e}. Click '← New Company' and reload.")
            st.stop()

    # Header
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown("<div class='phase-pill'>Phase 2 — Analyze</div>", unsafe_allow_html=True)
        st.title(a["company"])
        st.markdown(
            f"**{a['ticker']}** · {len(a['fiscal_years'])} years of 10-K data · "
            "Values in millions USD"
        )
    with hc2:
        if st.session_state.quote is None:
            with st.spinner("Loading quote…"):
                st.session_state.quote = _load_quote(a["ticker"])
        q = st.session_state.quote or {}
        if not q.get("error"):
            if q.get("price"):
                st.metric("Price",      f"${q['price']:,.2f}")
            if q.get("market_cap"):
                st.metric("Market Cap", fmt_usd(q["market_cap"]))
            if q.get("enterprise_value"):
                st.metric("EV",         fmt_usd(q["enterprise_value"]))

    st.markdown("---")

    tabs = st.tabs([
        "Income Statement",
        "Balance Sheet",
        "Cash Flow",
        "NOPLAT & ROIC",
        "Revenue & Segments",
        "SaaS Metrics",
        "Forecast",
        "Multiples",
    ])

    # ── Tab 0: Income Statement ───────────────────────────────────────
    with tabs[0]:
        is_df = a["is"]
        rev_s = is_df["Revenue"].dropna()

        k1, k2, k3, k4 = st.columns(4)
        rev_v  = rev_s.iloc[-1]
        gm_v   = is_df["Gross Profit"].dropna().iloc[-1] / rev_v * 100 if rev_v else 0
        ni_v   = is_df["Net Income"].dropna().iloc[-1] if is_df["Net Income"].notna().any() else None
        yoy_v  = (rev_s.iloc[-1] / rev_s.iloc[-2] - 1) * 100 if len(rev_s) >= 2 else None
        k1.metric("Revenue",       fmt_usd(rev_v),     f"{yoy_v:+.1f}% YoY" if yoy_v else None)
        k2.metric("Gross Margin",  f"{gm_v:.1f}%")
        k3.metric("Net Income",    fmt_usd(ni_v))
        k4.metric("Years of Data", str(len(a["fiscal_years"])))

        st.markdown("---")
        cs_tog = st.toggle("Common-size (% of Revenue)", value=False, key="cs_is")
        d = (a["cs_is"] if cs_tog else is_df)

        rows_is = ["Revenue","COGS","Gross Profit","R&D","SG&A","D&A",
                   "Operating Income (EBIT)","Interest Expense","Pretax Income",
                   "Income Tax Expense","Net Income","Stock-Based Compensation"]
        d = d[[c for c in rows_is if c in d.columns]].T
        if cs_tog:
            fmt_d = d.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        else:
            fmt_d = d.map(lambda x: fmt_usd(x) if pd.notna(x) else "—")

        is_bold = {"Gross Profit", "Operating Income (EBIT)", "Pretax Income", "Net Income"}
        is_double = {"Net Income"}
        st.markdown(
            style_statement(fmt_d, bold_rows=is_bold, double_total_rows=is_double).to_html(),
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.markdown("#### Revenue and Gross Profit Trend")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=is_df.index, y=is_df["Revenue"],
                             name="Revenue", marker_color=MID_GREEN, opacity=0.85))
        fig.add_trace(go.Bar(x=is_df.index, y=is_df["Gross Profit"],
                             name="Gross Profit", marker_color=LIGHT_GREEN, opacity=0.85))
        if len(rev_s) >= 2:
            yoy_list = [(rev_s.iloc[i]/rev_s.iloc[i-1]-1)*100 for i in range(1, len(rev_s))]
            fig.add_trace(go.Scatter(
                x=rev_s.index[1:], y=yoy_list, name="YoY Growth %",
                yaxis="y2", line=dict(color=AMBER, width=2.5), mode="lines+markers",
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, barmode="overlay",
            yaxis2=dict(title="Growth %", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(family="Space Grotesk", color=AMBER),
                        title_font=dict(family="Space Grotesk", color=AMBER)),
        )
        fig.update_xaxes(**_AX)
        fig.update_yaxes(title_text="USD millions", **_AX)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 1: Balance Sheet ──────────────────────────────────────────
    with tabs[1]:
        bs_df = a["bs"]
        ic_df = a["ic"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Assets",     fmt_usd(bs_df["Total Assets"].dropna().iloc[-1]))
        k2.metric("Total Equity",     fmt_usd(bs_df["Total Equity"].dropna().iloc[-1] if bs_df["Total Equity"].notna().any() else None))
        k3.metric("Long-Term Debt",   fmt_usd(bs_df["Long-Term Debt"].dropna().iloc[-1] if bs_df["Long-Term Debt"].notna().any() else None))
        k4.metric("Invested Capital", fmt_usd(ic_df["Total Invested Capital"].dropna().iloc[-1] if ic_df["Total Invested Capital"].notna().any() else None))

        st.markdown("---")
        cs_bs = st.toggle("Common-size (% of Total Assets)", value=False, key="cs_bs")
        d_bs  = (a["cs_bs"] if cs_bs else bs_df)

        rows_bs = ["Cash & Equivalents","Short-Term Investments","Accounts Receivable",
                   "Total Current Assets","PP&E (Net)","Goodwill","Intangibles (Net)",
                   "Total Assets","Accounts Payable","Deferred Revenue (Current)",
                   "Total Current Liabilities","Long-Term Debt","Total Liabilities",
                   "Retained Earnings","Total Equity"]
        d_bs = d_bs[[c for c in rows_bs if c in d_bs.columns]].T
        if cs_bs:
            fmt_bs = d_bs.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        else:
            fmt_bs = d_bs.map(lambda x: fmt_usd(x) if pd.notna(x) else "—")

        bs_bold = {"Total Current Assets", "Total Assets",
                   "Total Current Liabilities", "Total Liabilities", "Total Equity"}
        bs_double = {"Total Assets", "Total Equity"}
        st.markdown(
            style_statement(fmt_bs, bold_rows=bs_bold, double_total_rows=bs_double).to_html(),
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.markdown("#### Invested Capital Composition (Latest Year)")
        latest_ic = ic_df.dropna(subset=["Total Invested Capital"]).iloc[-1]
        ic_comp = {
            "Op. Working Capital": float(latest_ic.get("Operating Working Capital", 0) or 0),
            "Fixed Capital":       float(latest_ic.get("Total Fixed Capital", 0) or 0),
            "Intangible Capital":  float(latest_ic.get("Total Intangible Capital", 0) or 0),
        }
        fig_ic = go.Figure(go.Bar(
            x=list(ic_comp.keys()), y=list(ic_comp.values()),
            marker_color=[GREEN, MID_GREEN, LIGHT_GREEN],
        ))
        fig_ic.update_layout(**PLOTLY_LAYOUT, title="IC Decomposition (USD millions)", showlegend=False)
        fig_ic.update_xaxes(**_AX)
        fig_ic.update_yaxes(**_AX)
        st.plotly_chart(fig_ic, use_container_width=True)

    # ── Tab 2: Cash Flow ─────────────────────────────────────────────
    with tabs[2]:
        cf_df = a["cf"]

        k1, k2, k3, k4 = st.columns(4)
        cfo_v  = cf_df["Operating Cash Flow"].dropna().iloc[-1] if cf_df["Operating Cash Flow"].notna().any() else None
        fcf_v  = cf_df["Free Cash Flow"].dropna().iloc[-1]      if cf_df["Free Cash Flow"].notna().any()      else None
        cap_v  = cf_df["Capital Expenditures"].dropna().iloc[-1] if cf_df["Capital Expenditures"].notna().any() else None
        fcf_mg = (fcf_v / a["is"]["Revenue"].dropna().iloc[-1] * 100) if (fcf_v and a["is"]["Revenue"].notna().any()) else None
        k1.metric("Operating CF",   fmt_usd(cfo_v))
        k2.metric("Free Cash Flow", fmt_usd(fcf_v))
        k3.metric("CapEx",          fmt_usd(cap_v))
        k4.metric("FCF Margin",     f"{fcf_mg:.1f}%" if fcf_mg else "—")

        st.markdown("---")
        rows_cf = ["Operating Cash Flow","Capital Expenditures","Free Cash Flow",
                   "Acquisitions","Investing Cash Flow","Stock Repurchases",
                   "Dividends Paid","Financing Cash Flow","D&A (CFS)","SBC (CFS)"]
        cf_disp = cf_df[[c for c in rows_cf if c in cf_df.columns]].T
        fmt_cf  = cf_disp.map(lambda x: fmt_usd(x) if pd.notna(x) else "—")

        cf_bold = {"Operating Cash Flow", "Free Cash Flow",
                   "Investing Cash Flow", "Financing Cash Flow"}
        cf_double = {"Free Cash Flow"}
        st.markdown(
            style_statement(fmt_cf, bold_rows=cf_bold, double_total_rows=cf_double).to_html(),
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.markdown("#### CFO vs FCF vs Net Income")
        fig_cf = go.Figure()
        for col, color, name in [
            ("Operating Cash Flow", GREEN,     "CFO"),
            ("Free Cash Flow",      MID_GREEN, "FCF"),
        ]:
            if col in cf_df.columns:
                fig_cf.add_trace(go.Scatter(
                    x=cf_df.index, y=cf_df[col], name=name,
                    line=dict(color=color, width=2.5), mode="lines+markers",
                ))
        ni = a["is"]["Net Income"]
        if ni.notna().any():
            fig_cf.add_trace(go.Scatter(
                x=ni.index, y=ni, name="Net Income",
                line=dict(color=AMBER, width=2, dash="dash"), mode="lines+markers",
            ))
        fig_cf.update_layout(**PLOTLY_LAYOUT)
        fig_cf.update_xaxes(**_AX)
        fig_cf.update_yaxes(**_AX)
        st.plotly_chart(fig_cf, use_container_width=True)

    # ── Tab 3: NOPLAT & ROIC ─────────────────────────────────────────
    with tabs[3]:
        nd  = a["noplat"]
        rd  = a["roic"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("NOPLAT",       fmt_usd(nd["NOPLAT"].dropna().iloc[-1]))
        k2.metric("ROIC",         fmt_pct(rd["ROIC"].dropna().iloc[-1] if rd["ROIC"].notna().any() else None))
        k3.metric("NOPBT Margin", fmt_pct(nd["NOPBT Margin"].dropna().iloc[-1]))
        k4.metric("IC Turnover",  fmt_mult(rd["IC Turnover"].dropna().iloc[-1] if rd["IC Turnover"].notna().any() else None))

        st.markdown("---")
        cl, cr = st.columns(2)

        with cl:
            st.markdown("#### NOPLAT vs Revenue")
            fig_n = go.Figure()
            fig_n.add_trace(go.Bar(x=nd.index, y=nd["Revenue"],
                                   name="Revenue", marker_color=LIGHT_GREEN, opacity=0.7))
            fig_n.add_trace(go.Bar(x=nd.index, y=nd["NOPLAT"],
                                   name="NOPLAT", marker_color=GREEN))
            fig_n.update_layout(**PLOTLY_LAYOUT, barmode="overlay")
            fig_n.update_xaxes(**_AX)
            fig_n.update_yaxes(**_AX)
            st.plotly_chart(fig_n, use_container_width=True)

        with cr:
            st.markdown("#### ROIC Decomposition")
            fig_r = make_subplots(specs=[[{"secondary_y": True}]])
            fig_r.add_trace(go.Scatter(
                x=rd.index, y=rd["ROIC"]*100,
                name="ROIC %", line=dict(color=GREEN, width=3),
            ), secondary_y=False)
            fig_r.add_trace(go.Scatter(
                x=rd.index, y=rd["NOPBT Margin"]*100,
                name="NOPBT Margin %", line=dict(color=MID_GREEN, width=2, dash="dash"),
            ), secondary_y=False)
            if rd["IC Turnover"].notna().any():
                fig_r.add_trace(go.Scatter(
                    x=rd.index, y=rd["IC Turnover"],
                    name="IC Turnover", line=dict(color=AMBER, width=2),
                ), secondary_y=True)
            fig_r.update_layout(**PLOTLY_LAYOUT)
            fig_r.update_xaxes(**_AX)
            fig_r.update_yaxes(title_text="% (ROIC, Margin)", secondary_y=False, **_AX)
            fig_r.update_yaxes(title_text="IC Turnover (x)", secondary_y=True, **_AX)
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("#### Historical NOPLAT and ROIC Detail")
        show_r = ["Revenue","NOPLAT","Operating EBIT","NOPBT Margin","NOPLAT Margin",
                  "Total IC","Beginning IC","ROIC","IC Turnover","FCF","ΔIC"]
        rd_disp = rd[[c for c in show_r if c in rd.columns]].T
        pct_rs  = {"NOPBT Margin","NOPLAT Margin","ROIC"}
        mul_rs  = {"IC Turnover"}
        fmt_rd  = pd.DataFrame(index=rd_disp.index, columns=rd_disp.columns, dtype=object)
        for row in rd_disp.index:
            for col in rd_disp.columns:
                v = rd_disp.at[row, col]
                if pd.isna(v):
                    fmt_rd.at[row, col] = "—"
                elif row in pct_rs:
                    fmt_rd.at[row, col] = fmt_pct(v)
                elif row in mul_rs:
                    fmt_rd.at[row, col] = fmt_mult(v)
                else:
                    fmt_rd.at[row, col] = fmt_usd(v)
        st.dataframe(fmt_rd, use_container_width=True)

        st.markdown("---")

        # ── ROIC decomposition attribution ────────────────────────────
        sd = a.get("saas", pd.DataFrame())
        st.markdown("#### ROIC Attribution — Margin vs. Efficiency")
        st.caption(
            "ΔROIC decomposed into: contribution from NOPBT Margin change "
            "vs. IC Turnover change. Reveals whether returns are driven by "
            "pricing power/scale or by asset efficiency."
        )
        margin_contrib  = sd["Margin Contribution to ROIC"].dropna() * 100
        turnover_contrib = sd["Turnover Contribution to ROIC"].dropna() * 100
        if len(margin_contrib) >= 2:
            fig_attr = go.Figure()
            fig_attr.add_trace(go.Bar(
                x=margin_contrib.index, y=margin_contrib,
                name="Margin Driver", marker_color=MID_GREEN,
            ))
            fig_attr.add_trace(go.Bar(
                x=turnover_contrib.index, y=turnover_contrib,
                name="IC Efficiency Driver", marker_color=AMBER,
            ))
            fig_attr.update_layout(
                **PLOTLY_LAYOUT, barmode="relative",
                yaxis_ticksuffix=" ppts",
            )
            fig_attr.update_xaxes(**_AX)
            fig_attr.update_yaxes(**_AX, title_text="Contribution to ΔROIC (ppts)")
            st.plotly_chart(fig_attr, use_container_width=True)

        # ── Incremental ROIC heatmap ──────────────────────────────────
        st.markdown("#### Incremental ROIC by Year")
        st.caption(
            "ΔNOPLAt / ΔIC — shows whether each year's new capital investment "
            "earned above or below cost of capital. Green = value creating."
        )
        inc_roic = rd["Incremental ROIC"].dropna()
        if not inc_roic.empty:
            colors_inc = [
                GREEN if v > 0.09 else AMBER if v > 0 else "#C0392B"
                for v in inc_roic
            ]
            fig_inc = go.Figure(go.Bar(
                x=inc_roic.index, y=inc_roic * 100,
                marker_color=colors_inc,
                text=[f"{v*100:.0f}%" for v in inc_roic],
                textposition="outside",
            ))
            # WACC reference line at 9%
            fig_inc.add_hline(
                y=9, line_dash="dash", line_color=AMBER,
                annotation_text="WACC ~9%",
                annotation_position="top right",
            )
            fig_inc.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            fig_inc.update_xaxes(**_AX)
            fig_inc.update_yaxes(**_AX, title_text="Incremental ROIC %")
            st.plotly_chart(fig_inc, use_container_width=True)

        # ── Quality ROIC vs Reported ROIC + Capital Allocation ────────
        qr_col, ca_col = st.columns(2)

        with qr_col:
            st.markdown("#### Quality ROIC vs. Reported ROIC")
            st.caption("Tangible IC excludes Goodwill + Intangibles. The gap = M&A capital drag.")
            roic_pct    = rd["ROIC"].dropna() * 100
            quality_pct = sd["Quality ROIC"].dropna()
            if not roic_pct.empty:
                fig_qr = go.Figure()
                fig_qr.add_trace(go.Scatter(
                    x=roic_pct.index, y=roic_pct,
                    name="Reported ROIC", line=dict(color=GREEN, width=2.5),
                    mode="lines+markers",
                ))
                if not quality_pct.empty:
                    fig_qr.add_trace(go.Scatter(
                        x=quality_pct.index, y=quality_pct,
                        name="Quality ROIC (Tangible IC)", line=dict(color=AMBER, width=2, dash="dash"),
                        mode="lines+markers",
                    ))
                gw_pct = sd["Goodwill % of IC"].dropna()
                if not gw_pct.empty:
                    latest_gw = gw_pct.iloc[-1]
                    fig_qr.add_annotation(
                        x=gw_pct.index[-1], y=roic_pct.iloc[-1],
                        text=f"Goodwill: {latest_gw:.0f}% of IC",
                        showarrow=True, arrowhead=2,
                        font=dict(size=10, color=AMBER),
                        arrowcolor=AMBER,
                    )
                fig_qr.update_layout(**PLOTLY_LAYOUT)
                fig_qr.update_xaxes(**_AX)
                fig_qr.update_yaxes(**_AX, title_text="ROIC %")
                st.plotly_chart(fig_qr, use_container_width=True)

        with ca_col:
            st.markdown("#### Capital Allocation — Where IC Growth Comes From")
            st.caption("Each bar shows how total invested capital grew, split by type of deployment.")
            ic_d = a["ic"]
            owc_chg  = ic_d["Operating Working Capital"].diff().fillna(0)
            fix_chg  = ic_d["Total Fixed Capital"].diff().fillna(0)
            intg_chg = ic_d["Total Intangible Capital"].diff().fillna(0)
            valid_idx = owc_chg.index[1:]  # skip first year (no diff)
            if len(valid_idx) >= 2:
                fig_ca = go.Figure()
                fig_ca.add_trace(go.Bar(
                    x=valid_idx, y=owc_chg[valid_idx],
                    name="OWC Change", marker_color=LIGHT_GREEN,
                ))
                fig_ca.add_trace(go.Bar(
                    x=valid_idx, y=fix_chg[valid_idx],
                    name="Fixed Capital", marker_color=MID_GREEN,
                ))
                fig_ca.add_trace(go.Bar(
                    x=valid_idx, y=intg_chg[valid_idx],
                    name="M&A / Intangibles", marker_color=GREEN,
                ))
                fig_ca.update_layout(**PLOTLY_LAYOUT, barmode="relative")
                fig_ca.update_xaxes(**_AX)
                fig_ca.update_yaxes(**_AX, title_text="IC Change (USD millions)")
                st.plotly_chart(fig_ca, use_container_width=True)

    # ── Tab 4: Revenue & Segments ─────────────────────────────────────
    with tabs[4]:
        st.markdown("#### Revenue and Gross Margin Trend")
        rev_s = a["is"]["Revenue"].dropna()
        gm_s  = (a["is"]["Gross Profit"] / a["is"]["Revenue"]).dropna() * 100

        fig_rv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rv.add_trace(go.Bar(x=rev_s.index, y=rev_s,
                                name="Revenue", marker_color=MID_GREEN, opacity=0.8),
                         secondary_y=False)
        fig_rv.add_trace(go.Scatter(x=gm_s.index, y=gm_s,
                                    name="Gross Margin %",
                                    line=dict(color=AMBER, width=2.5), mode="lines+markers"),
                         secondary_y=True)
        fig_rv.update_layout(**PLOTLY_LAYOUT)
        fig_rv.update_xaxes(**_AX)
        fig_rv.update_yaxes(title_text="Revenue (USD millions)", secondary_y=False, **_AX)
        fig_rv.update_yaxes(title_text="Gross Margin %", secondary_y=True, **_AX)
        st.plotly_chart(fig_rv, use_container_width=True)

        st.markdown("#### Revenue Segment Breakdown (from 10-K)")
        cik = a.get("cik", "")
        if not cik:
            st.warning("CIK not available — cannot load segment data.")
        else:
            if st.session_state.segments is None:
                with st.spinner("Parsing segment data from SEC EDGAR 10-K…"):
                    st.session_state.segments = _load_segments(cik)

            seg = st.session_state.segments
            if seg.get("error"):
                st.info(f"ℹ️ {seg['error']}")
            elif seg.get("tables"):
                for i, df_seg in enumerate(seg["tables"]):
                    src = df_seg.attrs.get("source", f"Table {i+1}")
                    with st.expander(f"Segment: {src.title()}", expanded=(i == 0)):
                        fmt_seg = df_seg.map(lambda x: fmt_usd(x) if pd.notna(x) else "—")
                        st.dataframe(fmt_seg, use_container_width=True)
                        if len(df_seg) > 1:
                            fig_s = go.Figure()
                            for seg_name in df_seg.index:
                                fig_s.add_trace(go.Bar(
                                    name=seg_name,
                                    x=df_seg.columns.tolist(),
                                    y=df_seg.loc[seg_name].tolist(),
                                ))
                            fig_s.update_layout(
                                **PLOTLY_LAYOUT,
                                barmode="stack",
                                title=f"{src.title()} — Revenue by Segment",
                            )
                            fig_s.update_xaxes(title_text="Period", **_AX)
                            fig_s.update_yaxes(title_text="USD millions", **_AX)
                            st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("No segment tables extracted from this 10-K.")

    # ── Tab 5: SaaS Metrics ───────────────────────────────────────────
    with tabs[5]:
        sd = a.get("saas")
        if sd is None or (isinstance(sd, pd.DataFrame) and sd.empty):
            st.warning("SaaS metrics could not be computed for this company.")
        else:
            # ── Section A: Rule of 40 Scorecard ──────────────────────
            st.markdown("#### Rule of 40 Scorecard")
            st.caption(
                "Rule of 40 = Revenue Growth % + FCF Margin %.  "
                "Score ≥ 40 is considered healthy for SaaS/enterprise software. "
                "Green bars ≥ 40  ·  Amber 20–40  ·  Red < 20."
            )
            # saas_df stores % values already multiplied by 100 (e.g. 15.5 means 15.5%)
            r40   = sd["Rule of 40"].dropna()        if "Rule of 40"        in sd.columns else pd.Series(dtype=float)
            revg  = sd["Revenue Growth %"].dropna()  if "Revenue Growth %"  in sd.columns else pd.Series(dtype=float)
            fcfm  = sd["FCF Margin %"].dropna()      if "FCF Margin %"      in sd.columns else pd.Series(dtype=float)
            sbc_s = sd["SBC %"].dropna()             if "SBC %"             in sd.columns else pd.Series(dtype=float)

            if not r40.empty:
                k1, k2, k3, k4 = st.columns(4)
                latest_r40 = float(r40.iloc[-1])
                k1.metric(
                    "Rule of 40 (Latest)",
                    f"{latest_r40:.1f}",
                    delta=f"{latest_r40 - float(r40.iloc[-2]):.1f} vs prior" if len(r40) >= 2 else None,
                )
                k2.metric("FCF Margin",     f"{float(fcfm.iloc[-1]):.1f}%"  if not fcfm.empty  else "—")
                k3.metric("Revenue Growth", f"{float(revg.iloc[-1]):.1f}%"  if not revg.empty  else "—")
                k4.metric("SBC % Revenue",  f"{float(sbc_s.iloc[-1]):.1f}%" if not sbc_s.empty else "—")

                st.markdown("")
                colors_r40 = [GREEN if v >= 40 else AMBER if v >= 20 else "#C0392B" for v in r40]
                fig_r40 = go.Figure()
                fig_r40.add_trace(go.Bar(
                    x=r40.index, y=r40, marker_color=colors_r40,
                    text=[f"{v:.1f}" for v in r40], textposition="outside", name="Rule of 40",
                ))
                if not revg.empty:
                    fig_r40.add_trace(go.Scatter(
                        x=revg.index, y=revg, name="Revenue Growth %",
                        line=dict(color=MID_GREEN, width=2, dash="dash"), mode="lines+markers",
                    ))
                if not fcfm.empty:
                    fig_r40.add_trace(go.Scatter(
                        x=fcfm.index, y=fcfm, name="FCF Margin %",
                        line=dict(color=LIGHT_GREEN, width=2, dash="dot"), mode="lines+markers",
                    ))
                fig_r40.add_hline(y=40, line_dash="dash", line_color=GREEN,
                                  annotation_text="≥40 Healthy", annotation_position="top right")
                fig_r40.add_hline(y=20, line_dash="dash", line_color=AMBER,
                                  annotation_text="20 Watch",    annotation_position="top right")
                fig_r40.update_layout(**PLOTLY_LAYOUT, barmode="overlay")
                fig_r40.update_xaxes(**_AX)
                fig_r40.update_yaxes(**_AX, title_text="Score / %")
                st.plotly_chart(fig_r40, use_container_width=True)
            else:
                st.info("Rule of 40 not available (insufficient revenue history).")

            st.markdown("---")

            # ── Section B: SBC Trajectory ─────────────────────────────
            st.markdown("#### SBC Trajectory — Dilution Intensity")
            st.caption(
                "SBC as a % of revenue. Declining trend = management discipline on dilution. "
                "High SBC inflates reported FCF (SBC is added back to CFO but is a real cost)."
            )
            if not sbc_s.empty:
                peak_yr  = sbc_s.idxmax()
                peak_val = float(sbc_s.max())
                fig_sbc = go.Figure()
                fig_sbc.add_trace(go.Bar(
                    x=sbc_s.index, y=sbc_s,
                    marker_color=AMBER, opacity=0.80, name="SBC % Revenue",
                    text=[f"{v:.1f}%" for v in sbc_s], textposition="outside",
                ))
                fig_sbc.add_annotation(
                    x=peak_yr, y=peak_val,
                    text=f"Peak: {peak_val:.1f}% ({peak_yr})",
                    showarrow=True, arrowhead=2,
                    font=dict(size=10, color=AMBER), arrowcolor=AMBER, ax=0, ay=-35,
                )
                fig_sbc.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                fig_sbc.update_xaxes(**_AX)
                fig_sbc.update_yaxes(**_AX, title_text="SBC as % of Revenue")
                st.plotly_chart(fig_sbc, use_container_width=True)
            else:
                st.info("SBC data not available.")

            st.markdown("---")

            # ── Section C: Deferred Revenue ───────────────────────────
            st.markdown("#### Deferred Revenue — Forward Demand Signal")
            st.caption(
                "Deferred revenue = customers pre-paying before revenue is recognized.  "
                "When deferred revenue grows **faster** than reported revenue, bookings are accelerating "
                "ahead of recognized revenue — a bullish leading indicator."
            )
            dr_pct = sd["Deferred Rev % Rev"].dropna()    if "Deferred Rev % Rev"    in sd.columns else pd.Series(dtype=float)
            dr_g   = sd["Deferred Rev Growth %"].dropna() if "Deferred Rev Growth %" in sd.columns else pd.Series(dtype=float)

            if not dr_pct.empty:
                fig_dr = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dr.add_trace(go.Bar(
                    x=dr_pct.index, y=dr_pct,
                    name="Deferred Rev % of Revenue", marker_color=MID_GREEN, opacity=0.75,
                ), secondary_y=False)
                if not dr_g.empty:
                    fig_dr.add_trace(go.Scatter(
                        x=dr_g.index, y=dr_g, name="Deferred Rev Growth %",
                        line=dict(color=GREEN, width=2.5), mode="lines+markers",
                    ), secondary_y=True)
                if not revg.empty:
                    fig_dr.add_trace(go.Scatter(
                        x=revg.index, y=revg, name="Revenue Growth %",
                        line=dict(color=AMBER, width=2, dash="dash"), mode="lines+markers",
                    ), secondary_y=True)
                fig_dr.update_layout(**PLOTLY_LAYOUT)
                fig_dr.update_xaxes(**_AX)
                fig_dr.update_yaxes(title_text="Deferred Rev % of Revenue", secondary_y=False, **_AX)
                fig_dr.update_yaxes(title_text="Growth Rate %", secondary_y=True, **_AX)
                st.plotly_chart(fig_dr, use_container_width=True)
            else:
                st.info("Deferred revenue data not available.")

            st.markdown("---")

            # ── Section D: FCF Quality Bridge ─────────────────────────
            st.markdown("#### FCF Quality Bridge — Latest Year")
            st.caption(
                "Waterfall from Gross Profit → Free Cash Flow. "
                "Shows exactly where margin leakage occurs across R&D, SG&A, and CapEx."
            )
            try:
                is_d = a["is"]
                cf_d = a["cf"]
                yr   = is_d.dropna(subset=["Revenue"]).index[-1]
                gp    = float(is_d.at[yr, "Gross Profit"]) if "Gross Profit" in is_d.columns and pd.notna(is_d.at[yr, "Gross Profit"]) else 0
                rnd   = float(is_d.at[yr, "R&D"])          if "R&D" in is_d.columns         and pd.notna(is_d.at[yr, "R&D"])          else 0
                sga   = float(is_d.at[yr, "SG&A"])         if "SG&A" in is_d.columns        and pd.notna(is_d.at[yr, "SG&A"])         else 0
                capex_wf = float(cf_d.at[yr, "Capital Expenditures"]) if "Capital Expenditures" in cf_d.columns and pd.notna(cf_d.at[yr, "Capital Expenditures"]) else 0
                fcf_wf   = float(cf_d.at[yr, "Free Cash Flow"])       if "Free Cash Flow" in cf_d.columns       and pd.notna(cf_d.at[yr, "Free Cash Flow"])       else 0
                # "Other" absorbs SBC add-backs, working capital changes, etc.
                rnd_abs  = abs(rnd)
                sga_abs  = abs(sga)
                capex_abs = abs(capex_wf)
                other    = fcf_wf - (gp - rnd_abs - sga_abs - capex_abs)

                wf_labels   = ["Gross Profit", "R&D", "SG&A", "CapEx", "Other / WC", "Free Cash Flow"]
                wf_vals     = [gp, -rnd_abs, -sga_abs, -capex_abs, other, fcf_wf]
                wf_measures = ["absolute", "relative", "relative", "relative", "relative", "total"]

                fig_wf = go.Figure(go.Waterfall(
                    orientation="v", measure=wf_measures,
                    x=wf_labels, y=wf_vals,
                    connector=dict(line=dict(color="#C8C0B0", width=1)),
                    increasing=dict(marker_color=LIGHT_GREEN),
                    decreasing=dict(marker_color=AMBER),
                    totals=dict(marker_color=GREEN if fcf_wf >= 0 else "#C0392B"),
                    text=[fmt_usd(v) for v in wf_vals], textposition="outside",
                ))
                fig_wf.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                fig_wf.update_xaxes(**_AX)
                fig_wf.update_yaxes(**_AX, title_text="USD millions")
                st.plotly_chart(fig_wf, use_container_width=True)
            except Exception:
                st.info("FCF bridge could not be computed for this company.")

            st.markdown("---")

            # ── Section E: Efficiency Ratios Table ────────────────────
            st.markdown("#### SaaS Efficiency Metrics — Annual Summary")
            st.caption("Bold = most recent year.  Red CCC = positive cash conversion cycle (unusual weakness for SaaS).")

            # saas_df stores all % values already × 100 — mult=1 for all % cols
            metric_map = {
                "DSO (days)":      ("DSO",               1, "d"),
                "DPO (days)":      ("DPO",               1, "d"),
                "CCC (days)":      ("CCC",               1, "d"),
                "CapEx Intensity": ("Capex Intensity %", 1, "%"),
                "SBC %":           ("SBC %",             1, "%"),
                "FCF Margin %":    ("FCF Margin %",      1, "%"),
                "R&D %":           ("R&D %",             1, "%"),
                "SG&A %":          ("SG&A %",            1, "%"),
                "COGS %":          ("COGS %",            1, "%"),
                "Rule of 40":      ("Rule of 40",        1, ""),
                "Op. Leverage":    ("Operating Leverage", 1, "x"),
            }

            eff_rows = {}
            for label, (col, mult, suffix) in metric_map.items():
                if col in sd.columns:
                    eff_rows[label] = sd[col].dropna() * mult

            if eff_rows:
                eff_df = pd.DataFrame(eff_rows).T

                def _fmt_eff_cell(label, v):
                    if pd.isna(v):
                        return "—"
                    _, _, suffix = metric_map.get(label, ("", 1, ""))
                    if suffix == "d":
                        return f"{v:.0f}d"
                    elif suffix == "x":
                        return f"{v:.2f}x"
                    elif suffix == "%":
                        return f"{v:.1f}%"
                    else:
                        return f"{v:.1f}"

                fmt_eff = pd.DataFrame({
                    col: {lbl: _fmt_eff_cell(lbl, eff_df.at[lbl, col])
                          for lbl in eff_df.index if lbl in eff_df.index}
                    for col in eff_df.columns
                }, index=eff_df.index)

                latest_col = eff_df.columns[-1] if len(eff_df.columns) else None

                def _eff_styler(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    if latest_col in df.columns:
                        styles[latest_col] = "font-weight:600;background-color:#ECE7DA;"
                    if "CCC (days)" in df.index and "CCC" in sd.columns:
                        for col in df.columns:
                            try:
                                raw = float(sd["CCC"].at[col]) if col in sd["CCC"].index else None
                            except Exception:
                                raw = None
                            if raw is not None and not pd.isna(raw) and raw > 0:
                                existing = styles.at["CCC (days)", col]
                                styles.at["CCC (days)", col] = existing + ";color:#B8000B;font-weight:600;"
                    return styles

                st.dataframe(
                    fmt_eff.style.apply(_eff_styler, axis=None),
                    use_container_width=True,
                )
            else:
                st.info("Efficiency metrics not available for this company.")

    # ── Tab 6: Forecast ───────────────────────────────────────────────
    with tabs[6]:
        st.markdown("#### Revenue and NOPLAT Forecast")
        st.caption(
            "McKinsey reinvestment identity: Reinvestment Rate = g / ROIC  |  "
            "FCF = NOPLAT x (1 - g/ROIC)"
        )

        nd_f = a["noplat"]
        rd_f = a["roic"]
        is_f = a["is"]

        base_rev    = float(is_f["Revenue"].dropna().iloc[-1])
        base_ic     = float(rd_f["Total IC"].dropna().iloc[-1]) if rd_f["Total IC"].notna().any() else base_rev * 0.5
        base_noplat = float(nd_f["NOPLAT"].dropna().iloc[-1])   if nd_f["NOPLAT"].notna().any() else 0
        base_nopbt  = float(nd_f["NOPBT Margin"].dropna().iloc[-1])
        base_year   = int(is_f.index[-1])
        rev_s2      = is_f["Revenue"].dropna()
        latest_g    = float(rev_s2.iloc[-1] / rev_s2.iloc[-2] - 1) if len(rev_s2) >= 2 else 0.10

        with st.expander("Forecast Assumptions", expanded=True):
            st.caption("Enter all values as percentages (e.g. type 9 for 9%)")
            fa1, fa2, fa3 = st.columns(3)
            with fa1:
                term_g_pct = st.number_input(
                    "Terminal Growth Rate (%)",
                    min_value=0.0, max_value=8.0, value=3.0, step=0.25,
                    format="%.2f", key="fg_term",
                )
                term_g = term_g_pct / 100

                wacc_pct = st.number_input(
                    "WACC (%)",
                    min_value=5.0, max_value=18.0, value=9.0, step=0.25,
                    format="%.2f", key="fg_wacc",
                )
                wacc = wacc_pct / 100

                tax_pct = st.number_input(
                    "Operating Tax Rate (%)",
                    min_value=10.0, max_value=40.0,
                    value=round(float(a["tax_rate"]) * 100, 1),
                    step=0.5, format="%.2f", key="fg_tax",
                )
                tax_f = tax_pct / 100
            with fa2:
                mg_s_pct = st.number_input(
                    "NOPBT Margin — Year 1 (%)",
                    min_value=0.0, max_value=60.0,
                    value=round(base_nopbt * 100, 1),
                    step=0.5, format="%.2f", key="fg_ms",
                )
                mg_s = mg_s_pct / 100

                mg_e_pct = st.number_input(
                    "NOPBT Margin — Final Year (%)",
                    min_value=0.0, max_value=60.0,
                    value=round(min(base_nopbt * 100 + 5, 60.0), 1),
                    step=0.5, format="%.2f", key="fg_me",
                )
                mg_e = mg_e_pct / 100
            with fa3:
                n_yr = st.selectbox("Forecast Horizon (years)", [5, 7, 10], index=2, key="fg_yrs")
                d_sch = default_growth_schedule(latest_g, term_g, n_yr)
                st.markdown(
                    f"<small style='color:#6B7C6A'>Latest historical revenue growth: "
                    f"<b>{latest_g*100:.1f}%</b></small>",
                    unsafe_allow_html=True,
                )

        st.markdown("**Per-Year Revenue Growth Rates (%)**")
        gr_cols = st.columns(n_yr)
        growth_rates = []
        for i, c in enumerate(gr_cols):
            v_pct = c.number_input(
                f"Y{i+1}",
                min_value=-20.0, max_value=150.0,
                value=round(d_sch[i] * 100, 1),
                step=0.5, format="%.1f",
                key=f"fg_gr_{i}",
            )
            growth_rates.append(float(v_pct) / 100)

        fc = build_forecast(
            base_revenue       = base_rev,
            base_ic            = base_ic,
            base_noplat        = base_noplat,
            growth_rates       = growth_rates,
            nopbt_margin_start = mg_s,
            nopbt_margin_end   = mg_e,
            tax_rate           = tax_f,
            wacc               = wacc,
            terminal_growth    = term_g,
            base_year          = base_year,
        )

        pv_fcf = fc.attrs.get("pv_fcf_sum",   0) or 0
        pv_tv  = fc.attrs.get("pv_tv",         0) or 0
        impl   = fc.attrs.get("implied_value",  0) or 0
        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("PV of FCFs",         fmt_usd(pv_fcf))
        fk2.metric("PV of Terminal Val", fmt_usd(pv_tv))
        fk3.metric("Implied Value",      fmt_usd(impl))
        fk4.metric("TV / Total",         f"{pv_tv/impl*100:.0f}%" if impl else "—")

        st.markdown("#### Projected Financials")
        fc_cols = ["Revenue","Revenue Growth","NOPBT Margin","NOPLAT","ROIC","FCF","PV(FCF)"]
        fc_disp = fc[[c for c in fc_cols if c in fc.columns]].copy()
        pct_fc  = {"Revenue Growth","NOPBT Margin","ROIC"}
        fc_str  = pd.DataFrame({
            col: fc_disp[col].apply(
                lambda x: ("—" if pd.isna(x) else fmt_pct(x) if col in pct_fc else fmt_usd(x))
            )
            for col in fc_disp.columns
        })
        st.dataframe(fc_str, use_container_width=True)

        fig_fc = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Revenue", "NOPLAT and FCF", "ROIC", "PV of FCF"],
        )
        fig_fc.add_trace(go.Bar(x=fc.index, y=fc["Revenue"], marker_color=MID_GREEN, name="Revenue"), row=1, col=1)
        fig_fc.add_trace(go.Scatter(x=fc.index, y=fc["NOPLAT"], line=dict(color=GREEN,  width=2), name="NOPLAT"), row=1, col=2)
        fig_fc.add_trace(go.Scatter(x=fc.index, y=fc["FCF"],    line=dict(color=AMBER, width=2), name="FCF"),    row=1, col=2)
        fig_fc.add_trace(go.Scatter(x=fc.index, y=fc["ROIC"]*100, line=dict(color=GREEN, width=2), name="ROIC %"), row=2, col=1)
        fig_fc.add_trace(go.Bar(x=fc.index, y=fc["PV(FCF)"], marker_color=LIGHT_GREEN, name="PV FCF"), row=2, col=2)
        fig_fc.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=520)
        fig_fc.update_xaxes(**_AX)
        fig_fc.update_yaxes(**_AX)
        st.plotly_chart(fig_fc, use_container_width=True)

    # ── Tab 7: Multiples ──────────────────────────────────────────────
    with tabs[7]:
        st.markdown("#### Market Valuation and Peer Comps")
        st.caption("Live data from Yahoo Finance · SaaS and enterprise software peer universe")

        q2 = st.session_state.quote or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EV / Revenue", fmt_mult(q2.get("ev_revenue")))
        m2.metric("EV / EBITDA",  fmt_mult(q2.get("ev_ebitda")))
        m3.metric("EV / EBIT",    fmt_mult(q2.get("ev_ebit")))
        m4.metric("P / E",        fmt_mult(q2.get("pe")))

        st.markdown("---")

        with st.spinner("Loading peer comps…"):
            comps = _load_comps(a["ticker"])

        if comps.empty:
            st.warning("Could not load peer data. Yahoo Finance may be rate-limiting.")
        else:
            subject = comps.attrs.get("subject", a["ticker"])

            st.markdown("#### Peer Comparison Table")
            disp = comps.copy()
            for col in ["Market Cap", "EV"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda x: f"${x:.1f}B" if pd.notna(x) else "—")
            disp["Price"] = disp["Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
            for col in ["EV/Revenue","EV/EBITDA","EV/EBIT","P/E"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(fmt_mult)

            def _hl(row):
                base = [""] * len(row)
                if row["Ticker"] == subject:
                    return ["background-color:#EAF2EC;font-weight:600"] * len(row)
                return base

            styled = disp.style.apply(_hl, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("#### SaaS Industry Benchmark Ranges")
            bench = []
            mult_key_map = {
                "EV/Revenue": "ev_revenue",
                "EV/EBITDA":  "ev_ebitda",
                "EV/EBIT":    "ev_ebit",
                "P/E":        "pe",
            }
            for metric, vals in INDUSTRY_BENCHMARKS.items():
                subj_v = q2.get(mult_key_map.get(metric, ""))
                bench.append({
                    "Multiple":       metric,
                    "Low":            f"{vals['low']}x",
                    "Median":         f"{vals['median']}x",
                    "High":           f"{vals['high']}x",
                    a["ticker"]:      fmt_mult(subj_v),
                })
            st.dataframe(pd.DataFrame(bench), use_container_width=True, hide_index=True)

            st.markdown("#### EV/Revenue Ranking (Peers)")
            valid = comps[comps["EV/Revenue"].notna()].copy()
            if not valid.empty:
                colors = [GREEN if t == subject else MID_GREEN for t in valid["Ticker"]]
                fig_sp = go.Figure(go.Bar(
                    x=valid["EV/Revenue"],
                    y=valid["Ticker"],
                    orientation="h",
                    marker_color=colors,
                    text=valid["EV/Revenue"].apply(fmt_mult),
                    textposition="outside",
                ))
                fig_sp.update_layout(
                    **PLOTLY_LAYOUT,
                    height=max(300, len(valid) * 38),
                    showlegend=False,
                )
                fig_sp.update_xaxes(title_text="EV / Revenue", **_AX)
                fig_sp.update_yaxes(**_AX)
                st.plotly_chart(fig_sp, use_container_width=True)

    # ── Phase transition ──────────────────────────────────────────────
    st.markdown("---")
    _, adv, _ = st.columns([2, 1, 2])
    with adv:
        if st.button("Continue to Investigation →", use_container_width=True):
            st.session_state.phase = 3
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  PHASE 3: INVESTIGATE  ══════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.phase == 3:
    _phase_bar(3)
    a     = st.session_state.analysis
    if "saas" not in a or "cik" not in a:
        try:
            a = run_analysis(st.session_state.ticker, st.session_state.raw_data)
            st.session_state.analysis = a
        except Exception as _e:
            st.error(f"Could not refresh analysis: {_e}. Click '← New Company' and reload.")
            st.stop()
    goals = st.session_state.goals
    sd    = a.get("saas", pd.DataFrame())
    rd    = a["roic"]
    nd    = a["noplat"]
    is_d  = a["is"]
    cf_d  = a["cf"]

    st.markdown("<div class='phase-pill'>Phase 3 — Investigate</div>", unsafe_allow_html=True)
    st.title("Build Your Analytical Edge")
    st.markdown(
        f"Four structured modules calibrated to **{a['company']}**'s actual financials. "
        "Each module shows you the data, challenges you to interpret it, and offers a framework hint. "
        "Your analysis here anchors the thesis you write in Phase 4."
    )
    st.markdown("---")

    # ── Module 1: Read the Three Statements ──────────────────────────
    st.markdown("### Module 1 — Read the Three Statements")
    st.markdown(
        "*Master the income statement → cash flow bridge. "
        "The gap between Net Income and Free Cash Flow reveals the true quality of earnings.*"
    )

    # Compute the reconciliation numbers
    try:
        latest_yr  = is_d.dropna(subset=["Revenue"]).index[-1]
        ni_val     = float(is_d.at[latest_yr, "Net Income"])            if "Net Income"            in is_d.columns and pd.notna(is_d.at[latest_yr, "Net Income"])            else None
        sbc_val    = float(cf_d.at[latest_yr, "SBC (CFS)"])             if "SBC (CFS)"             in cf_d.columns and pd.notna(cf_d.at[latest_yr, "SBC (CFS)"])             else None
        da_val     = float(cf_d.at[latest_yr, "D&A (CFS)"])             if "D&A (CFS)"             in cf_d.columns and pd.notna(cf_d.at[latest_yr, "D&A (CFS)"])             else None
        cfo_val    = float(cf_d.at[latest_yr, "Operating Cash Flow"])   if "Operating Cash Flow"   in cf_d.columns and pd.notna(cf_d.at[latest_yr, "Operating Cash Flow"])   else None
        capex_val  = float(cf_d.at[latest_yr, "Capital Expenditures"])  if "Capital Expenditures"  in cf_d.columns and pd.notna(cf_d.at[latest_yr, "Capital Expenditures"])  else None
        fcf_val    = float(cf_d.at[latest_yr, "Free Cash Flow"])        if "Free Cash Flow"        in cf_d.columns and pd.notna(cf_d.at[latest_yr, "Free Cash Flow"])        else None
        dr_curr_bs = float(a["bs"].at[latest_yr, "Deferred Revenue (Current)"]) if "Deferred Revenue (Current)" in a["bs"].columns and pd.notna(a["bs"].at[latest_yr, "Deferred Revenue (Current)"]) else None
        ar_bs      = float(a["bs"].at[latest_yr, "Accounts Receivable"])         if "Accounts Receivable"        in a["bs"].columns and pd.notna(a["bs"].at[latest_yr, "Accounts Receivable"])        else None
        m1_data_ok = (ni_val is not None and fcf_val is not None)
    except Exception:
        m1_data_ok = False
        ni_val = fcf_val = cfo_val = None

    with st.expander("📊 Show Data — IS → CF Reconciliation Bridge", expanded=False):
        if m1_data_ok:
            bridge = {
                "Net Income":             ni_val,
                "+ SBC (non-cash)":       sbc_val,
                "+ D&A (non-cash)":       da_val,
                "± Deferred Revenue":     dr_curr_bs,
                "± Accounts Receivable":  ar_bs,
                "= Operating Cash Flow":  cfo_val,
                "− CapEx":                capex_val,
                "= Free Cash Flow":       fcf_val,
            }
            rows = []
            for k, v in bridge.items():
                rows.append({"Line Item": k, str(latest_yr): fmt_usd(v) if v is not None else "—"})
            st.table(pd.DataFrame(rows).set_index("Line Item"))
        else:
            st.info("Reconciliation data not available for this company.")

    with st.expander("💡 Guided Hint — The SaaS Earnings Quality Framework", expanded=False):
        st.markdown("""
For SaaS companies, **FCF is typically higher than Net Income** because:
1. **SBC add-back** — stock-based compensation reduces Net Income but is added back to CFO (it's non-cash, though it dilutes shareholders)
2. **Deferred Revenue growth** — when customers pre-pay, cash is collected before revenue is recognized, boosting CFO above what the income statement shows
3. **D&A** — depreciation reduces net income but doesn't consume cash

**The key question:** Is FCF quality *real* or inflated?
- SBC-heavy FCF = management is paying employees with equity, not cash. Real FCF = FCF minus SBC.
- Deferred revenue-driven FCF = genuinely strong (customers trust the product enough to pre-pay).
- CapEx-light FCF = software leverage in action. High CapEx intensity erodes FCF quality.
        """)

    if ni_val is not None and fcf_val is not None:
        diff = fcf_val - ni_val
        direction = "higher" if diff > 0 else "lower"
        m1_prompt = (
            f"**{a['company']}'s** Net Income was **{md_safe(fmt_usd(ni_val))}** in fiscal {latest_yr}. "
            f"Free Cash Flow was **{md_safe(fmt_usd(fcf_val))}** — "
            f"**{md_safe(fmt_usd(abs(diff)))} {direction}** than Net Income. "
            f"The largest reconciling items are shown in the data panel above. "
            f"Explain *why* FCF is {direction} than Net Income. "
            f"What does this gap reveal about the *quality* of {a['company']}'s earnings model? "
            f"Would you adjust FCF for SBC when assessing true cash generation, and why?"
        )
    else:
        m1_prompt = (
            f"Review the cash flow statement for **{a['company']}** and explain why Operating "
            f"Cash Flow differs from Net Income. What does this reveal about earnings quality?"
        )

    st.markdown(f"> {m1_prompt}")
    m1_ans = st.text_area(
        "Your analysis:", height=130,
        placeholder="Explain the NI → FCF reconciliation and what it tells you about earnings quality…",
        key="p3_q1",
        value=st.session_state.phase3_answers.get("q1", ""),
        label_visibility="collapsed",
    )
    st.session_state.phase3_answers["q1"] = m1_ans

    st.markdown("<hr class='eq-divider'>", unsafe_allow_html=True)

    # ── Module 2: SaaS Unit Economics ────────────────────────────────
    st.markdown("### Module 2 — SaaS Unit Economics Assessment")
    st.markdown(
        "*Rule of 40, SBC trajectory, and deferred revenue dynamics are the three "
        "lenses that separate durable SaaS compounders from growth-at-any-cost stories.*"
    )

    # Extract last 3 years of SaaS metrics
    m2_years = []
    m2_r40_vals, m2_revg_vals, m2_fcfm_vals, m2_sbc_vals, m2_dg_vals = [], [], [], [], []
    if not sd.empty:
        try:
            r40_s  = sd["Rule of 40"].dropna()        if "Rule of 40"        in sd.columns else pd.Series(dtype=float)
            revg_s = sd["Revenue Growth %"].dropna()  if "Revenue Growth %"  in sd.columns else pd.Series(dtype=float)
            fcfm_s = sd["FCF Margin %"].dropna()      if "FCF Margin %"      in sd.columns else pd.Series(dtype=float)
            sbc_s2 = sd["SBC %"].dropna()             if "SBC %"             in sd.columns else pd.Series(dtype=float)
            dg_s   = sd["Deferred Rev Growth %"].dropna() if "Deferred Rev Growth %" in sd.columns else pd.Series(dtype=float)
            common_idx = r40_s.index[-3:] if len(r40_s) >= 3 else r40_s.index
            for yr_i in common_idx:
                m2_years.append(str(yr_i))
                m2_r40_vals.append(f"{r40_s.at[yr_i]:.1f}" if yr_i in r40_s.index else "—")
                m2_revg_vals.append(f"{revg_s.at[yr_i]:.1f}%" if yr_i in revg_s.index else "—")
                m2_fcfm_vals.append(f"{fcfm_s.at[yr_i]:.1f}%" if yr_i in fcfm_s.index else "—")
                m2_sbc_vals.append(f"{sbc_s2.at[yr_i]:.1f}%" if yr_i in sbc_s2.index else "—")
                m2_dg_vals.append(f"{dg_s.at[yr_i]:.1f}%" if yr_i in dg_s.index else "—")
        except Exception:
            pass

    with st.expander("📊 Show Data — SaaS Unit Economics (Last 3 Years)", expanded=False):
        if m2_years:
            m2_table = pd.DataFrame({
                "Metric": ["Rule of 40", "  Revenue Growth %", "  FCF Margin %", "SBC % of Revenue", "Deferred Rev Growth %"],
                **{yr: vals for yr, vals in zip(m2_years, zip(m2_r40_vals, m2_revg_vals, m2_fcfm_vals, m2_sbc_vals, m2_dg_vals))},
            }).set_index("Metric")
            st.dataframe(m2_table, use_container_width=True)
        else:
            st.info("SaaS metrics not computed. Check the SaaS Metrics tab for details.")

    with st.expander("💡 Guided Hint — Decomposing the Rule of 40", expanded=False):
        st.markdown("""
**Rule of 40 = Revenue Growth % + FCF Margin %**

The best SaaS companies optimize both dimensions simultaneously. When diagnosing a change:
- **R40 improves via FCF margin** (not growth) → possible growth deceleration; is margin sustainable or a result of under-investing?
- **R40 improves via growth** (not margin) → classic early-stage SaaS; when does margin follow?
- **R40 declines** → watch whether both levers deteriorate (structural) vs. one temporarily compresses

**SBC lens:** High SBC inflates FCF Margin because SBC is added back to CFO but is a real economic cost. Adjusted FCF Margin = (FCF − SBC) / Revenue. If SBC is declining as a % of revenue, the company is maturing toward cash earnings.

**Deferred revenue lens:** When deferred revenue grows *faster* than revenue → future revenue pipeline is filling. When it grows *slower* → potential bookings headwind ahead of the reported revenue line.
        """)

    if m2_years:
        r40_trend = "improved" if m2_r40_vals and m2_r40_vals[0] != "—" and m2_r40_vals[-1] != "—" and float(m2_r40_vals[-1]) > float(m2_r40_vals[0]) else "declined"
        m2_prompt = (
            f"**{a['company']}'s** Rule of 40 score {r40_trend} over the last "
            f"{len(m2_years)} years (from **{m2_r40_vals[0]}** to **{m2_r40_vals[-1]}**). "
            f"Decompose this shift: did it come primarily from FCF margin expansion or revenue growth? "
            f"What does the deferred revenue growth trend signal about the sustainability of reported revenue? "
            f"Is SBC trending toward normalization — and if not, what is the *true* FCF margin after adjusting for dilution?"
        )
    else:
        m2_prompt = (
            f"Using the SaaS Metrics tab data, assess **{a['company']}'s** Rule of 40 trajectory. "
            f"Decompose whether the trend is driven by FCF margin or growth. "
            f"Comment on SBC trends and deferred revenue as a leading indicator."
        )

    st.markdown(f"> {m2_prompt}")
    m2_ans = st.text_area(
        "Your analysis:", height=130,
        placeholder="Decompose Rule of 40, assess SBC trajectory, interpret deferred revenue dynamics…",
        key="p3_q2",
        value=st.session_state.phase3_answers.get("q2", ""),
        label_visibility="collapsed",
    )
    st.session_state.phase3_answers["q2"] = m2_ans

    st.markdown("<hr class='eq-divider'>", unsafe_allow_html=True)

    # ── Module 3: Revenue Driver Decomposition ───────────────────────
    st.markdown("### Module 3 — Revenue Driver Decomposition")
    st.markdown(
        "*Revenue growth is a headline number. Understanding *what drove it* — "
        "segment mix, pricing, volume, geography — is what separates rigorous analysis from storytelling.*"
    )

    rev_s3 = is_d["Revenue"].dropna()
    has_segs = (
        st.session_state.segments is not None
        and st.session_state.segments.get("tables")
        and len(st.session_state.segments["tables"]) > 0
    )
    op_lev = sd["Operating Leverage"].dropna() if (not sd.empty and "Operating Leverage" in sd.columns) else pd.Series(dtype=float)
    gm_s3  = (is_d["Gross Profit"] / is_d["Revenue"]).dropna() * 100 if "Gross Profit" in is_d.columns else pd.Series(dtype=float)
    rnd_pct3 = sd["R&D %"].dropna()  if (not sd.empty and "R&D %" in sd.columns)  else pd.Series(dtype=float)
    sga_pct3 = sd["SG&A %"].dropna() if (not sd.empty and "SG&A %" in sd.columns) else pd.Series(dtype=float)

    with st.expander("📊 Show Data — Revenue Decomposition Drivers", expanded=False):
        if has_segs:
            st.markdown("**Revenue by Segment / Geography (from 10-K)**")
            for i, df_seg in enumerate(st.session_state.segments["tables"][:2]):
                src = df_seg.attrs.get("source", f"Table {i+1}")
                st.markdown(f"*{src.title()}*")
                fmt_seg3 = df_seg.map(lambda x: fmt_usd(x) if pd.notna(x) else "—")
                st.dataframe(fmt_seg3, use_container_width=True)
        else:
            # Show revenue metrics as a proxy
            revg_for_m3 = sd["Revenue Growth %"].dropna() if (not sd.empty and "Revenue Growth %" in sd.columns) else pd.Series(dtype=float)
            m3_rows = {"Revenue Growth %": revg_for_m3}
            if not gm_s3.empty:
                m3_rows["Gross Margin %"] = gm_s3
            if not rnd_pct3.empty:
                m3_rows["R&D %"] = rnd_pct3
            if not sga_pct3.empty:
                m3_rows["SG&A %"] = sga_pct3
            if not op_lev.empty:
                m3_rows["Operating Leverage (x)"] = op_lev
            if m3_rows:
                m3_df = pd.DataFrame(m3_rows).T
                fmt_m3 = m3_df.copy().astype(str)
                for label in m3_df.index:
                    for col in m3_df.columns:
                        v = m3_df.at[label, col]
                        if pd.isna(v):
                            fmt_m3.at[label, col] = "—"
                        elif "Leverage" in label:
                            fmt_m3.at[label, col] = f"{float(v):.2f}x"
                        else:
                            fmt_m3.at[label, col] = f"{float(v):.1f}%"
                st.dataframe(fmt_m3, use_container_width=True)

    with st.expander("💡 Guided Hint — Operating Leverage and Revenue Quality", expanded=False):
        st.markdown("""
**Operating leverage = EBIT Growth % / Revenue Growth %**
- When > 1: costs are growing *slower* than revenue → margins expanding
- When < 1: costs outpacing revenue → investments are growing faster than returns
- When negative: EBIT compressing while revenue grows → structural cost inflation

**Revenue decomposition framework:**
1. **Volume vs. Price** — Is growth driven by more customers/seats, or higher prices per customer?
2. **Geographic mix** — Emerging markets grow faster but at lower margins; a mix shift can compress blended margins even as revenue rises
3. **Segment mix** — High-margin subscription vs. low-margin professional services; a rising services mix inflates revenue but compresses gross margin
4. **Deferred revenue as a leading indicator** — See Module 2. When deferred revenue growth outpaces revenue growth, future quarters have a revenue tailwind already booked.

**For 10-K segments:** Calculate each segment's *contribution to total revenue growth* as:
(Segment Revenue YoY Change) / (Prior Year Total Revenue)
        """)

    if len(rev_s3) >= 3:
        g_latest = (rev_s3.iloc[-1] / rev_s3.iloc[-2] - 1) * 100
        g_prior  = (rev_s3.iloc[-2] / rev_s3.iloc[-3] - 1) * 100
        g_word   = "accelerated" if g_latest > g_prior + 2 else ("decelerated" if g_latest < g_prior - 2 else "remained stable")
        seg_note = (
            "Geographic and product segment data from the 10-K is shown in the data panel above. "
            if has_segs else
            "Segment data was not parseable from this 10-K, so use the revenue drivers shown above. "
        )
        m3_prompt = (
            f"**{a['company']}'s** revenue growth {g_word} from **{g_prior:.1f}%** to "
            f"**{g_latest:.1f}%** in the most recent year. {seg_note}"
            f"Identify the primary driver of this {'deceleration' if 'decel' in g_word else 'acceleration' if 'accel' in g_word else 'stability'} — "
            f"is it volume, pricing, segment mix, or geographic mix? "
            f"What does the operating leverage profile tell you about cost structure sustainability at this growth rate?"
        )
    else:
        m3_prompt = (
            f"Analyze the revenue growth drivers for **{a['company']}**. "
            f"What is the primary source of revenue growth — volume, pricing, or segment expansion? "
            f"What does the operating leverage profile suggest about cost structure scalability?"
        )

    st.markdown(f"> {m3_prompt}")
    m3_ans = st.text_area(
        "Your analysis:", height=130,
        placeholder="Identify the primary revenue growth driver. Assess operating leverage and cost structure…",
        key="p3_q3",
        value=st.session_state.phase3_answers.get("q3", ""),
        label_visibility="collapsed",
    )
    st.session_state.phase3_answers["q3"] = m3_ans

    st.markdown("<hr class='eq-divider'>", unsafe_allow_html=True)

    # ── Module 4: ROIC Decomposition & Capital Quality ────────────────
    st.markdown("### Module 4 — ROIC Decomposition & Capital Quality")
    st.markdown(
        "*ROIC is the single most important measure of business quality. "
        "Decomposing it into margin and efficiency drivers — and stripping out M&A inflation — "
        "reveals whether returns are sustainable or illusory.*"
    )

    roic_s  = rd["ROIC"].dropna()    * 100 if "ROIC"         in rd.columns else pd.Series(dtype=float)
    nopbt_s = rd["NOPBT Margin"].dropna() * 100 if "NOPBT Margin" in rd.columns else pd.Series(dtype=float)
    ict_s   = rd["IC Turnover"].dropna()      if "IC Turnover"  in rd.columns else pd.Series(dtype=float)
    inc_r   = rd["Incremental ROIC"].dropna() * 100 if "Incremental ROIC" in rd.columns else pd.Series(dtype=float)
    gw_pct4 = sd["Goodwill % of IC"].dropna() if (not sd.empty and "Goodwill % of IC" in sd.columns) else pd.Series(dtype=float)
    qr4     = sd["Quality ROIC"].dropna()     if (not sd.empty and "Quality ROIC"    in sd.columns) else pd.Series(dtype=float)

    with st.expander("📊 Show Data — ROIC Decomposition Table", expanded=False):
        m4_rows = {}
        if not roic_s.empty:
            m4_rows["ROIC %"] = roic_s
        if not nopbt_s.empty:
            m4_rows["NOPBT Margin %"] = nopbt_s
        if not ict_s.empty:
            m4_rows["IC Turnover (x)"] = ict_s
        if not gw_pct4.empty:
            m4_rows["Goodwill % of IC"] = gw_pct4
        if not qr4.empty:
            m4_rows["Quality ROIC %"] = qr4  # already in % from transformer
        if not inc_r.empty:
            m4_rows["Incremental ROIC %"] = inc_r
        if m4_rows:
            m4_df = pd.DataFrame(m4_rows).T
            fmt_m4 = m4_df.copy().astype(str)
            for label in m4_df.index:
                for col in m4_df.columns:
                    v = m4_df.at[label, col]
                    if pd.isna(v):
                        fmt_m4.at[label, col] = "—"
                    elif "Turnover" in label:
                        fmt_m4.at[label, col] = f"{float(v):.2f}x"
                    else:
                        fmt_m4.at[label, col] = f"{float(v):.1f}%"
            st.dataframe(fmt_m4, use_container_width=True)
        else:
            st.info("ROIC decomposition data not available.")

    with st.expander("💡 Guided Hint — The ROIC Decomposition Identity", expanded=False):
        st.markdown("""
**ROIC = NOPBT Margin × IC Turnover × (1 − Tax Rate)**

This decomposition tells you *why* ROIC is changing:
- **NOPBT Margin improving** → pricing power, scale leverage, cost discipline. Sustainable if driven by gross margin, fragile if driven by cutting R&D.
- **IC Turnover improving** → more revenue per dollar of invested capital. Asset efficiency rising. Common in SaaS as revenue scales against relatively fixed infrastructure.
- **Both declining simultaneously** → structural deterioration; likely requires re-investment thesis.

**Quality ROIC** strips out goodwill and intangibles from the denominator:
- **Quality ROIC >> Reported ROIC** → the company has made large M&A investments that sit as goodwill; organic returns are actually stronger than the headline ROIC shows
- **Quality ROIC ≈ Reported ROIC** → asset-light; goodwill is minimal; ROIC is an honest representation

**Incremental ROIC = ΔNOPLAT / ΔIC**
This is the most important forward-looking metric. A company can have mediocre average ROIC but a rising incremental ROIC (new capital deployed at high returns) — that's a value-creation signal. Incremental ROIC above WACC (~9%) = new investment creates shareholder value.
        """)

    if len(roic_s) >= 3:
        roic_chg    = float(roic_s.iloc[-1]) - float(roic_s.iloc[-3])
        roic_word   = "expanded" if roic_chg > 1 else ("compressed" if roic_chg < -1 else "remained stable")
        gw_note     = f"Goodwill represents **{float(gw_pct4.iloc[-1]):.0f}%** of invested capital. " if not gw_pct4.empty else ""
        inc_note    = f"Incremental ROIC in the most recent year was **{float(inc_r.iloc[-1]):.1f}%**. " if not inc_r.empty else ""
        m4_prompt = (
            f"**{a['company']}'s** ROIC {roic_word} from **{float(roic_s.iloc[-3]):.1f}%** to "
            f"**{float(roic_s.iloc[-1]):.1f}%** over 3 years. "
            f"Decompose the change: was it driven by NOPBT Margin expansion (pricing/scale) "
            f"or IC Turnover improvement (asset efficiency)? {gw_note}"
            f"How does this M&A footprint affect your assessment of the company's *true* return on capital? "
            f"{inc_note}"
            f"Does incremental ROIC suggest recent capital allocation decisions are creating or destroying value above WACC?"
        )
    else:
        m4_prompt = (
            f"Decompose **{a['company']}'s** ROIC using the NOPBT Margin × IC Turnover framework. "
            f"Assess whether ROIC improvement is driven by margin or efficiency. "
            f"Comment on how goodwill affects your view of true capital returns, and what Incremental ROIC reveals."
        )

    st.markdown(f"> {m4_prompt}")
    m4_ans = st.text_area(
        "Your analysis:", height=130,
        placeholder="Decompose ROIC drivers. Assess Quality ROIC vs reported. Evaluate incremental ROIC vs WACC…",
        key="p3_q4",
        value=st.session_state.phase3_answers.get("q4", ""),
        label_visibility="collapsed",
    )
    st.session_state.phase3_answers["q4"] = m4_ans

    st.markdown("---")
    answered = sum(
        1 for i in range(1, 5)
        if (st.session_state.get(f"p3_q{i}") or "").strip()
    )
    st.caption(f"{answered} / 4 modules completed.")

    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("← Back to Analysis", use_container_width=True):
            st.session_state.phase = 2
            st.rerun()
    with bc2:
        if st.button("Build My Thesis →", use_container_width=True):
            if answered < 1:
                st.warning("Complete at least one module before proceeding.")
            else:
                st.session_state.phase = 4
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  PHASE 4: THESIS  ═══════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.phase == 4:
    _phase_bar(4)
    a     = st.session_state.analysis
    goals = st.session_state.goals
    q4    = st.session_state.quote or {}

    st.markdown("<div class='phase-pill'>Phase 4 — Thesis Cultivation</div>", unsafe_allow_html=True)
    st.title(f"Investment Thesis: {a['company']}")
    st.markdown(
        "Commit your thinking to writing. A good thesis is falsifiable — "
        "it states not just *why* you believe something, but *what would change your mind*."
    )
    st.markdown("---")

    # Reference panel
    with st.expander("Reference: Your Phase 3 Analysis", expanded=False):
        module_labels = [
            "Module 1 — Three Statements",
            "Module 2 — SaaS Unit Economics",
            "Module 3 — Revenue Decomposition",
            "Module 4 — ROIC & Capital Quality",
        ]
        for i, label in enumerate(module_labels):
            ans = st.session_state.phase3_answers.get(f"q{i+1}", "")
            if ans:
                st.markdown(f"**{label}:** {ans}")

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        rec = st.radio(
            "**Investment Recommendation**",
            ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
            horizontal=True, key="th_rec",
        )
        thesis_body = st.text_area(
            "**Investment Thesis** — What do you believe that the market underappreciates?",
            height=160,
            placeholder=(
                "e.g. The market is treating CRM as a mature-growth software company, "
                "but Agentforce represents a step-change in total addressable market..."
            ),
            key="th_thesis",
        )
        key_drivers = st.text_area(
            "**Key Drivers** — What must be true for this to play out?",
            height=120,
            placeholder=(
                "- Revenue growth reaccelerates to 15%+ as Agentforce gains traction\n"
                "- NOPBT margin expands 300bps over 3 years\n"
                "- FCF conversion stays above 30%"
            ),
            key="th_drivers",
        )
        st.markdown("**Kill Criteria Builder** — What would make you exit?")
        st.caption("Check each criterion and set thresholds. All checked items are saved to your thesis.")
        kc1, kc2 = st.columns(2)
        with kc1:
            use_roic = st.checkbox("ROIC drops below threshold", key="kc_roic")
            kc_roic_val, kc_roic_yrs = 8.0, 2
            if use_roic:
                kc_roic_val = st.number_input("ROIC threshold (%)", 0.0, 20.0, 8.0, 0.5, key="kc_roic_v")
                kc_roic_yrs = st.number_input("For N consecutive years", 1, 5, 2, key="kc_roic_y")
            use_r40 = st.checkbox("Rule of 40 falls below threshold", key="kc_r40")
            kc_r40_val = 20
            if use_r40:
                kc_r40_val = st.number_input("Rule of 40 threshold", 0, 60, 20, key="kc_r40_v")
            use_gm = st.checkbox("Gross Margin compresses by >X ppts", key="kc_gm")
            kc_gm_val = 5.0
            if use_gm:
                kc_gm_val = st.number_input("GM compression threshold (ppts)", 1.0, 20.0, 5.0, 0.5, key="kc_gm_v")
        with kc2:
            use_revg = st.checkbox("Revenue Growth decelerates below threshold", key="kc_revg")
            kc_revg_val = 10.0
            if use_revg:
                kc_revg_val = st.number_input("Revenue Growth threshold (%)", -20.0, 50.0, 10.0, 0.5, key="kc_revg_v")
            use_fcfm = st.checkbox("FCF Margin falls below threshold", key="kc_fcfm")
            kc_fcfm_val = 10.0
            if use_fcfm:
                kc_fcfm_val = st.number_input("FCF Margin threshold (%)", -30.0, 40.0, 10.0, 0.5, key="kc_fcfm_v")
            use_custom = st.checkbox("Custom criterion", key="kc_custom_chk")
            kc_custom_txt = ""
            if use_custom:
                kc_custom_txt = st.text_input(
                    "Custom criterion text",
                    placeholder="e.g. Agentforce ARR growth < 20% in FY2027",
                    key="kc_custom_v",
                )
        # Build kill criteria string for storage
        _kc_parts = []
        if use_roic:
            _kc_parts.append(f"ROIC drops below {kc_roic_val:.1f}% for {kc_roic_yrs} consecutive years")
        if use_r40:
            _kc_parts.append(f"Rule of 40 score falls below {kc_r40_val}")
        if use_revg:
            _kc_parts.append(f"Revenue Growth decelerates below {kc_revg_val:.1f}%")
        if use_fcfm:
            _kc_parts.append(f"FCF Margin falls below {kc_fcfm_val:.1f}%")
        if use_gm:
            _kc_parts.append(f"Gross Margin compresses more than {kc_gm_val:.1f} ppts")
        if use_custom and kc_custom_txt.strip():
            _kc_parts.append(kc_custom_txt.strip())
        kill_criteria = "\n".join(f"- {p}" for p in _kc_parts) if _kc_parts else ""
        risks = st.text_area(
            "**Key Risks**",
            height=100,
            placeholder=(
                "- Macro slowdown compresses enterprise IT budgets\n"
                "- Hyperscaler AI offerings cannibalize agent workflows"
            ),
            key="th_risks",
        )

    with col_r:
        st.markdown("**Price Targets**")
        cur_price = float(q4.get("price") or 0)
        tgt_price = st.number_input(
            "12-Month Price Target (USD)",
            min_value=0.0, value=round(cur_price * 1.25, 2) if cur_price else 0.0,
            step=5.0, format="%.2f", key="th_target",
        )
        horizon = st.selectbox(
            "Investment Horizon",
            ["< 6 months", "6-12 months", "1-2 years", "2-3 years", "3-5 years"],
            index=2, key="th_horizon",
        )
        if cur_price > 0:
            upside = (tgt_price / cur_price - 1) * 100
            st.metric("Current Price",  f"${cur_price:,.2f}")
            st.metric("Target Price",   f"${tgt_price:,.2f}")
            st.metric("Implied Upside", f"{upside:+.1f}%")

        st.markdown("---")
        st.markdown("**Research Goals Addressed**")
        for g in goals:
            st.markdown(f"✓ {g}")

    st.markdown("---")
    _, sc, _ = st.columns([1.5, 1, 1.5])
    with sc:
        save_btn = st.button("Save Thesis", use_container_width=True)

    if save_btn:
        if not thesis_body or len(thesis_body.strip()) < 50:
            st.error("Please write a substantive thesis (at least 50 characters).")
        else:
            try:
                tracker = ThesisTracker()
                tracker.add_thesis(
                    ticker          = a["ticker"],
                    company         = a["company"],
                    recommendation  = rec,
                    current_price   = cur_price,
                    target_price    = tgt_price,
                    time_horizon    = horizon,
                    thesis          = thesis_body,
                    key_assumptions = key_drivers,
                    kill_criteria   = kill_criteria,
                )
                st.session_state.thesis_submitted = True
                st.success(f"Thesis saved for {a['company']}.")
                st.balloons()
            except Exception as e:
                st.error(f"Error saving thesis: {e}")

    # Saved thesis log
    st.markdown("---")
    st.markdown("### Saved Theses")
    try:
        tracker = ThesisTracker()
        all_t   = tracker.get_theses()
        if all_t:
            for t in sorted(all_t, key=lambda x: x.get("created_at",""), reverse=True):
                rec_colors = {
                    "Strong Buy": "#1A5C30", "Buy": "#2D6A4F",
                    "Hold": "#7A6914", "Sell": "#7A1A1A", "Strong Sell": "#4A0000",
                }
                rc = rec_colors.get(t.get("recommendation",""), "#0F2D1E")
                with st.expander(
                    f"{t['ticker']}  {t['company']}  |  {t.get('recommendation','—')}  |  "
                    f"{(t.get('created_at','')[:10])}",
                    expanded=False,
                ):
                    tc1, tc2 = st.columns([3, 1])
                    with tc1:
                        st.markdown(f"**Thesis:** {t.get('thesis','')}")
                        if t.get("key_assumptions"):
                            st.markdown(f"**Drivers:** {t['key_assumptions']}")
                        if t.get("kill_criteria"):
                            st.markdown(f"**Kill Criteria:** {t['kill_criteria']}")
                    with tc2:
                        st.markdown(
                            f"<div style='background:{rc};color:#F5F0E8;border-radius:8px;"
                            f"padding:8px 12px;text-align:center;font-family:Inter;"
                            f"font-weight:600'>{t.get('recommendation','—')}</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric("Target",  f"${t.get('target_price',0):,.2f}")
                        st.metric("Horizon", t.get("time_horizon","—"))
                        if st.button("Delete", key=f"del_{t['id']}"):
                            tracker.delete_thesis(t["id"])
                            st.rerun()
        else:
            st.info("No theses saved yet.")
    except Exception as e:
        st.warning(f"Could not load saved theses: {e}")

    st.markdown("---")
    back_c, _ = st.columns([1, 3])
    with back_c:
        if st.button("← Back to Investigation", use_container_width=True):
            st.session_state.phase = 3
            st.rerun()
