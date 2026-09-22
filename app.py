from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CREDIT RISK ANALYTICS
# Main Model + Application Model
# ============================================================

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MAIN_DIR = DATA_DIR / "Main Model"
APP_DIR = DATA_DIR / "Application Model"

# Fallbacks make the app tolerant of files copied with a suffix such as (1).
def resolve_csv(folder: Path, filename: str) -> Path:
    exact = folder / filename
    if exact.exists():
        return exact
    stem = Path(filename).stem
    candidates = sorted(folder.glob(f"{stem}*.csv"))
    if candidates:
        return candidates[0]
    return exact


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background: #f5f6f8;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #202938;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #293548 100%);
        padding: 40px 44px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, .08);
    }

    .hero .eyebrow {
        color: #aeb9c9;
        text-transform: uppercase;
        letter-spacing: 2.2px;
        font-size: .68rem;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .hero h1 {
        font-family: "Playfair Display", serif;
        font-size: 2.45rem;
        line-height: 1.12;
        margin: 0 0 11px 0;
        color: #ffffff;
    }

    .hero p {
        color: #d3dbe6;
        line-height: 1.75;
        max-width: 900px;
        margin: 0;
        font-size: .96rem;
    }

    .section-title {
        font-family: "Playfair Display", serif;
        color: #111827;
        font-size: 1.55rem;
        margin: 31px 0 7px 0;
    }

    .section-subtitle {
        color: #64748b;
        line-height: 1.65;
        margin-bottom: 17px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 2px 9px rgba(15, 23, 42, .035);
    }

    .card h3 {
        color: #111827;
        margin: 0 0 8px 0;
        font-size: 1.02rem;
    }

    .card p, .card li {
        color: #475569;
        line-height: 1.68;
        font-size: .91rem;
    }

    .kpi {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 96px;
        box-shadow: 0 2px 9px rgba(15, 23, 42, .035);
    }

    .kpi-label {
        color: #64748b;
        font-size: .75rem;
        font-weight: 600;
    }

    .kpi-value {
        color: #111827;
        font-size: 1.48rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .kpi-note {
        color: #94a3b8;
        font-size: .71rem;
        margin-top: 4px;
    }

    .callout {
        border-left: 4px solid #64748b;
        background: #ffffff;
        padding: 15px 18px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        color: #475569;
        line-height: 1.7;
    }

    .callout strong {
        color: #1e293b;
    }

    .insight {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px 17px;
        color: #475569;
        line-height: 1.7;
        margin-top: 12px;
    }

    .tag {
        display: inline-block;
        background: #e2e8f0;
        color: #334155;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: .68rem;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin: 18px 0 24px 0;
    }

    .pipeline-item {
        background: #ffffff;
        border: 1px solid #dbe1e8;
        border-radius: 9px;
        padding: 9px 12px;
        font-size: .77rem;
        color: #334155;
    }

    .arrow {
        color: #94a3b8;
        font-weight: 700;
    }

    .result-card {
        background: linear-gradient(145deg, #111827, #2b3749);
        border-radius: 20px;
        padding: 28px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 35px rgba(15, 23, 42, .12);
    }

    .result-score {
        font-family: "Playfair Display", serif;
        font-size: 3.3rem;
        font-weight: 700;
        line-height: 1;
        margin: 8px 0;
    }

    .result-label {
        font-size: .70rem;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        color: #b9c4d2;
        font-weight: 700;
    }

    .result-pd {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 9px 0 4px;
    }

    .risk-pill {
        display: inline-block;
        border: 1px solid rgba(255,255,255,.25);
        padding: 7px 13px;
        border-radius: 999px;
        margin-top: 10px;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .5px;
    }

    .footer {
        border-top: 1px solid #e2e8f0;
        margin-top: 52px;
        padding: 22px 0 12px;
        color: #94a3b8;
        font-size: .74rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERIC HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_data(folder: Path, filename: str) -> pd.DataFrame:
    return read_csv(str(resolve_csv(folder, filename)))


def col(df: pd.DataFrame, *names):
    if df.empty:
        return None
    lookup = {str(c).strip().upper(): c for c in df.columns}
    for name in names:
        if str(name).strip().upper() in lookup:
            return lookup[str(name).strip().upper()]
    return None


def to_number(value, default=np.nan):
    if pd.isna(value):
        return default
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    text = text.replace("%", "")
    try:
        return float(text)
    except Exception:
        return default


def percent_value(value, digits=2):
    n = to_number(value)
    if pd.isna(n):
        return "-"
    if abs(n) > 1:
        n = n / 100.0
    return f"{n * 100:.{digits}f}%"


def money_idr(value):
    n = to_number(value, 0)
    return f"Rp {n:,.0f}".replace(",", ".")


def kpi(label, value, note=""):
    return f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """


def hero(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def card(title, body):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#475569"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def parse_percent_series(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip(),
        errors="coerce",
    ) / 100.0


def clean_display_df(df):
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).replace("nan", "")
    return out


# ============================================================
# LOAD MAIN MODEL FILES
# ============================================================

raw_df = load_data(DATA_DIR, "application_train.csv")
main_final = load_data(MAIN_DIR, "final_credit_scorecard.csv")
main_portfolio = load_data(MAIN_DIR, "portfolio_summary.csv")
main_risk = load_data(MAIN_DIR, "risk_segment_summary.csv")
main_score = load_data(MAIN_DIR, "score_band_summary.csv")
main_pd_band = load_data(MAIN_DIR, "pd_band_summary.csv")
main_importance = load_data(MAIN_DIR, "variable_importance.csv")
main_calibration = load_data(MAIN_DIR, "calibration_check.csv")

# ============================================================
# LOAD APPLICATION MODEL FILES
# ============================================================

app_binning = load_data(APP_DIR, "application_binning_rules.csv")
app_calibration = load_data(APP_DIR, "application_calibration.csv")
app_decile = load_data(APP_DIR, "application_decile_performance.csv")
app_features = load_data(APP_DIR, "application_feature_selection.csv")
app_lift = load_data(APP_DIR, "application_lift_performance.csv")
app_coeff = load_data(APP_DIR, "application_model_coefficients.csv")
app_metadata = load_data(APP_DIR, "application_model_metadata.csv")
app_performance = load_data(APP_DIR, "application_model_performance.csv")
app_risk_bands = load_data(APP_DIR, "application_risk_bands.csv")
app_concentration = load_data(APP_DIR, "application_risk_concentration.csv")
app_distribution = load_data(APP_DIR, "application_risk_distribution.csv")
app_risk_summary = load_data(APP_DIR, "application_risk_summary.csv")
app_score_config = load_data(APP_DIR, "application_scorecard_config.csv")
app_simulation_result = load_data(APP_DIR, "application_simulation_result.csv")
app_simulation_risk = load_data(APP_DIR, "application_simulation_risk_summary.csv")
app_stability = load_data(APP_DIR, "application_stability.csv")
app_threshold = load_data(APP_DIR, "application_threshold.csv")
app_woe = load_data(APP_DIR, "application_woe_mapping.csv")


# ============================================================
# APPLICATION MODEL ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def build_application_rules(binning_df: pd.DataFrame, woe_df: pd.DataFrame):
    bins = {} if binning_df.empty else binning_df.copy()
    woe = {} if woe_df.empty else woe_df.copy()

    if isinstance(bins, pd.DataFrame):
        bins.columns = [str(c).strip().upper() for c in bins.columns]
        bins["VARIABLE"] = bins["VARIABLE"].ffill()
        bins["BIN_VALUE"] = bins["BIN_VALUE"].ffill()

    if isinstance(woe, pd.DataFrame):
        woe.columns = [str(c).strip().upper() for c in woe.columns]
        woe = woe.dropna(subset=["VARIABLE", "BIN_VALUE", "WOE"]).copy()
        woe["VARIABLE"] = woe["VARIABLE"].astype(str).str.strip().str.upper()
        woe["BIN_VALUE"] = woe["BIN_VALUE"].astype(str).str.strip()
        woe["WOE"] = pd.to_numeric(woe["WOE"], errors="coerce")
        woe = woe.dropna(subset=["WOE"])

    return bins, woe


BIN_RULES, WOE_MAP = build_application_rules(app_binning, app_woe)


def find_woe(variable, bin_value):
    if not isinstance(WOE_MAP, pd.DataFrame) or WOE_MAP.empty:
        return np.nan
    v = str(variable).strip().upper()
    b = str(bin_value).strip()
    rows = WOE_MAP[
        (WOE_MAP["VARIABLE"].astype(str).str.upper() == v)
        & (WOE_MAP["BIN_VALUE"].astype(str).str.strip() == b)
    ]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["WOE"])


def assign_numeric_bin(variable, value):
    if not isinstance(BIN_RULES, pd.DataFrame) or BIN_RULES.empty:
        return None
    rows = BIN_RULES[
        BIN_RULES["VARIABLE"].astype(str).str.upper() == variable.upper()
    ].copy()
    if rows.empty or pd.isna(value):
        return None

    x = float(value)
    rows["LOWER_BOUND"] = pd.to_numeric(rows["LOWER_BOUND"], errors="coerce")
    rows["UPPER_BOUND"] = pd.to_numeric(rows["UPPER_BOUND"], errors="coerce")

    for _, r in rows.iterrows():
        lower = r["LOWER_BOUND"]
        upper = r["UPPER_BOUND"]
        if pd.isna(lower):
            continue
        if pd.isna(upper):
            if x >= lower:
                return str(r["BIN_VALUE"])
        elif x >= lower and x < upper:
            return str(r["BIN_VALUE"])
    return None


def assign_categorical_bin(variable, value):
    if not isinstance(BIN_RULES, pd.DataFrame) or BIN_RULES.empty:
        return None
    rows = BIN_RULES[
        BIN_RULES["VARIABLE"].astype(str).str.upper() == variable.upper()
    ].copy()
    if rows.empty:
        return None

    value_text = str(value).strip()
    allowed = rows["BIN_VALUE"].astype(str).str.strip().tolist()
    if value_text in allowed:
        return value_text

    if variable.upper() == "INCOME_TYPE":
        return "Other / Rare"
    if variable.upper() == "FAMILY_STATUS":
        return "Unknown"
    return None


def calculate_application_model(age, employment, income_type, education, credit_amount, family_status):
    # Binning follows the Application Model SAS output.
    bin_age = assign_numeric_bin("AGE", age)
    bin_employment = assign_numeric_bin("EMPLOYMENT_DURATION", employment)
    bin_income = assign_categorical_bin("INCOME_TYPE", income_type)
    bin_education = assign_categorical_bin("EDUCATION", education)
    bin_family = assign_categorical_bin("FAMILY_STATUS", family_status)
    bin_credit = assign_numeric_bin("CREDIT_AMOUNT", credit_amount)

    woe_age = find_woe("AGE", bin_age)
    woe_employment = find_woe("EMPLOYMENT", bin_employment)
    woe_income = find_woe("INCOME_TYPE", bin_income)
    woe_education = find_woe("EDUCATION", bin_education)
    woe_family = find_woe("FAMILY", bin_family)
    woe_credit = find_woe("CREDIT_AMOUNT", bin_credit)

    coefficients = {}
    if not app_coeff.empty:
        for _, row in app_coeff.iterrows():
            term = str(row.get("TERM", "")).strip().upper()
            coefficients[term] = to_number(row.get("COEFFICIENT"))

    intercept = coefficients.get("INTERCEPT", -2.4344)
    logit = intercept
    components = []

    model_inputs = [
        ("WOE_AGE", woe_age, "AGE"),
        ("WOE_EMPLOYMENT", woe_employment, "EMPLOYMENT_DURATION"),
        ("WOE_INCOME_TYPE", woe_income, "INCOME_TYPE"),
        ("WOE_EDUCATION", woe_education, "EDUCATION"),
        ("WOE_CREDIT_AMOUNT", woe_credit, "CREDIT_AMOUNT"),
        ("WOE_FAMILY", woe_family, "FAMILY_STATUS"),
    ]

    missing = []
    for term, woe_value, source in model_inputs:
        beta = coefficients.get(term, np.nan)
        if pd.isna(woe_value) or pd.isna(beta):
            missing.append(source)
            continue
        contribution = float(beta) * float(woe_value)
        logit += contribution
        components.append({
            "Variable": source,
            "WOE": float(woe_value),
            "Coefficient": float(beta),
            "Contribution": contribution,
        })

    if missing:
        raise ValueError("WOE mapping tidak ditemukan untuk: " + ", ".join(missing))

    pd_model = 1.0 / (1.0 + np.exp(-logit))

    config = {}
    if not app_score_config.empty:
        for _, row in app_score_config.iterrows():
            config[str(row["PARAMETER"]).strip().upper()] = to_number(row["VALUE"])

    factor = config.get("FACTOR", 20 / np.log(2))
    offset = config.get("OFFSET", 600 - factor * np.log(50))
    good_odds = (1 - pd_model) / pd_model if pd_model > 0 else np.inf
    score = offset + factor * np.log(good_odds)

    risk_segment, risk_order = get_application_risk_band(score)

    return {
        "PD": pd_model,
        "LOGIT": logit,
        "SCORE": score,
        "GOOD_ODDS": good_odds,
        "RISK_SEGMENT": risk_segment,
        "RISK_ORDER": risk_order,
        "BIN_AGE": bin_age,
        "BIN_EMPLOYMENT": bin_employment,
        "BIN_INCOME_TYPE": bin_income,
        "BIN_EDUCATION": bin_education,
        "BIN_FAMILY": bin_family,
        "BIN_CREDIT_AMOUNT": bin_credit,
        "WOE_AGE": woe_age,
        "WOE_EMPLOYMENT": woe_employment,
        "WOE_INCOME_TYPE": woe_income,
        "WOE_EDUCATION": woe_education,
        "WOE_FAMILY": woe_family,
        "WOE_CREDIT_AMOUNT": woe_credit,
        "COMPONENTS": pd.DataFrame(components),
    }


def get_application_risk_band(score):
    if not isinstance(app_risk_bands, pd.DataFrame) or app_risk_bands.empty:
        if score <= 549:
            return "HIGH RISK", 1
        if score <= 599:
            return "MEDIUM RISK", 2
        return "LOW RISK", 3

    bands = app_risk_bands.copy()
    bands["MIN_SCORE_NUM"] = pd.to_numeric(bands["MIN_SCORE"], errors="coerce")
    bands["MAX_SCORE_NUM"] = pd.to_numeric(bands["MAX_SCORE"], errors="coerce")
    bands["RISK_ORDER_NUM"] = pd.to_numeric(bands["RISK_ORDER"], errors="coerce")

    for _, r in bands.sort_values("RISK_ORDER_NUM").iterrows():
        lo = r["MIN_SCORE_NUM"]
        hi = r["MAX_SCORE_NUM"]
        if pd.isna(lo) and not pd.isna(hi) and score <= hi:
            return str(r["RISK_SEGMENT"]), int(r["RISK_ORDER_NUM"])
        if not pd.isna(lo) and pd.isna(hi) and score >= lo:
            return str(r["RISK_SEGMENT"]), int(r["RISK_ORDER_NUM"])
        if not pd.isna(lo) and not pd.isna(hi) and lo <= score <= hi:
            return str(r["RISK_SEGMENT"]), int(r["RISK_ORDER_NUM"])

    return "UNKNOWN", 0


# ============================================================
# STATIC PROJECT FACTS FROM THE COMPLETED SAS WORKFLOW
# ============================================================

PHASE1_CHECKS = pd.DataFrame({
    "Check": [
        "Duplicate customer ID",
        "Missing TARGET",
        "Invalid TARGET",
        "Invalid income / credit / annuity",
        "Invalid EXT_SOURCE",
        "Invalid age",
        "Special DAYS_EMPLOYED code",
        "Potential anomaly records",
    ],
    "Result": [0, 0, 0, 0, 0, 55374, 55374, 55647],
    "Status": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "REVIEW", "REVIEW"],
})

PHASE1_MISSING = pd.DataFrame({
    "Feature": ["EXT_SOURCE_1", "EXT_SOURCE_3", "EXT_SOURCE_2", "AMT_GOODS_PRICE", "AMT_ANNUITY", "CNT_FAM_MEMBERS"],
    "Missing": [173378, 60965, 660, 278, 12, 2],
    "Missing %": ["56.38%", "19.83%", "0.21%", "0.09%", "0.00%", "0.00%"],
})

PHASE3_AGE = pd.DataFrame({
    "Age Group": ["20–29", "30–39", "40–49", "50–59", "60+"],
    "Bad Rate": [11.44, 9.59, 7.64, 6.12, 4.92],
    "Index": [1.42, 1.19, .95, .76, .61],
})

PHASE4_IV = pd.DataFrame({
    "Variable": [
        "EXT_SOURCE_AVG", "EMPLOYED_YEARS", "AGE_GROUP", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "AMT_REQ_CREDIT_BUREAU_YEAR",
        "NAME_HOUSING_TYPE", "EXT_SOURCE_MISSING_COUNT", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"
    ],
    "IV": [.5483, .1072, .0802, .0579, .0508, .0217, .0172, .0157, .0109, .0078, .0049],
    "Interpretation": [
        "Suspicious", "Medium", "Weak", "Weak", "Weak", "Weak",
        "Very Weak", "Very Weak", "Very Weak", "Very Weak", "Very Weak"
    ],
})

PHASE5_COEFF = pd.DataFrame({
    "Term": ["Intercept", "WOE_EXT_SOURCE", "WOE_EMPLOYMENT", "WOE_AGE", "WOE_INCOME_TYPE", "WOE_EDUCATION", "WOE_FAMILY"],
    "Coefficient": [-2.4329, -.9249, -.5185, .00143, -.3805, -.7912, -.3608],
    "P-value": ["<0.0001", "<0.0001", "<0.0001", "0.9680", "<0.0001", "<0.0001", "<0.0001"],
})


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:10px 4px 18px;">
        <div style="font-size:.66rem;letter-spacing:2px;color:#94a3b8;font-weight:700;">CREDIT RISK ANALYTICS</div>
        <div style="font-size:1.35rem;font-weight:700;color:white;margin-top:5px;">RiskScore</div>
        <div style="font-size:.75rem;color:#94a3b8;margin-top:4px;">SAS → Scorecard → Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV = {
    "PROJECT": [
        "🏠 Project Overview",
        "📖 Business Understanding",
        "📦 Data Understanding",
    ],
    "MAIN MODEL": [
        "🔍 Phase 1 — Data Quality",
        "🧹 Phase 2 — Data Preparation",
        "📊 Phase 3 — Risk Profiling",
        "📐 Phase 4 — WOE & IV",
        "🤖 Phase 5 — Credit Risk Model",
        "📈 Phase 6 — Model Performance",
        "🎯 Phase 7 — Credit Scorecard",
        "⚠️ Phase 8 — Risk Segmentation",
        "📋 Phase 9 — Portfolio Monitoring",
    ],
    "APPLICATION": [
        "🧮 Credit Risk Simulator",
        "🔬 Scenario Simulator",
        "📊 Application Model",
    ],
    "EXPLORATION": [
        "🔎 Customer Lookup",
        "💡 Key Findings",
    ],
}

page_options = []
for group, pages in NAV.items():
    st.sidebar.markdown(
        f'<div style="font-size:.65rem;letter-spacing:1.4px;color:#64748b;font-weight:700;margin:16px 4px 7px;">{group}</div>',
        unsafe_allow_html=True,
    )
    page_options.extend(pages)

page = st.sidebar.radio("", page_options, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Home Credit Default Risk")
st.sidebar.caption("Main model: Logistic Regression + WOE")
st.sidebar.caption("Application model: user-oriented scorecard")


# ============================================================
# PAGE: PROJECT OVERVIEW
# ============================================================

if page == "🏠 Project Overview":
    hero(
        "End-to-End Credit Risk Project",
        "From Raw Applications to Risk Insight",
        "A portfolio project that combines SAS-based credit risk modeling with an interactive Streamlit application for portfolio analysis and user-oriented risk simulation.",
    )

    if not main_portfolio.empty:
        total_customer = to_number(main_portfolio.iloc[0].get("TOTAL_CUSTOMER"), 92252)
        bad_rate = to_number(main_portfolio.iloc[0].get("ACTUAL_BAD_RATE"), .08072)
        avg_pd = to_number(main_portfolio.iloc[0].get("AVERAGE_PD"), .0807652)
        avg_score = to_number(main_portfolio.iloc[0].get("AVERAGE_SCORE"), 591.23)
    else:
        total_customer, bad_rate, avg_pd, avg_score = 92252, .08072, .0807652, 591.23

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Validation customers", f"{int(total_customer):,}", "Main Model"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Actual bad rate", percent_value(bad_rate), "Validation portfolio"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Average PD", percent_value(avg_pd), "Main Model"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Average score", f"{avg_score:.1f}", "Main Model"), unsafe_allow_html=True)

    section("How the project works")
    st.markdown(
        """
        <div class="pipeline">
            <div class="pipeline-item">Business Problem</div><div class="arrow">→</div>
            <div class="pipeline-item">Data Understanding</div><div class="arrow">→</div>
            <div class="pipeline-item">Data Preparation</div><div class="arrow">→</div>
            <div class="pipeline-item">Risk Profiling</div><div class="arrow">→</div>
            <div class="pipeline-item">WOE & IV</div><div class="arrow">→</div>
            <div class="pipeline-item">Logistic Model</div><div class="arrow">→</div>
            <div class="pipeline-item">Scorecard</div><div class="arrow">→</div>
            <div class="pipeline-item">Monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    with a:
        card("Main Model", "<p>The analytical model developed from the SAS workflow. It is used for model evaluation, scorecard construction, customer-level outputs, and portfolio monitoring.</p>")
    with b:
        card("Application Model", "<p>A user-oriented model that avoids internal external-credit-score inputs and uses six observable features: age, employment duration, income type, education, credit amount, and family status.</p>")
    with c:
        card("Streamlit Layer", "<p>The final interface connects the analytical results with an interactive risk simulator and scenario analysis.</p>")

    section("Project scope")
    scope = pd.DataFrame({
        "Layer": ["Raw Data", "Main Model", "Application Model", "Application"],
        "Purpose": [
            "Understand the original customer/application data",
            "Develop and monitor the analytical credit risk model",
            "Translate model logic into a user-oriented assessment",
            "Allow users to simulate a profile and compare scenarios",
        ],
    })
    st.dataframe(scope, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: BUSINESS UNDERSTANDING
# ============================================================

elif page == "📖 Business Understanding":
    hero(
        "Project Context",
        "Business Understanding",
        "The project starts from a credit-risk business problem and translates it into analytical objectives and an application-oriented solution.",
    )

    section("Background")
    st.markdown(
        """
        <div class="card">
            <p>Credit applications come from customers with different demographic, employment, financial, and family characteristics. A credit risk model can summarize these characteristics into an estimated Probability of Default (PD), which can then be translated into a score and risk segment.</p>
            <p>This project uses the Home Credit Default Risk application dataset to demonstrate an end-to-end analytical workflow: from data quality and feature engineering to WOE/IV, Logistic Regression, scorecard construction, validation, and portfolio monitoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Problem Statement")
    st.markdown(
        """
        <div class="callout">
            <strong>Core analytical question:</strong><br>
            How can historical customer application data be transformed into an interpretable estimate of default risk and translated into a credit score and risk segment?
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Business Questions")
    q1, q2, q3 = st.columns(3)
    with q1:
        card("Risk identification", "<p>Which customer and application characteristics show different observed default patterns?</p>")
    with q2:
        card("Risk measurement", "<p>How well can a statistical model distinguish customers with different observed default outcomes?</p>")
    with q3:
        card("Business translation", "<p>How can predicted PD be converted into an interpretable score and risk segment?</p>")

    section("Project Goals")
    goals = [
        "Understand the structure and quality of the application dataset.",
        "Create analytical features from raw application variables.",
        "Profile historical default patterns before modeling.",
        "Evaluate candidate predictors using WOE and Information Value.",
        "Develop a Logistic Regression model for Probability of Default.",
        "Evaluate discrimination, KS, calibration, and validation stability.",
        "Transform PD into a PDO-based credit score and risk segment.",
        "Build an interactive application model for user-oriented simulation.",
    ]
    st.markdown(
        "<div class='card'><ul>" + "".join(f"<li>{g}</li>" for g in goals) + "</ul></div>",
        unsafe_allow_html=True,
    )

    section("Solution Statement")
    st.markdown(
        """
        <div class="card">
            <p>The solution is an end-to-end <strong>Credit Risk Analytics</strong> workflow. The Main Model provides the analytical and portfolio layer, while the Application Model provides a user-oriented simulation layer using variables that a user can reasonably know without entering internal external-credit-score values.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Important scope")
    st.markdown(
        """
        <div class="callout">
            <strong>This application is an analytical/educational simulation.</strong><br>
            The estimated PD, score, and risk segment are outputs of the models developed in this project. They are not an actual lender approval decision, loan eligibility guarantee, or financial recommendation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: DATA UNDERSTANDING
# ============================================================

elif page == "📦 Data Understanding":
    hero(
        "Source Dataset",
        "Data Understanding",
        "Explore the original application dataset, target definition, feature groups, missingness, and the transition from raw variables to engineered features.",
    )

    if raw_df.empty:
        st.warning("application_train.csv belum ditemukan. Pastikan file berada di Data/application_train.csv.")
    else:
        rows, columns = raw_df.shape
        numeric = raw_df.select_dtypes(include=np.number).shape[1]
        categorical = columns - numeric
        target_col = col(raw_df, "TARGET")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi("Observations", f"{rows:,}", "Original records"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Original variables", f"{columns:,}", "Including TARGET"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("Numeric", f"{numeric:,}", "Detected from CSV"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("Categorical", f"{categorical:,}", "Detected from CSV"), unsafe_allow_html=True)

        section("What does one row represent?")
        st.markdown(
            """
            <div class="card">
                <p>One row represents one customer/application record in the Home Credit Default Risk dataset.</p>
                <p><strong>TARGET = 0</strong> represents a good/non-default outcome, while <strong>TARGET = 1</strong> represents a default outcome.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if target_col:
            target_counts = raw_df[target_col].value_counts(dropna=False)
            good = int(target_counts.get(0, 0))
            bad = int(target_counts.get(1, 0))
            left, right = st.columns([1, 1])
            with left:
                target_table = pd.DataFrame({
                    "Outcome": ["Non-default", "Default"],
                    "Count": [good, bad],
                    "Share": [f"{good/rows:.2%}", f"{bad/rows:.2%}"],
                })
                st.dataframe(target_table, use_container_width=True, hide_index=True)
            with right:
                fig = px.pie(names=["Non-default", "Default"], values=[good, bad], hole=.60, title="Target Distribution")
                st.plotly_chart(chart_layout(fig, 330), use_container_width=True)

        section("Feature Explorer", "Search the original dataset by feature name and inspect type and missingness.")
        search = st.text_input("Search feature", placeholder="EXT_SOURCE, CREDIT, INCOME, EMPLOYED...")
        feature_rows = []
        for c in raw_df.columns:
            if search and search.lower() not in str(c).lower():
                continue
            s = raw_df[c]
            non_null = s.dropna()
            feature_rows.append({
                "Feature": c,
                "Type": str(s.dtype),
                "Missing": int(s.isna().sum()),
                "Missing %": f"{s.isna().mean():.2%}",
                "Example": str(non_null.iloc[0])[:55] if not non_null.empty else "-",
            })
        st.caption(f"{len(feature_rows):,} features shown")
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True, height=430)

        section("Feature groups")
        groups = {
            "Customer & Demographic": ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"],
            "Income & Financial": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
            "Employment": ["DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"],
            "External Credit": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
            "Credit Bureau / Inquiry": ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"],
        }
        group_rows = []
        for group, features in groups.items():
            existing = [f for f in features if f in raw_df.columns]
            group_rows.append({"Feature Group": group, "Features Found": len(existing), "Examples": ", ".join(existing[:6]) + (" ..." if len(existing) > 6 else "")})
        st.dataframe(pd.DataFrame(group_rows), use_container_width=True, hide_index=True)

        section("Raw → Engineered")
        engineered = pd.DataFrame({
            "Raw variable(s)": [
                "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_CREDIT + AMT_INCOME_TOTAL", "AMT_ANNUITY + AMT_INCOME_TOTAL", "AMT_GOODS_PRICE + AMT_CREDIT", "AMT_CREDIT + CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL + CNT_FAM_MEMBERS", "EXT_SOURCE_1 / 2 / 3"
            ],
            "Engineered feature": [
                "AGE_YEARS / AGE_GROUP", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "GOODS_CREDIT_RATIO", "CREDIT_PER_FAMILY", "INCOME_PER_FAMILY", "EXT_SOURCE_AVG / EXT_SOURCE_MISSING_COUNT"
            ],
            "Purpose": [
                "Represent age in years and bands", "Represent employment duration", "Measure credit relative to income", "Measure payment burden relative to income", "Compare goods price and credit", "Approximate credit per family member", "Approximate income per family member", "Combine external scores and capture missingness"
            ],
        })
        st.dataframe(engineered, use_container_width=True, hide_index=True)

        section("Raw data preview")
        st.dataframe(raw_df.head(10), use_container_width=True, height=340)


# ============================================================
# PHASE 1
# ============================================================

elif page == "🔍 Phase 1 — Data Quality":
    hero("SAS Phase 1", "Data Quality Check", "Quality checks were performed before feature engineering and modeling.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Rows", "307,511", "Original records"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Unique customers", "307,511", "Duplicate ID = 0"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Default rate", "8.07%", "TARGET = 1"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Special employment code", "55,374", "Reviewed / transformed later"), unsafe_allow_html=True)

    section("Quality checks")
    st.dataframe(PHASE1_CHECKS, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="callout">
        <strong>Important:</strong> the 55,647 potential anomaly records are not automatically bad records. Most of the review population is related to the special <code>DAYS_EMPLOYED = 365243</code> value, which was handled in Phase 2.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Missing-value hotspots")
    st.dataframe(PHASE1_MISSING, use_container_width=True, hide_index=True)


# ============================================================
# PHASE 2
# ============================================================

elif page == "🧹 Phase 2 — Data Preparation":
    hero("SAS Phase 2", "Data Preparation & Feature Engineering", "The raw variables were cleaned and transformed into analytical features.")

    a, b = st.columns(2)
    with a:
        card("Special DAYS_EMPLOYED value", "<p><code>DAYS_EMPLOYED = 365243</code> was treated as missing rather than a valid employment duration.</p>")
    with b:
        card("Age transformation", "<p><code>AGE_YEARS = ABS(DAYS_BIRTH) / 365.25</code> converts the day-based age into years.</p>")

    section("Engineered variables")
    eng = pd.DataFrame({
        "Feature": ["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "GOODS_CREDIT_RATIO", "CREDIT_PER_FAMILY", "INCOME_PER_FAMILY", "EXT_SOURCE_AVG", "EXT_SOURCE_MISSING_COUNT", "AGE_GROUP", "INCOME_GROUP"],
        "Purpose": ["Age in years", "Employment duration in years", "Credit relative to income", "Annuity relative to income", "Goods price relative to credit", "Credit per family member", "Income per family member", "Average external score", "Number of missing external scores", "Age segmentation", "Income segmentation"],
    })
    st.dataframe(eng, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("Observations", "307,511", "Retained"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Variables", "133", "After engineering"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("DAYS_EMPLOYED special code", "0", "After handling"), unsafe_allow_html=True)

    section("External score missingness")
    ext = pd.DataFrame({"Missing external scores": ["0", "1", "2", "3"], "Customers": [109589, 161013, 36737, 172], "Share": ["35.64%", "52.36%", "11.95%", "0.06%"]})
    st.dataframe(ext, use_container_width=True, hide_index=True)


# ============================================================
# PHASE 3
# ============================================================

elif page == "📊 Phase 3 — Risk Profiling":
    hero("SAS Phase 3", "Risk Profiling", "Descriptive analysis was used to understand historical default patterns before model development.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("Customers", "307,511", "Development portfolio"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Defaults", "24,825", "TARGET = 1"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Baseline bad rate", "8.07%", "Portfolio baseline"), unsafe_allow_html=True)

    section("Observed default rate by age")
    fig = px.bar(PHASE3_AGE, x="Age Group", y="Bad Rate", text="Bad Rate", title="Observed Bad Rate by Age Group")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.dataframe(PHASE3_AGE, use_container_width=True, hide_index=True)

    section("Other observed patterns")
    patterns = pd.DataFrame({
        "Dimension": ["Income Group", "Education", "Housing", "Family Status", "Employment", "EXT_SOURCE_AVG", "Credit Bureau Inquiry"],
        "Observed pattern": [
            "MID 8.55%; LOW 8.20%; HIGH 7.14%",
            "Lower secondary 10.93%; higher education 5.36%",
            "Rented apartment 12.31%; house/apartment 7.80%",
            "Civil marriage 9.94%; married 7.56%",
            "1–3 years 11.07%; 10+ years 5.19%",
            "Lower external-score bands show higher observed bad rates",
            "Missing inquiry information 10.34%; zero inquiries 7.13%",
        ],
    })
    st.dataframe(patterns, use_container_width=True, hide_index=True)

    st.markdown("<div class='callout'><strong>Interpretation:</strong> these are descriptive historical relationships. They do not by themselves establish causality.</div>", unsafe_allow_html=True)


# ============================================================
# PHASE 4
# ============================================================

elif page == "📐 Phase 4 — WOE & IV":
    hero("SAS Phase 4", "Weight of Evidence & Information Value", "WOE transforms model variables into scorecard-friendly representations, while IV summarizes predictive information in the candidate variables.")

    section("Information Value ranking")
    fig = px.bar(PHASE4_IV.sort_values("IV"), x="IV", y="Variable", orientation="h", text="IV", title="Information Value")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(chart_layout(fig, 500), use_container_width=True)
    st.dataframe(PHASE4_IV, use_container_width=True, hide_index=True)

    st.markdown("<div class='callout'><strong>Model candidates:</strong> EXT_SOURCE_AVG, EMPLOYED_YEARS, AGE_GROUP, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, and NAME_FAMILY_STATUS. EXT_SOURCE_AVG has substantially higher IV than the other candidates and therefore deserves stability/leakage review in a production setting.</div>", unsafe_allow_html=True)


# ============================================================
# PHASE 5
# ============================================================

elif page == "🤖 Phase 5 — Credit Risk Model":
    hero("SAS Phase 5", "Credit Risk Model", "A Logistic Regression model was developed using WOE-transformed predictors to estimate Probability of Default.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Validation AUC", "0.7175", "Discrimination"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Gini", "0.4350", "2 × AUC − 1"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("KS", "0.3286", "Separation"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Validation bad rate", "8.07%", "Observed"), unsafe_allow_html=True)

    section("Model variables")
    model_vars = pd.DataFrame({
        "Model feature": ["WOE_EXT_SOURCE", "WOE_EMPLOYMENT", "WOE_AGE", "WOE_INCOME_TYPE", "WOE_EDUCATION", "WOE_FAMILY"],
        "Source variable": ["EXT_SOURCE_AVG", "EMPLOYED_YEARS", "AGE_GROUP", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"],
    })
    st.dataframe(model_vars, use_container_width=True, hide_index=True)

    section("Coefficients")
    st.dataframe(PHASE5_COEFF, use_container_width=True, hide_index=True)
    st.markdown("<div class='callout'><strong>Model note:</strong> WOE_AGE has p-value 0.9680. It was retained in the documented Phase 5/7 specification; a future iteration could review stability, business relevance, and collinearity.</div>", unsafe_allow_html=True)


# ============================================================
# PHASE 6
# ============================================================

elif page == "📈 Phase 6 — Model Performance":
    hero("SAS Phase 6", "Model Performance & Monitoring", "The Main Model was evaluated using discrimination, separation, calibration, and development-versus-validation stability.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("AUC", "0.7175", "Validation"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Gini", "0.4350", "Validation"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("KS", "0.3286", "Validation"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Mean PD", "8.08%", "Validation"), unsafe_allow_html=True)

    section("Development vs validation")
    performance = pd.DataFrame({
        "Metric": ["Bad Rate", "AUC", "Gini", "KS"],
        "Development": ["8.07%", "≈ 0.716", "≈ 0.432", "≈ 0.32"],
        "Validation": ["8.07%", "0.7175", "0.4350", "0.3286"],
    })
    st.dataframe(performance, use_container_width=True, hide_index=True)

    section("Calibration")
    if not main_calibration.empty:
        st.dataframe(clean_display_df(main_calibration), use_container_width=True, hide_index=True)
    else:
        st.dataframe(pd.DataFrame({"Metric": ["Mean predicted PD", "Actual bad rate", "Calibration error"], "Value": ["8.077%", "8.072%", "0.0041%"]}), use_container_width=True, hide_index=True)

    st.markdown("<div class='insight'>Development-versus-validation PSI is best interpreted as sample stability because the validation sample comes from the same historical dataset rather than a future production period.</div>", unsafe_allow_html=True)


# ============================================================
# PHASE 7
# ============================================================

elif page == "🎯 Phase 7 — Credit Scorecard":
    hero("SAS Phase 7", "Credit Scorecard", "Predicted default probability is translated into an interpretable PDO-based credit score.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Base score", "600", "Reference"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Base odds", "20 : 1", "Good : Bad"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("PDO", "20", "Points to double odds"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Factor", "28.8539", "PDO / ln(2)"), unsafe_allow_html=True)

    section("PD → Score")
    pds = pd.DataFrame({"PD": [1, 2, 5, 10, 20, 30], "Credit Score": [646.15, 625.86, 598.52, 576.96, 553.56, 538.01]})
    fig = px.line(pds, x="PD", y="Credit Score", markers=True, title="Credit Score by Probability of Default")
    fig.update_xaxes(title="Probability of Default (%)")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.dataframe(pds.assign(PD=pds["PD"].astype(str) + "%"), use_container_width=True, hide_index=True)

    if not main_score.empty:
        section("Observed score bands")
        st.dataframe(clean_display_df(main_score), use_container_width=True, hide_index=True)

    st.markdown("<div class='callout'><strong>Score interpretation:</strong> within this scorecard, a higher score corresponds to a lower model-predicted probability of default.</div>", unsafe_allow_html=True)


# ============================================================
# PHASE 8
# ============================================================

elif page == "⚠️ Phase 8 — Risk Segmentation":
    hero("SAS Phase 8", "Risk Segmentation", "Customers are grouped into risk bands based on the Main Model's predicted PD.")

    if main_risk.empty:
        st.warning("risk_segment_summary.csv tidak ditemukan.")
    else:
        st.dataframe(clean_display_df(main_risk), use_container_width=True, hide_index=True)

        risk_col = col(main_risk, "RISK_SEGMENT", "RISK_SEGMENT_FINAL")
        bad_col = col(main_risk, "ACTUAL_BAD_RATE", "BAD_RATE")
        count_col = col(main_risk, "CUSTOMER_COUNT", "CUSTOMERS", "APPLICATIONS")
        if risk_col and bad_col:
            plot_df = main_risk.copy()
            plot_df["Bad Rate Numeric"] = parse_percent_series(plot_df[bad_col]) if plot_df[bad_col].dtype == object else pd.to_numeric(plot_df[bad_col], errors="coerce")
            fig = px.bar(plot_df, x=risk_col, y="Bad Rate Numeric", text="Bad Rate Numeric", title="Observed Bad Rate by Risk Segment")
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        if count_col and risk_col:
            plot_df = main_risk.copy()
            plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce")
            fig = px.pie(plot_df, names=risk_col, values=count_col, hole=.58, title="Customer Distribution by Risk Segment")
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)


# ============================================================
# PHASE 9
# ============================================================

elif page == "📋 Phase 9 — Portfolio Monitoring":
    hero("SAS Phase 9", "Final Portfolio Monitoring", "The final Main Model output combines customer-level scorecard results with portfolio, score-band, PD-band, variable-importance, and calibration summaries.")

    if not main_portfolio.empty:
        p = main_portfolio.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi("Customers", f"{int(to_number(p.get('TOTAL_CUSTOMER'), 0)):,}", "Validation portfolio"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Actual bad rate", percent_value(p.get("ACTUAL_BAD_RATE")), "Historical outcome"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("Average PD", percent_value(p.get("AVERAGE_PD")), "Predicted risk"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("Average score", f"{to_number(p.get('AVERAGE_SCORE')):.2f}", "Credit score"), unsafe_allow_html=True)

        section("Portfolio summary")
        st.dataframe(clean_display_df(main_portfolio), use_container_width=True, hide_index=True)

    if not main_risk.empty:
        section("Risk segment summary")
        st.dataframe(clean_display_df(main_risk), use_container_width=True, hide_index=True)

    if not main_score.empty:
        section("Score band summary")
        st.dataframe(clean_display_df(main_score), use_container_width=True, hide_index=True)

    if not main_pd_band.empty:
        section("PD band summary")
        st.dataframe(clean_display_df(main_pd_band), use_container_width=True, hide_index=True)

    if not main_importance.empty:
        section("Variable importance")
        st.dataframe(clean_display_df(main_importance), use_container_width=True, hide_index=True)
        var_col = col(main_importance, "VARIABLE", "FEATURE", "TERM")
        coef_col = col(main_importance, "ABS_COEFFICIENT", "ABSOLUTE_COEFFICIENT", "COEFFICIENT")
        if var_col and coef_col:
            v = main_importance.copy()
            v[coef_col] = pd.to_numeric(v[coef_col], errors="coerce").abs()
            fig = px.bar(v.sort_values(coef_col), x=coef_col, y=var_col, orientation="h", title="Variable Importance")
            st.plotly_chart(chart_layout(fig, 420), use_container_width=True)

    if not main_calibration.empty:
        section("Calibration check")
        st.dataframe(clean_display_df(main_calibration), use_container_width=True, hide_index=True)


# ============================================================
# PAGE: APPLICATION MODEL
# ============================================================

elif page == "📊 Application Model":
    hero("Application Layer", "Application Credit Risk Model", "This model translates the SAS application-oriented model into an interactive Streamlit experience without requiring internal external-credit-score variables.")

    meta = {}
    if not app_metadata.empty:
        for _, r in app_metadata.iterrows():
            meta[str(r["ATTRIBUTE"]).strip()] = r["VALUE"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Model", str(meta.get("MODEL_NAME", "Application Credit Risk Model")), str(meta.get("MODEL_VERSION", ""))), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Validation AUC", f"{to_number(meta.get('VALIDATION_AUC'), .6307):.4f}", "Application Model"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Validation Gini", f"{to_number(meta.get('VALIDATION_GINI'), .2614):.4f}", "Application Model"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Validation KS", str(meta.get("VALIDATION_KS", "19.75%")), "Application Model"), unsafe_allow_html=True)

    section("Why a separate Application Model?")
    st.markdown(
        """
        <div class="card">
            <p>The Main Model includes <code>EXT_SOURCE_AVG</code>, which is an internal-style external credit information feature and is not a realistic field for a general user to know or enter manually.</p>
            <p>The Application Model therefore uses six user-observable variables selected from the application data: <strong>AGE, EMPLOYMENT_DURATION, INCOME_TYPE, EDUCATION, CREDIT_AMOUNT, and FAMILY_STATUS</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Selected variables")
    if not app_features.empty:
        selected = app_features[app_features["SELECTION_STATUS"].astype(str).str.upper().eq("SELECTED")].copy()
        st.dataframe(selected, use_container_width=True, hide_index=True)
    else:
        st.warning("application_feature_selection.csv tidak ditemukan.")

    section("Model performance")
    if not app_performance.empty:
        perf = app_performance.copy()
        st.dataframe(perf, use_container_width=True, hide_index=True)
        if "SAMPLE" in perf.columns and "AUC" in perf.columns:
            fig = px.bar(perf, x="SAMPLE", y="AUC", text="AUC", title="AUC by Sample")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(chart_layout(fig, 330), use_container_width=True)

    section("Scorecard configuration")
    if not app_score_config.empty:
        st.dataframe(app_score_config, use_container_width=True, hide_index=True)

    section("Risk bands")
    if not app_risk_bands.empty:
        st.dataframe(app_risk_bands, use_container_width=True, hide_index=True)

    section("Model coefficients")
    if not app_coeff.empty:
        st.dataframe(app_coeff, use_container_width=True, hide_index=True)

    section("Model limitations")
    st.markdown("<div class='callout'><strong>Documented limitation:</strong> the Application Model metadata states that external credit score is not used. Its intended use is risk estimation / analytical support, not an actual approval decision.</div>", unsafe_allow_html=True)


# ============================================================
# PAGE: CREDIT RISK SIMULATOR
# ============================================================

elif page == "🧮 Credit Risk Simulator":
    hero("Interactive Application", "Credit Risk Simulator", "Enter a profile using information that a typical applicant can reasonably know. The application then applies the SAS Application Model's binning, WOE mapping, coefficients, and scorecard configuration.")

    if app_binning.empty or app_woe.empty or app_coeff.empty or app_score_config.empty:
        st.error("Application Model files belum lengkap. Pastikan folder Data/Application Model berisi binning rules, WOE mapping, coefficients, dan scorecard config.")
    else:
        section("1 · Personal profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=30, step=1)
        with c2:
            family_options = sorted(app_binning.loc[app_binning["VARIABLE"].astype(str).str.upper().eq("FAMILY_STATUS"), "BIN_VALUE"].dropna().astype(str).unique().tolist())
            family_status = st.selectbox("Family status", family_options if family_options else ["Single / not married"])
        with c3:
            education_options = sorted(app_binning.loc[app_binning["VARIABLE"].astype(str).str.upper().eq("EDUCATION"), "BIN_VALUE"].dropna().astype(str).unique().tolist())
            education = st.selectbox("Education", education_options if education_options else ["Higher education"])

        section("2 · Employment")
        c1, c2 = st.columns(2)
        with c1:
            income_options = sorted(app_binning.loc[app_binning["VARIABLE"].astype(str).str.upper().eq("INCOME_TYPE"), "BIN_VALUE"].dropna().astype(str).unique().tolist())
            income_type = st.selectbox("Income type", income_options if income_options else ["Working"])
        with c2:
            employment = st.number_input("Employment duration (years)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)

        section("3 · Credit")
        credit_amount = st.number_input("Credit amount", min_value=1_000.0, max_value=10_000_000.0, value=250_000.0, step=10_000.0, format="%.0f")
        st.caption(f"Input: {money_idr(credit_amount)}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        calculate = st.button("Assess My Risk", type="primary", use_container_width=True)

        if calculate:
            try:
                result = calculate_application_model(age, employment, income_type, education, credit_amount, family_status)
                st.session_state["last_simulation"] = {
                    "inputs": {
                        "age": age,
                        "employment": employment,
                        "income_type": income_type,
                        "education": education,
                        "credit_amount": credit_amount,
                        "family_status": family_status,
                    },
                    "result": result,
                }
            except Exception as exc:
                st.error(f"Model calculation could not be completed: {exc}")

        if "last_simulation" in st.session_state:
            sim = st.session_state["last_simulation"]
            result = sim["result"]
            inputs = sim["inputs"]

            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            section("Your estimated risk profile")

            r1, r2 = st.columns([1, 1.3])
            with r1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Credit score</div>
                        <div class="result-score">{result['SCORE']:.0f}</div>
                        <div class="result-label">Estimated probability of default</div>
                        <div class="result-pd">{result['PD']:.2%}</div>
                        <div class="risk-pill">{result['RISK_SEGMENT']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown("### What the model did")
                st.markdown(
                    f"""
                    <div class="card">
                        <p>The application inputs were converted into the model's predefined bins and WOE values, then passed through the Logistic Regression equation and scorecard formula.</p>
                        <p><strong>Risk segment:</strong> {result['RISK_SEGMENT']}<br>
                        <strong>Good-to-bad odds:</strong> {result['GOOD_ODDS']:.2f} : 1<br>
                        <strong>Logit:</strong> {result['LOGIT']:.4f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            section("Input summary")
            input_table = pd.DataFrame({
                "Input": ["Age", "Employment duration", "Income type", "Education", "Family status", "Credit amount"],
                "Value": [f"{inputs['age']} years", f"{inputs['employment']:.1f} years", inputs["income_type"], inputs["education"], inputs["family_status"], money_idr(inputs["credit_amount"])],
            })
            st.dataframe(input_table, use_container_width=True, hide_index=True)

            section("Model transformation")
            transform_table = pd.DataFrame({
                "Feature": ["AGE", "EMPLOYMENT_DURATION", "INCOME_TYPE", "EDUCATION", "FAMILY_STATUS", "CREDIT_AMOUNT"],
                "Bin": [result["BIN_AGE"], result["BIN_EMPLOYMENT"], result["BIN_INCOME_TYPE"], result["BIN_EDUCATION"], result["BIN_FAMILY"], result["BIN_CREDIT_AMOUNT"]],
                "WOE": [result["WOE_AGE"], result["WOE_EMPLOYMENT"], result["WOE_INCOME_TYPE"], result["WOE_EDUCATION"], result["WOE_FAMILY"], result["WOE_CREDIT_AMOUNT"]],
            })
            st.dataframe(transform_table, use_container_width=True, hide_index=True)

            section("Model contribution")
            components = result["COMPONENTS"].copy()
            if not components.empty:
                components["Absolute contribution"] = components["Contribution"].abs()
                fig = px.bar(
                    components.sort_values("Absolute contribution"),
                    x="Absolute contribution",
                    y="Variable",
                    orientation="h",
                    text="Contribution",
                    title="Absolute contribution to model logit",
                )
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                st.plotly_chart(chart_layout(fig, 400), use_container_width=True)

            st.markdown("<div class='callout'><strong>Important:</strong> this is a model-based risk simulation. It is not a lender approval decision or a guarantee that a credit application will be accepted.</div>", unsafe_allow_html=True)


# ============================================================
# PAGE: SCENARIO SIMULATOR
# ============================================================

elif page == "🔬 Scenario Simulator":
    hero("Interactive Comparison", "Scenario Simulator", "Compare two user profiles and see how the Application Model responds to changes in the inputs.")

    defaults = st.session_state.get("last_simulation", None)
    if defaults:
        base = defaults["inputs"]
    else:
        base = {"age": 30, "employment": 3.0, "income_type": "Working", "education": "Higher education", "credit_amount": 250000.0, "family_status": "Single / not married"}

    income_options = sorted(app_binning.loc[app_binning["VARIABLE"].astype(str).str.upper().eq("INCOME_TYPE"), "BIN_VALUE"].dropna().astype(str).unique().tolist()) if not app_binning.empty else ["Working"]
    education_options = sorted(app_binning.loc[app_binning["VARIABLE"].astype(str).str.upper().eq("EDUCATION"), "BIN_VALUE"].dropna().astype(str).unique().tolist()) if not app_binning.empty else ["Higher education"]
    family_options = sorted(app_binning.loc[app_binning["VARIABLE"].astype(str).str.upper().eq("FAMILY_STATUS"), "BIN_VALUE"].dropna().astype(str).unique().tolist()) if not app_binning.empty else ["Single / not married"]

    section("Current profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age_a = st.number_input("Age", 20, 100, int(base["age"]), key="scenario_age_a")
        employment_a = st.number_input("Employment years", 0.0, 60.0, float(base["employment"]), .5, key="scenario_emp_a")
    with c2:
        income_a = st.selectbox("Income type", income_options, index=income_options.index(base["income_type"]) if base["income_type"] in income_options else 0, key="scenario_income_a")
        education_a = st.selectbox("Education", education_options, index=education_options.index(base["education"]) if base["education"] in education_options else 0, key="scenario_edu_a")
    with c3:
        family_a = st.selectbox("Family status", family_options, index=family_options.index(base["family_status"]) if base["family_status"] in family_options else 0, key="scenario_family_a")
        credit_a = st.number_input("Credit amount", 1000.0, 10000000.0, float(base["credit_amount"]), 10000.0, format="%.0f", key="scenario_credit_a")

    section("Alternative scenario")
    c1, c2, c3 = st.columns(3)
    with c1:
        age_b = st.number_input("Age", 20, 100, int(age_a), key="scenario_age_b")
        employment_b = st.number_input("Employment years", 0.0, 60.0, float(employment_a), .5, key="scenario_emp_b")
    with c2:
        income_b = st.selectbox("Income type", income_options, index=income_options.index(income_a) if income_a in income_options else 0, key="scenario_income_b")
        education_b = st.selectbox("Education", education_options, index=education_options.index(education_a) if education_a in education_options else 0, key="scenario_edu_b")
    with c3:
        family_b = st.selectbox("Family status", family_options, index=family_options.index(family_a) if family_a in family_options else 0, key="scenario_family_b")
        credit_b = st.number_input("Credit amount", 1000.0, 10000000.0, float(credit_a), 10000.0, format="%.0f", key="scenario_credit_b")

    if st.button("Compare Scenarios", type="primary", use_container_width=True):
        try:
            result_a = calculate_application_model(age_a, employment_a, income_a, education_a, credit_a, family_a)
            result_b = calculate_application_model(age_b, employment_b, income_b, education_b, credit_b, family_b)
            st.session_state["scenario_results"] = (result_a, result_b)
        except Exception as exc:
            st.error(f"Scenario calculation could not be completed: {exc}")

    if "scenario_results" in st.session_state:
        ra, rb = st.session_state["scenario_results"]
        section("Comparison")
        comparison = pd.DataFrame({
            "Metric": ["Credit Score", "Estimated PD", "Risk Segment"],
            "Current": [f"{ra['SCORE']:.0f}", f"{ra['PD']:.2%}", ra["RISK_SEGMENT"]],
            "Alternative": [f"{rb['SCORE']:.0f}", f"{rb['PD']:.2%}", rb["RISK_SEGMENT"]],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        d_score = rb["SCORE"] - ra["SCORE"]
        d_pd = rb["PD"] - ra["PD"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(kpi("Score change", f"{d_score:+.0f}", "Alternative − Current"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("PD change", f"{d_pd:+.2%}", "Alternative − Current"), unsafe_allow_html=True)

        st.markdown("<div class='callout'><strong>Scenario interpretation:</strong> the comparison describes how the Application Model responds to changed inputs. It does not establish that changing a real-world characteristic will cause the same outcome outside the model.</div>", unsafe_allow_html=True)


# ============================================================
# PAGE: CUSTOMER LOOKUP
# ============================================================

elif page == "🔎 Customer Lookup":
    hero("Main Model Exploration", "Customer Lookup", "Inspect an individual record from the final Main Model validation output.")

    if main_final.empty:
        st.warning("final_credit_scorecard.csv tidak ditemukan di Data/Main Model.")
    else:
        id_col = col(main_final, "CUSTOMER_ID", "SK_ID_CURR", "ID")
        score_col = col(main_final, "CREDIT_SCORE", "SCORE")
        pd_col = col(main_final, "PREDICTED_PD", "PD", "P_1")
        risk_col = col(main_final, "RISK_SEGMENT_FINAL", "RISK_SEGMENT")
        target_col = col(main_final, "TARGET")

        if id_col is None:
            st.warning("Customer ID column tidak terdeteksi.")
            st.dataframe(main_final.head(20), use_container_width=True)
        else:
            ids = main_final[id_col].dropna().astype(str).tolist()
            selected = st.selectbox("Select customer", ids)
            row_df = main_final[main_final[id_col].astype(str).eq(selected)]
            if not row_df.empty:
                row = row_df.iloc[0]
                score_value = to_number(row[score_col]) if score_col else np.nan
                pd_value = to_number(row[pd_col]) if pd_col else np.nan
                risk_value = row[risk_col] if risk_col else "-"
                target_value = row[target_col] if target_col else np.nan

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(kpi("Customer ID", selected, "Selected record"), unsafe_allow_html=True)
                with c2:
                    st.markdown(kpi("Credit score", f"{score_value:.2f}" if not pd.isna(score_value) else "-", "Main Model"), unsafe_allow_html=True)
                with c3:
                    st.markdown(kpi("Predicted PD", percent_value(pd_value), "Main Model"), unsafe_allow_html=True)
                with c4:
                    st.markdown(kpi("Risk segment", str(risk_value), "Main Model"), unsafe_allow_html=True)

                profile = pd.DataFrame({
                    "Metric": ["Customer ID", "Credit Score", "Predicted PD", "Risk Segment", "Actual Target"],
                    "Value": [selected, f"{score_value:.2f}" if not pd.isna(score_value) else "-", percent_value(pd_value), str(risk_value), int(target_value) if pd.notna(target_value) else "-"],
                })
                st.dataframe(profile, use_container_width=True, hide_index=True)
                with st.expander("View all Main Model fields"):
                    detail = pd.DataFrame({"Field": main_final.columns, "Value": [row[c] for c in main_final.columns]})
                    st.dataframe(detail, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: KEY FINDINGS
# ============================================================

elif page == "💡 Key Findings":
    hero("Portfolio Takeaways", "Key Findings", "The final findings connect the analytical results back to the original business problem and the application layer.")

    findings = [
        ("01", "Portfolio baseline", "The original dataset contains 307,511 application records with an observed default rate of 8.07%."),
        ("02", "Main Model information", "EXT_SOURCE_AVG produced the highest IV among the evaluated Main Model candidate variables at 0.5483."),
        ("03", "Main Model validation", "The Main Model achieved approximately 0.7175 AUC, 0.4350 Gini, and 0.3286 KS on validation."),
        ("04", "Main Model segmentation", "Observed bad rates increase across the documented Main Model risk segments from Very Low Risk to Very High Risk."),
        ("05", "Application Model", "The Application Model uses six user-oriented variables and does not require external credit score input."),
        ("06", "Application Model validation", "The Application Model validation AUC is 0.6307 with Gini 0.2614 and KS 19.75%."),
        ("07", "Application layer", "The Streamlit simulator translates user inputs into bins, WOE, PD, score, and risk segment using the SAS Application Model outputs."),
    ]
    for n, title, body in findings:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:12px;">
                <div style="display:flex;gap:18px;">
                    <div style="font-weight:700;color:#94a3b8;">{n}</div>
                    <div><div style="font-size:1.02rem;font-weight:700;color:#111827;">{title}</div><div style="margin-top:5px;color:#64748b;line-height:1.65;">{body}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='callout'><strong>Portfolio perspective:</strong> the project demonstrates a complete analytical lifecycle from business understanding and raw data through statistical modeling, scorecard construction, portfolio monitoring, and interactive application.</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Credit Risk Analytics · SAS · Python · Pandas · Plotly · Streamlit<br>
        Analytical simulation — not an actual credit approval decision.
    </div>
    """,
    unsafe_allow_html=True,
)
