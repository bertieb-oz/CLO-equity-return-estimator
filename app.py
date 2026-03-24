"""
CLO Equity Monthly Return Estimation Model
==========================================
A Streamlit application that replicates and extends a CLO equity monthly return
estimation model with two output series:
  Series 1: Pure estimated monthly return (model only, no true-up)
  Series 2: Actual-extended monthly return (true-up where Flat Rock available,
            estimated thereafter)

Includes automated parameter optimisation via differential evolution.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution

# =============================================================================
# CONSTANTS AND DEFAULT PARAMETER VALUES
# =============================================================================
DEFAULTS = {
    "date_order": "NewestFirst",
    "effective_income_leverage": 4.25,
    "clo_funding_spread": 0.0175,
    "credit_drag": 0.01,
    "fees_expenses": 0.0125,
    "carry_smoothing_months": 3,
    "loan_price_beta": 3.25,
    "capital_smoothing_months": 2,
    "true_up_method": "Proportional",
}

# Parameter bounds for sliders and optimisation
PARAM_BOUNDS = {
    "effective_income_leverage": (3.5, 5.0, 0.05),
    "clo_funding_spread": (0.014, 0.021, 0.0005),
    "credit_drag": (0.0075, 0.0125, 0.0005),
    "fees_expenses": (0.01, 0.015, 0.0005),
    "carry_smoothing_months": (3, 6, 1),
    "loan_price_beta": (3.0, 3.75, 0.05),
    "capital_smoothing_months": (2, 3, 1),
}

REQUIRED_COLUMNS_COUNT = 5  # A through E

# =============================================================================
# DATA LOADING AND PARSING
# =============================================================================
@st.cache_data
def load_excel(file_bytes: bytes) -> pd.DataFrame:
    """Parse the Inputs sheet from the uploaded Excel file.
    Returns a raw DataFrame with standardised column names.
    Row 1 = headers, Row 2 = format hints (skipped), Rows 3+ = data.
    """
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Inputs",
        header=0,       # row 1 as header
        skiprows=[1],   # skip the format-hint row (row 2)
    )
    if df.shape[1] < REQUIRED_COLUMNS_COUNT:
        raise ValueError(
            f"Expected at least {REQUIRED_COLUMNS_COUNT} columns in the Inputs sheet, "
            f"got {df.shape[1]}."
        )
    # Standardise column names regardless of original headers
    df.columns = list(df.columns[:REQUIRED_COLUMNS_COUNT]) + list(df.columns[REQUIRED_COLUMNS_COUNT:])
    col_map = {
        df.columns[0]: "date",
        df.columns[1]: "sofr",
        df.columns[2]: "loan_spread",
        df.columns[3]: "loan_price",
        df.columns[4]: "flat_rock_return",
    }
    df = df.rename(columns=col_map)
    df = df[["date", "sofr", "loan_spread", "loan_price", "flat_rock_return"]].copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Coerce numeric columns
    for col in ["sofr", "loan_spread", "loan_price", "flat_rock_return"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_data(df_raw: pd.DataFrame, date_order: str) -> pd.DataFrame:
    """Sort data oldest-first, forward-fill continuous inputs, and validate."""
    df = df_raw.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill continuous market data columns (NOT flat_rock_return, which is
    # intentionally sparse — only populated at quarter-ends)
    for col in ["sofr", "loan_spread", "loan_price"]:
        df[col] = df[col].ffill()

    # Drop any rows that still have NaN in essential columns (e.g. leading rows
    # before first data point)
    df = df.dropna(subset=["sofr", "loan_spread", "loan_price"]).reset_index(drop=True)

    return df


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================
def backwards_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Backwards-looking simple moving average with expanding window at boundaries.
    For each row t (oldest-first), average rows [t-(n-1) ... t].
    At boundaries where fewer than n rows exist, use all available rows.
    """
    result = series.rolling(window=window, min_periods=1).mean()
    return result


def run_model(df: pd.DataFrame, params: dict) -> dict:
    """
    Core calculation engine.

    df: DataFrame with columns [date, sofr, loan_spread, loan_price, flat_rock_return]
        sorted oldest-first, date as pd.Timestamp
    params: dict with keys matching DEFAULTS

    Returns: dict with keys:
        'monthly': DataFrame with all intermediate and final columns
        'series1': Series (estimated monthly return)
        'series2': Series (actual-extended monthly return)
        'quarterly': DataFrame with quarterly summary
        'verification_errors': list of (quarter_label, discrepancy) tuples
    """
    d = df.copy()
    n = len(d)

    eil = params["effective_income_leverage"]
    cfs = params["clo_funding_spread"]
    cd = params["credit_drag"]
    fe = params["fees_expenses"]
    csm = int(round(params["carry_smoothing_months"]))
    lpb = params["loan_price_beta"]
    ksm = int(round(params["capital_smoothing_months"]))
    true_up_method = params["true_up_method"]

    # --- Step 1: Loan price monthly return ---
    d["loan_price_return"] = d["loan_price"].pct_change()
    d.loc[d.index[0], "loan_price_return"] = 0.0

    # --- Step 2: Annual yield and spread calculations ---
    d["loan_yield_annual"] = d["sofr"] + d["loan_spread"]
    d["net_excess_spread_annual"] = d["loan_spread"] - cfs - cd - fe
    d["equity_carry_gross_annual"] = d["sofr"] + (d["net_excess_spread_annual"] * eil)
    d["equity_carry_net_annual"] = d["equity_carry_gross_annual"] - fe

    # --- Step 3: Monthly carry ---
    d["carry_monthly"] = d["equity_carry_net_annual"] / 12.0

    # --- Step 4: Carry smoothing ---
    d["carry_smoothed"] = backwards_moving_average(d["carry_monthly"], csm)

    # --- Step 5: Equity capital return ---
    d["equity_capital_return_monthly"] = d["loan_price_return"] * lpb

    # --- Step 6: Capital smoothing ---
    d["capital_smoothed"] = backwards_moving_average(d["equity_capital_return_monthly"], ksm)

    # --- Step 7: Estimated total return (Series 1) ---
    d["estimated_return"] = d["carry_smoothed"] + d["capital_smoothed"]

    # --- Step 8: Quarter identification ---
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    d["is_quarter_end"] = d["month"].isin([3, 6, 9, 12])
    # Quarter label: e.g. Q1-2021
    d["quarter_num"] = ((d["month"] - 1) // 3) + 1
    d["quarter_label"] = "Q" + d["quarter_num"].astype(str) + "-" + d["year"].astype(str)

    # Assign each row to a quarter group key (year, quarter_num)
    d["qkey"] = list(zip(d["year"], d["quarter_num"]))

    # --- Step 9-11: True-up and Series 2 ---
    d["actual_extended_return"] = d["estimated_return"].copy()

    # Build a lookup: qkey -> flat_rock_return (from quarter-end rows)
    qe_mask = d["is_quarter_end"] & d["flat_rock_return"].notna()
    fr_lookup = dict(zip(d.loc[qe_mask, "qkey"], d.loc[qe_mask, "flat_rock_return"]))

    # For each quarter with a Flat Rock return, compute provisional and apply true-up
    d["provisional_quarter_return"] = np.nan
    d["true_up_factor"] = np.nan

    verification_errors = []

    for qkey, fr_val in fr_lookup.items():
        q_mask = d["qkey"] == qkey
        q_idx = d.index[q_mask]

        if len(q_idx) == 0:
            continue

        # Provisional quarterly return = compound of estimated monthly returns in this quarter
        est_returns = d.loc[q_idx, "estimated_return"].values
        provisional = np.prod(1 + est_returns) - 1

        # Store provisional at quarter-end
        qe_idx = q_idx[d.loc[q_idx, "is_quarter_end"]]
        if len(qe_idx) > 0:
            d.loc[qe_idx[0], "provisional_quarter_return"] = provisional

        # True-up factor
        # For proportional method, we need scale^n * prod(1+r_i) = (1+fr_val)
        # so scale = ((1+fr_val) / (1+provisional))^(1/n)
        n_months_in_q = len(q_idx)
        if true_up_method == "Proportional" and n_months_in_q > 0:
            raw_ratio = (1 + fr_val) / (1 + provisional) if (1 + provisional) != 0 else 1.0
            scale = raw_ratio ** (1.0 / n_months_in_q)
        else:
            scale = (1 + fr_val) / (1 + provisional) if (1 + provisional) != 0 else 1.0

        if len(qe_idx) > 0:
            d.loc[qe_idx[0], "true_up_factor"] = scale

        # Apply true-up to get actual-extended returns
        if true_up_method == "Proportional":
            # Each month's gross return is scaled by the same factor (nth-root scaling)
            for idx in q_idx:
                d.loc[idx, "actual_extended_return"] = (1 + d.loc[idx, "estimated_return"]) * scale - 1
        else:
            # EndLoaded: first months keep estimated, last month absorbs residual
            if len(q_idx) >= 2:
                # All months except last keep estimated
                for idx in q_idx[:-1]:
                    d.loc[idx, "actual_extended_return"] = d.loc[idx, "estimated_return"]
                # Last month absorbs residual
                prior_compound = np.prod(1 + d.loc[q_idx[:-1], "estimated_return"].values)
                last_idx = q_idx[-1]
                d.loc[last_idx, "actual_extended_return"] = (1 + fr_val) / prior_compound - 1
            elif len(q_idx) == 1:
                # Single month quarter (partial data): just set to flat rock
                d.loc[q_idx[0], "actual_extended_return"] = fr_val

        # Verification: compound of actual-extended returns should equal Flat Rock
        actual_compound = np.prod(1 + d.loc[q_idx, "actual_extended_return"].values) - 1
        discrepancy = abs(actual_compound - fr_val)
        if discrepancy > 1e-6:
            qlabel = d.loc[q_idx[-1], "quarter_label"] if len(q_idx) > 0 else str(qkey)
            verification_errors.append((qlabel, discrepancy))

    # --- Detect last Flat Rock release date ---
    fr_dates = d.loc[d["flat_rock_return"].notna(), "date"]
    last_fr_date = fr_dates.max() if len(fr_dates) > 0 else None

    # --- Series 3: Flat Rock actual returns spread to monthly (geometric) ---
    # For each quarter with a Flat Rock return, compute the equal monthly return
    # such that compounding n months gives the quarterly return:
    #   (1 + r_monthly)^n = (1 + r_quarterly)  =>  r_monthly = (1 + r_quarterly)^(1/n) - 1
    # Months in quarters without Flat Rock data are left as NaN.
    d["flat_rock_monthly"] = np.nan
    for qkey, fr_val in fr_lookup.items():
        q_mask = d["qkey"] == qkey
        q_idx = d.index[q_mask]
        n_m = len(q_idx)
        if n_m > 0:
            monthly_equiv = (1 + fr_val) ** (1.0 / n_m) - 1
            d.loc[q_idx, "flat_rock_monthly"] = monthly_equiv

    # --- Build quarterly summary ---
    quarterly_rows = []
    for qkey in d["qkey"].unique():
        q_mask = d["qkey"] == qkey
        q_data = d.loc[q_mask]
        qlabel = q_data["quarter_label"].iloc[0]

        est_qtr = np.prod(1 + q_data["estimated_return"].values) - 1
        act_qtr = np.prod(1 + q_data["actual_extended_return"].values) - 1

        # Flat Rock return for this quarter
        fr_val = fr_lookup.get(qkey, np.nan)

        # Estimation error
        est_error = (est_qtr - fr_val) if not np.isnan(fr_val) else np.nan

        # Status
        if not np.isnan(fr_val):
            status = "Official"
        elif last_fr_date is not None and q_data["date"].max() > last_fr_date:
            status = "Post-Release Estimate"
        else:
            status = "Estimated"

        quarterly_rows.append({
            "Quarter": qlabel,
            "Flat Rock Return (%)": fr_val * 100 if not np.isnan(fr_val) else np.nan,
            "Estimated Quarterly Return (%)": est_qtr * 100,
            "Actual-Extended Quarterly Return (%)": act_qtr * 100,
            "Estimation Error (%)": est_error * 100 if not np.isnan(est_error) else np.nan,
            "Status": status,
        })

    quarterly_df = pd.DataFrame(quarterly_rows)

    # Annualised returns
    n_months = len(d)
    total_s1 = np.prod(1 + d["estimated_return"].values) - 1
    total_s2 = np.prod(1 + d["actual_extended_return"].values) - 1
    ann_s1 = (1 + total_s1) ** (12.0 / n_months) - 1 if n_months > 0 else 0
    ann_s2 = (1 + total_s2) ** (12.0 / n_months) - 1 if n_months > 0 else 0

    # Flat Rock annualised — only over months that have actual data
    fr_valid = d["flat_rock_monthly"].dropna()
    if len(fr_valid) > 0:
        total_fr = np.prod(1 + fr_valid.values) - 1
        ann_fr = (1 + total_fr) ** (12.0 / len(fr_valid)) - 1
    else:
        ann_fr = np.nan

    return {
        "monthly": d,
        "series1": d["estimated_return"],
        "series2": d["actual_extended_return"],
        "quarterly": quarterly_df,
        "verification_errors": verification_errors,
        "last_fr_date": last_fr_date,
        "annualised_s1": ann_s1,
        "annualised_s2": ann_s2,
        "annualised_fr": ann_fr,
    }


# =============================================================================
# OPTIMISATION FUNCTION
# =============================================================================
def objective_function(x, df, true_up_method):
    """Compute SSE between provisional quarterly returns and Flat Rock returns.
    x = [eil, cfs, cd, fe, csm, lpb, ksm]
    """
    params = {
        "effective_income_leverage": x[0],
        "clo_funding_spread": x[1],
        "credit_drag": x[2],
        "fees_expenses": x[3],
        "carry_smoothing_months": int(round(x[4])),
        "loan_price_beta": x[5],
        "capital_smoothing_months": int(round(x[6])),
        "true_up_method": true_up_method,
    }
    result = run_model(df, params)
    d = result["monthly"]

    # Compute SSE over quarters with known Flat Rock returns
    qe_mask = d["is_quarter_end"] & d["flat_rock_return"].notna()
    sse = 0.0
    for _, row in d.loc[qe_mask].iterrows():
        qkey = row["qkey"]
        q_mask = d["qkey"] == qkey
        est_returns = d.loc[q_mask, "estimated_return"].values
        provisional = np.prod(1 + est_returns) - 1
        fr = row["flat_rock_return"]
        sse += (provisional - fr) ** 2
    return sse


def run_optimisation(df, current_params):
    """Run differential evolution to find optimal parameters."""
    bounds = [
        PARAM_BOUNDS["effective_income_leverage"][:2],
        PARAM_BOUNDS["clo_funding_spread"][:2],
        PARAM_BOUNDS["credit_drag"][:2],
        PARAM_BOUNDS["fees_expenses"][:2],
        PARAM_BOUNDS["carry_smoothing_months"][:2],
        PARAM_BOUNDS["loan_price_beta"][:2],
        PARAM_BOUNDS["capital_smoothing_months"][:2],
    ]

    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(df, current_params["true_up_method"]),
        maxiter=100,
        popsize=10,
        tol=0.01,
        seed=42,
    )

    opt_params = {
        "effective_income_leverage": round(result.x[0] / 0.05) * 0.05,
        "clo_funding_spread": round(result.x[1] / 0.0005) * 0.0005,
        "credit_drag": round(result.x[2] / 0.0005) * 0.0005,
        "fees_expenses": round(result.x[3] / 0.0005) * 0.0005,
        "carry_smoothing_months": int(round(result.x[4])),
        "loan_price_beta": round(result.x[5] / 0.05) * 0.05,
        "capital_smoothing_months": int(round(result.x[6])),
    }

    # Compute current RMSE for comparison
    current_sse = objective_function(
        [
            current_params["effective_income_leverage"],
            current_params["clo_funding_spread"],
            current_params["credit_drag"],
            current_params["fees_expenses"],
            current_params["carry_smoothing_months"],
            current_params["loan_price_beta"],
            current_params["capital_smoothing_months"],
        ],
        df,
        current_params["true_up_method"],
    )

    # Count quarters with Flat Rock data
    d_temp = df.copy()
    d_temp["month"] = d_temp["date"].dt.month
    n_quarters = d_temp.loc[d_temp["month"].isin([3, 6, 9, 12]) & d_temp["flat_rock_return"].notna()].shape[0]
    n_quarters = max(n_quarters, 1)

    current_rmse = np.sqrt(current_sse / n_quarters)
    opt_rmse = np.sqrt(result.fun / n_quarters)

    return opt_params, current_rmse, opt_rmse


# =============================================================================
# EXCEL EXPORT HELPERS
# =============================================================================
def to_excel_monthly(df_display: pd.DataFrame) -> bytes:
    """Export monthly table to Excel with formatting."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_display.to_excel(writer, sheet_name="Monthly Returns", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Monthly Returns"]

        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#1F4E79", "font_color": "white",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        even_fmt = workbook.add_format({"bg_color": "#D6E4F0", "border": 1})
        odd_fmt = workbook.add_format({"border": 1})

        for col_num, col_name in enumerate(df_display.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
            worksheet.set_column(col_num, col_num, max(len(str(col_name)) + 2, 14))

        for row_num in range(1, len(df_display) + 1):
            fmt = even_fmt if row_num % 2 == 0 else odd_fmt
            for col_num in range(len(df_display.columns)):
                val = df_display.iloc[row_num - 1, col_num]
                if pd.isna(val):
                    worksheet.write_blank(row_num, col_num, None, fmt)
                else:
                    worksheet.write(row_num, col_num, val, fmt)

        worksheet.freeze_panes(1, 0)

    return output.getvalue()


def to_excel_quarterly(df_display: pd.DataFrame) -> bytes:
    """Export quarterly table to Excel with formatting."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_display.to_excel(writer, sheet_name="Quarterly Summary", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Quarterly Summary"]

        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#1F4E79", "font_color": "white",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        even_fmt = workbook.add_format({"bg_color": "#D6E4F0", "border": 1})
        odd_fmt = workbook.add_format({"border": 1})

        for col_num, col_name in enumerate(df_display.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
            worksheet.set_column(col_num, col_num, max(len(str(col_name)) + 2, 14))

        for row_num in range(1, len(df_display) + 1):
            fmt = even_fmt if row_num % 2 == 0 else odd_fmt
            for col_num in range(len(df_display.columns)):
                val = df_display.iloc[row_num - 1, col_num]
                if pd.isna(val):
                    worksheet.write_blank(row_num, col_num, None, fmt)
                else:
                    worksheet.write(row_num, col_num, val, fmt)

        worksheet.freeze_panes(1, 0)

    return output.getvalue()


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="CLO Equity Model", page_icon="📊", layout="wide")

st.title("📊 CLO Equity Monthly Return Estimation Model")
st.caption("Upload your Inputs Excel file, configure parameters, and view estimated vs actual-extended CLO equity returns.")

# --- Session state initialisation ---
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "opt_results" not in st.session_state:
    st.session_state.opt_results = None
if "apply_opt" not in st.session_state:
    st.session_state.apply_opt = False

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # File uploader
    uploaded_file = st.file_uploader("Upload Inputs Excel (.xlsx)", type=["xlsx"])
    if uploaded_file is not None:
        st.session_state.uploaded_bytes = uploaded_file.getvalue()

    st.divider()
    st.subheader("Model Parameters")

    # If optimised params should be applied, use them as defaults
    opt = st.session_state.opt_results
    use_opt = st.session_state.apply_opt and opt is not None

    date_order = st.selectbox(
        "Date order in file",
        ["NewestFirst", "OldestFirst"],
        index=0,
    )

    # Start date filter
    use_start_filter = st.checkbox("Override analysis start date", value=False)
    if use_start_filter:
        start_date_override = st.date_input(
            "Analysis start date",
            value=datetime.date(2021, 1, 1),
            help="Trim early data to remove noisy initialisation period.",
        )
    else:
        start_date_override = None

    effective_income_leverage = st.slider(
        "Effective income leverage",
        min_value=3.5, max_value=5.0, step=0.05,
        value=opt["params"]["effective_income_leverage"] if use_opt else DEFAULTS["effective_income_leverage"],
        key="eil_slider",
    )

    clo_funding_spread_pct = st.slider(
        "CLO funding spread (%)",
        min_value=1.40, max_value=2.10, step=0.05,
        value=(opt["params"]["clo_funding_spread"] * 100) if use_opt else (DEFAULTS["clo_funding_spread"] * 100),
        key="cfs_slider",
    )
    clo_funding_spread = clo_funding_spread_pct / 100.0

    credit_drag_pct = st.slider(
        "Credit drag (%)",
        min_value=0.75, max_value=1.25, step=0.05,
        value=(opt["params"]["credit_drag"] * 100) if use_opt else (DEFAULTS["credit_drag"] * 100),
        key="cd_slider",
    )
    credit_drag = credit_drag_pct / 100.0

    fees_expenses_pct = st.slider(
        "Fees & expenses (%)",
        min_value=1.00, max_value=1.50, step=0.05,
        value=(opt["params"]["fees_expenses"] * 100) if use_opt else (DEFAULTS["fees_expenses"] * 100),
        key="fe_slider",
    )
    fees_expenses = fees_expenses_pct / 100.0

    carry_smoothing_months = st.slider(
        "Carry smoothing months",
        min_value=3, max_value=6, step=1,
        value=opt["params"]["carry_smoothing_months"] if use_opt else DEFAULTS["carry_smoothing_months"],
        key="csm_slider",
    )

    loan_price_beta = st.slider(
        "Loan price beta",
        min_value=3.0, max_value=3.75, step=0.05,
        value=opt["params"]["loan_price_beta"] if use_opt else DEFAULTS["loan_price_beta"],
        key="lpb_slider",
    )

    capital_smoothing_months = st.slider(
        "Capital smoothing months",
        min_value=2, max_value=3, step=1,
        value=opt["params"]["capital_smoothing_months"] if use_opt else DEFAULTS["capital_smoothing_months"],
        key="ksm_slider",
    )

    true_up_method = st.selectbox(
        "True-up method",
        ["Proportional", "EndLoaded"],
        index=0,
    )

    # Clear the apply flag after rendering (so it only takes effect once)
    if st.session_state.apply_opt:
        st.session_state.apply_opt = False

    st.divider()

    # Optimise button
    if st.session_state.uploaded_bytes is not None:
        if st.button("🔍 Optimise Parameters", use_container_width=True):
            with st.spinner("Running optimisation (may take 15-30 seconds)..."):
                try:
                    df_raw = load_excel(st.session_state.uploaded_bytes)
                    df_sorted = prepare_data(df_raw, date_order)
                    current_p = {
                        "effective_income_leverage": effective_income_leverage,
                        "clo_funding_spread": clo_funding_spread,
                        "credit_drag": credit_drag,
                        "fees_expenses": fees_expenses,
                        "carry_smoothing_months": carry_smoothing_months,
                        "loan_price_beta": loan_price_beta,
                        "capital_smoothing_months": capital_smoothing_months,
                        "true_up_method": true_up_method,
                    }
                    opt_params, cur_rmse, opt_rmse = run_optimisation(df_sorted, current_p)
                    st.session_state.opt_results = {
                        "params": opt_params,
                        "current_rmse": cur_rmse,
                        "opt_rmse": opt_rmse,
                    }
                except Exception as e:
                    st.error(f"Optimisation failed: {e}")

        # Show optimisation results
        if st.session_state.opt_results is not None:
            opt = st.session_state.opt_results
            st.divider()
            st.subheader("📈 Optimisation Results")
            st.metric("Current RMSE", f"{opt['current_rmse']*100:.4f}%")
            st.metric("Optimised RMSE", f"{opt['opt_rmse']*100:.4f}%",
                       delta=f"{(opt['opt_rmse'] - opt['current_rmse'])*100:.4f}%")

            st.markdown("**Suggested parameters:**")
            opt_p = opt["params"]
            st.text(
                f"  Income leverage: {opt_p['effective_income_leverage']:.2f}\n"
                f"  Funding spread:  {opt_p['clo_funding_spread']*100:.2f}%\n"
                f"  Credit drag:     {opt_p['credit_drag']*100:.2f}%\n"
                f"  Fees & expenses: {opt_p['fees_expenses']*100:.2f}%\n"
                f"  Carry smooth:    {opt_p['carry_smoothing_months']}\n"
                f"  Price beta:      {opt_p['loan_price_beta']:.2f}\n"
                f"  Capital smooth:  {opt_p['capital_smoothing_months']}"
            )

            if st.button("✅ Apply Optimised Parameters", use_container_width=True):
                st.session_state.apply_opt = True
                st.rerun()

    st.divider()
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.opt_results = None
        st.session_state.apply_opt = False
        # Clear slider keys to force defaults on next run
        for k in ["eil_slider", "cfs_slider", "cd_slider", "fe_slider",
                   "csm_slider", "lpb_slider", "ksm_slider"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# =============================================================================
# MAIN PANEL
# =============================================================================
if st.session_state.uploaded_bytes is None:
    st.info("👈 Upload an Excel file in the sidebar to get started.")
    st.stop()

# Load and prepare data
try:
    df_raw = load_excel(st.session_state.uploaded_bytes)
    df_sorted = prepare_data(df_raw, date_order)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Apply start date filter if set
if start_date_override is not None:
    start_ts = pd.Timestamp(start_date_override)
    df_sorted = df_sorted.loc[df_sorted["date"] >= start_ts].reset_index(drop=True)

if len(df_sorted) < 6:
    st.error("Insufficient data: need at least 6 months of observations.")
    st.stop()

# Build current params dict
params = {
    "effective_income_leverage": effective_income_leverage,
    "clo_funding_spread": clo_funding_spread,
    "credit_drag": credit_drag,
    "fees_expenses": fees_expenses,
    "carry_smoothing_months": carry_smoothing_months,
    "loan_price_beta": loan_price_beta,
    "capital_smoothing_months": capital_smoothing_months,
    "true_up_method": true_up_method,
}

# Run model
try:
    result = run_model(df_sorted, params)
except Exception as e:
    st.error(f"Model calculation error: {e}")
    st.stop()

monthly = result["monthly"]
quarterly = result["quarterly"]
last_fr_date = result["last_fr_date"]

# Verification warnings
if result["verification_errors"]:
    for qlabel, disc in result["verification_errors"]:
        st.warning(f"⚠️ True-up verification failed for {qlabel}: discrepancy = {disc:.8f}")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📈 Chart", "📋 Monthly Returns", "📊 Quarterly Summary"])

# --- TAB 1: CHART ---
with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Monthly Returns (%)", "Cumulative Return Index (base=100)"),
        row_heights=[0.55, 0.45],
    )

    # --- TOP PANEL: Monthly returns ---

    # Model estimate — solid grey up to last Flat Rock date
    if last_fr_date is not None:
        mask_model = monthly["date"] <= last_fr_date
        mask_ext = monthly["date"] >= last_fr_date

        fig.add_trace(
            go.Scatter(
                x=monthly.loc[mask_model, "date"],
                y=monthly.loc[mask_model, "estimated_return"] * 100,
                name="Model Estimate",
                line=dict(color="grey", width=1.5),
                hovertemplate="Date: %{x|%b %Y}<br>Model: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )
        # Estimated extension — dashed grey beyond last Flat Rock
        fig.add_trace(
            go.Scatter(
                x=monthly.loc[mask_ext, "date"],
                y=monthly.loc[mask_ext, "estimated_return"] * 100,
                name="Estimated Extension",
                line=dict(color="grey", dash="dot", width=1.5),
                hovertemplate="Date: %{x|%b %Y}<br>Est. Extension: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )
    else:
        # No Flat Rock data — entire series is estimated
        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["estimated_return"] * 100,
                name="Model Estimate",
                line=dict(color="grey", dash="dot", width=1.5),
                hovertemplate="Date: %{x|%b %Y}<br>Model: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )

    # Flat Rock actual returns (geometric monthly equivalent)
    fr_monthly_mask = monthly["flat_rock_monthly"].notna()
    if fr_monthly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=monthly.loc[fr_monthly_mask, "date"],
                y=monthly.loc[fr_monthly_mask, "flat_rock_monthly"] * 100,
                name="Flat Rock Actual (monthly equiv.)",
                line=dict(color="#2CA02C", width=2),
                hovertemplate="Date: %{x|%b %Y}<br>Flat Rock: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )

    # Vertical line at last Flat Rock date
    if last_fr_date is not None:
        fig.add_vline(
            x=last_fr_date, line_dash="dash", line_color="red", line_width=1.5,
            row=1, col=1,
        )
        fig.add_annotation(
            x=last_fr_date, y=1.0, yref="paper",
            text="Last Flat Rock →",
            showarrow=False, xanchor="right", font=dict(size=10, color="red"),
        )

    # --- BOTTOM PANEL: Cumulative return index ---

    cum_est = 100 * np.cumprod(1 + monthly["estimated_return"].values)

    # Flat Rock cumulative — compound only over months with actual data
    fr_monthly_vals = monthly["flat_rock_monthly"].values.copy()
    cum_fr = np.full(len(fr_monthly_vals), np.nan)
    running = 100.0
    for i in range(len(fr_monthly_vals)):
        if not np.isnan(fr_monthly_vals[i]):
            running *= (1 + fr_monthly_vals[i])
            cum_fr[i] = running

    if last_fr_date is not None:
        mask_model = monthly["date"] <= last_fr_date
        mask_ext = monthly["date"] >= last_fr_date

        fig.add_trace(
            go.Scatter(
                x=monthly.loc[mask_model, "date"],
                y=cum_est[mask_model],
                name="Model Cumulative",
                line=dict(color="grey", width=1.5),
                showlegend=False,
                hovertemplate="Date: %{x|%b %Y}<br>Model Cum.: %{y:.1f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly.loc[mask_ext, "date"],
                y=cum_est[mask_ext],
                name="Est. Extension Cumulative",
                line=dict(color="grey", dash="dot", width=1.5),
                showlegend=False,
                hovertemplate="Date: %{x|%b %Y}<br>Est. Extension Cum.: %{y:.1f}<extra></extra>",
            ),
            row=2, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=monthly["date"], y=cum_est,
                name="Model Cumulative",
                line=dict(color="grey", dash="dot", width=1.5),
                showlegend=False,
                hovertemplate="Date: %{x|%b %Y}<br>Model Cum.: %{y:.1f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Flat Rock cumulative
    fr_cum_mask = pd.Series(cum_fr).notna().values
    if fr_cum_mask.any():
        fig.add_trace(
            go.Scatter(
                x=monthly.loc[fr_cum_mask, "date"],
                y=cum_fr[fr_cum_mask],
                name="Flat Rock Cumulative",
                line=dict(color="#2CA02C", width=2),
                showlegend=False,
                hovertemplate="Date: %{x|%b %Y}<br>Flat Rock Cum.: %{y:.1f}<extra></extra>",
            ),
            row=2, col=1,
        )

    if last_fr_date is not None:
        fig.add_vline(
            x=last_fr_date, line_dash="dash", line_color="red", line_width=1.5,
            row=2, col=1,
        )

    fig.update_layout(
        title="CLO Equity Monthly Returns — Model Estimate vs Flat Rock Actual",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(t=80),
    )
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Index Level", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ann. Return (Model Estimate)", f"{result['annualised_s1']*100:.2f}%")
    ann_fr = result["annualised_fr"]
    col2.metric("Ann. Return (Flat Rock)", f"{ann_fr*100:.2f}%" if not np.isnan(ann_fr) else "N/A")
    col3.metric("Data Points", f"{len(monthly)} months")
    col4.metric("Last Flat Rock", last_fr_date.strftime("%b %Y") if last_fr_date else "N/A")

    # --- Trailing period returns table ---
    if last_fr_date is not None:
        anchor_date = last_fr_date
        st.subheader(f"Trailing Period Returns (annualised, to {anchor_date.strftime('%b %Y')})")
    else:
        anchor_date = monthly["date"].max()
        st.subheader("Trailing Period Returns (annualised)")

    periods = {"1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60, "10Y": 120}

    # Filter to data up to and including the anchor date
    aligned = monthly.loc[monthly["date"] <= anchor_date]

    period_rows = []
    for label, n_months_back in periods.items():
        cutoff = anchor_date - pd.DateOffset(months=n_months_back)
        window = aligned.loc[aligned["date"] > cutoff]
        n_actual = len(window)

        if n_actual < 2:
            period_rows.append({
                "Period": label,
                "Model Estimate (ann.)": "—",
                "Flat Rock Actual (ann.)": "—",
                "Difference": "—",
            })
            continue

        # Model: compound monthly estimated returns then annualise
        total_model = np.prod(1 + window["estimated_return"].values) - 1
        ann_model = (1 + total_model) ** (12.0 / n_actual) - 1

        # Flat Rock: compound only non-NaN monthly equivalents
        fr_vals = window["flat_rock_monthly"].dropna()
        if len(fr_vals) > 0:
            total_fr = np.prod(1 + fr_vals.values) - 1
            ann_fr_period = (1 + total_fr) ** (12.0 / len(fr_vals)) - 1
            fr_str = f"{ann_fr_period * 100:.2f}%"
            diff_str = f"{(ann_model - ann_fr_period) * 100:+.2f}%"
        else:
            fr_str = "—"
            diff_str = "—"

        period_rows.append({
            "Period": label,
            "Model Estimate (ann.)": f"{ann_model * 100:.2f}%",
            "Flat Rock Actual (ann.)": fr_str,
            "Difference": diff_str,
        })

    period_df = pd.DataFrame(period_rows)
    st.dataframe(period_df, use_container_width=False, hide_index=True)

# --- TAB 2: MONTHLY RETURN TABLE ---
with tab2:
    display_df = pd.DataFrame({
        "Date": monthly["date"].dt.strftime("%b-%Y"),
        "SOFR (%)": (monthly["sofr"] * 100).round(2),
        "Loan Spread (bps)": (monthly["loan_spread"] * 10000).round(0).astype("Int64"),
        "Loan Price Level": monthly["loan_price"].round(2),
        "Loan Price Return (%)": (monthly["loan_price_return"] * 100).round(2),
        "Carry Smoothed (% mthly)": (monthly["carry_smoothed"] * 100).round(2),
        "Capital Return Smoothed (% mthly)": (monthly["capital_smoothed"] * 100).round(2),
        "Estimated Return (%)": (monthly["estimated_return"] * 100).round(2),
        "Flat Rock Monthly Equiv. (%)": monthly["flat_rock_monthly"].apply(
            lambda x: f"{x*100:.2f}" if not pd.isna(x) else ""
        ),
        "Flat Rock Qtr Return (%)": monthly["flat_rock_return"].apply(
            lambda x: f"{x*100:.2f}" if not pd.isna(x) else ""
        ),
    })

    # Sort newest-first for display
    display_df = display_df.iloc[::-1].reset_index(drop=True)

    st.dataframe(display_df, use_container_width=True, height=500)

    xlsx_monthly = to_excel_monthly(display_df)
    st.download_button(
        "📥 Download Monthly Table as Excel",
        data=xlsx_monthly,
        file_name="clo_monthly_returns.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --- TAB 3: QUARTERLY SUMMARY ---
with tab3:
    q_display = quarterly.copy()

    # Format numeric columns
    for col in ["Flat Rock Return (%)", "Estimated Quarterly Return (%)",
                "Estimation Error (%)"]:
        if col in q_display.columns:
            q_display[col] = q_display[col].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else ""
            )

    # Drop Actual-Extended column if present
    if "Actual-Extended Quarterly Return (%)" in q_display.columns:
        q_display = q_display.drop(columns=["Actual-Extended Quarterly Return (%)"])

    # Reverse order for newest-first display
    q_display = q_display.iloc[::-1].reset_index(drop=True)

    st.dataframe(q_display, use_container_width=True, height=400)

    # Annualised summary row
    ann_fr = result["annualised_fr"]
    ann_fr_str = f"{ann_fr*100:.2f}%" if not np.isnan(ann_fr) else "N/A"
    st.markdown(
        f"**Annualised Returns** — Model Estimate: **{result['annualised_s1']*100:.2f}%** | "
        f"Flat Rock Actual: **{ann_fr_str}** | "
        f"Period: {monthly['date'].min().strftime('%b %Y')} – {monthly['date'].max().strftime('%b %Y')} "
        f"({len(monthly)} months)"
    )

    # Convert for Excel export (use numeric quarterly df, not formatted)
    xlsx_quarterly = to_excel_quarterly(q_display)
    st.download_button(
        "📥 Download Quarterly Summary as Excel",
        data=xlsx_quarterly,
        file_name="clo_quarterly_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
