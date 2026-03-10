"""
Mortgage Amortization & Refinance Comparison Tool
==================================================
A production-quality Streamlit app for mortgage brokers and borrowers to:
  - Calculate full amortization schedules (current and refinance scenarios)
  - Compare lifetime costs, interest, and payoff timelines
  - Evaluate the impact of extra payments and refinancing decisions
  - Generate interactive charts and exportable tables

Run with:  streamlit run app.py
"""

import math
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mortgage Refinance Comparison Tool",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_currency(value: float) -> str:
    """Return a USD-formatted string, e.g. $1,234.56"""
    return f"${value:,.2f}"


def fmt_percent(value: float) -> str:
    """Return a percentage string, e.g. 3.50%"""
    return f"{value:.2f}%"


def fmt_months(months: int) -> str:
    """Return a human-readable months string, e.g. 360 months (30.0 yrs)"""
    years = months / 12
    return f"{months} months ({years:.1f} yrs)"


# ---------------------------------------------------------------------------
# Core calculation helpers
# ---------------------------------------------------------------------------

def monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Calculate the fixed monthly P&I payment using the standard amortization formula.

    M = P * [r(1+r)^n] / [(1+r)^n - 1]

    Parameters
    ----------
    principal      : loan amount in dollars
    annual_rate    : annual interest rate as a percentage (e.g. 6.5 for 6.5%)
    term_months    : loan term in months

    Returns
    -------
    float : monthly payment (principal + interest only)
    """
    if principal <= 0 or term_months <= 0:
        return 0.0
    if annual_rate == 0:
        # Zero-interest edge case: equal principal-only payments
        return principal / term_months
    r = annual_rate / 100 / 12  # monthly rate
    return principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)


def build_amortization_schedule(
    principal: float,
    annual_rate: float,
    term_months: int,
    start_date: date,
    property_value: float,
    pmi_rate_annual: float,
    tax_insurance_annual_pct: float,
    monthly_extra: float = 0.0,
    onetime_extra_month: int = 0,
    onetime_extra_amount: float = 0.0,
    starting_month_number: int = 1,
) -> pd.DataFrame:
    """
    Build a month-by-month amortization schedule as a pandas DataFrame.

    PMI is charged when the current loan balance > 80% of property_value.
    PMI is calculated as (pmi_rate_annual / 100) * balance / 12 each month.
    Tax & insurance is a flat monthly cost derived from tax_insurance_annual_pct.

    Parameters
    ----------
    principal               : opening loan balance
    annual_rate             : annual interest rate (%)
    term_months             : loan term in months
    start_date              : first payment date (date object)
    property_value          : current appraised property value
    pmi_rate_annual         : annual PMI rate (%)
    tax_insurance_annual_pct: annual tax+insurance as % of property value
    monthly_extra           : additional principal paid every month
    onetime_extra_month     : payment number (relative to this schedule) for lump sum
    onetime_extra_amount    : dollar amount of the one-time extra payment
    starting_month_number   : used when combining pre- and post-refi schedules

    Returns
    -------
    pd.DataFrame with columns:
        Payment_Number, Payment_Date, Beginning_Balance, Scheduled_Payment,
        Principal, Interest, Extra_Payment, Ending_Balance,
        PMI, Tax_Insurance, Total_Monthly_Outflow,
        Cumulative_Interest, Cumulative_Principal, Cumulative_Total_Paid
    """
    if principal <= 0 or term_months <= 0:
        return pd.DataFrame()

    r = annual_rate / 100 / 12  # monthly interest rate
    scheduled_pi = monthly_payment(principal, annual_rate, term_months)
    monthly_tax_ins = property_value * (tax_insurance_annual_pct / 100) / 12

    rows = []
    balance = principal
    cum_interest = 0.0
    cum_principal = 0.0
    cum_total = 0.0

    for i in range(1, term_months + 1):
        if balance <= 0:
            break

        payment_number = starting_month_number + i - 1
        payment_date = start_date + relativedelta(months=i - 1)
        beginning_balance = balance

        # Interest for this month
        interest = balance * r if annual_rate > 0 else 0.0

        # Scheduled principal portion
        principal_portion = scheduled_pi - interest
        # Guard against floating-point edge cases on last payment
        principal_portion = max(0.0, principal_portion)

        # Extra payments
        extra = monthly_extra
        if i == onetime_extra_month:
            extra += onetime_extra_amount

        # Total principal reduction this month
        total_principal = principal_portion + extra

        # Do not reduce balance below zero
        if beginning_balance - total_principal < 0:
            total_principal = beginning_balance
            extra = total_principal - principal_portion
            extra = max(0.0, extra)
            principal_portion = beginning_balance - extra
            if principal_portion < 0:
                principal_portion = 0.0
                extra = beginning_balance

        ending_balance = beginning_balance - total_principal
        ending_balance = max(0.0, ending_balance)

        # PMI: charged when balance > 80% LTV; based on current balance
        ltv = beginning_balance / property_value if property_value > 0 else 0.0
        pmi = (beginning_balance * (pmi_rate_annual / 100) / 12) if ltv > 0.80 else 0.0

        actual_payment = min(scheduled_pi, beginning_balance + interest)

        cum_interest += interest
        cum_principal += principal_portion + extra
        cum_total += actual_payment + extra + pmi + monthly_tax_ins

        rows.append(
            {
                "Payment_Number": payment_number,
                "Payment_Date": payment_date,
                "Beginning_Balance": round(beginning_balance, 2),
                "Scheduled_Payment": round(scheduled_pi, 2),
                "Principal": round(principal_portion, 2),
                "Interest": round(interest, 2),
                "Extra_Payment": round(extra, 2),
                "Ending_Balance": round(ending_balance, 2),
                "PMI": round(pmi, 2),
                "Tax_Insurance": round(monthly_tax_ins, 2),
                "Total_Monthly_Outflow": round(
                    actual_payment + extra + pmi + monthly_tax_ins, 2
                ),
                "Cumulative_Interest": round(cum_interest, 2),
                "Cumulative_Principal": round(cum_principal, 2),
                "Cumulative_Total_Paid": round(cum_total, 2),
            }
        )

        balance = ending_balance
        if balance == 0:
            break

    return pd.DataFrame(rows)


def build_refinance_schedule(
    original_schedule_pre_refi: pd.DataFrame,
    refi_month: int,
    refi_annual_rate: float,
    refi_term_months: int,
    property_value: float,
    pmi_rate_annual: float,
    tax_insurance_annual_pct: float,
    closing_costs: float = 0.0,
    roll_closing_costs: bool = True,
    monthly_extra: float = 0.0,
    onetime_extra_month: int = 0,
    onetime_extra_amount: float = 0.0,
) -> pd.DataFrame:
    """
    Build the combined amortization schedule for the refinance scenario:
      - Pre-refi months: taken directly from original_schedule_pre_refi (rows 1..refi_month)
      - Post-refi months: new schedule starting from remaining principal at refi_month

    Closing costs can be:
      - Rolled into the new loan balance (roll_closing_costs=True)
      - Treated as upfront cash (roll_closing_costs=False) — deducted from comparison

    Parameters
    ----------
    original_schedule_pre_refi : pre-refi amortization schedule (through refi_month)
    refi_month                 : the month number at which refinancing occurs
    refi_annual_rate           : new interest rate (%)
    refi_term_months           : new loan term in months
    property_value             : appraised property value (unchanged)
    pmi_rate_annual            : PMI rate for new loan (%)
    tax_insurance_annual_pct   : annual tax+ins as % of property value
    closing_costs              : dollar amount of closing costs
    roll_closing_costs         : if True, add closing costs to new principal
    monthly_extra              : extra monthly payment on new loan
    onetime_extra_month        : one-time extra payment month (relative to new schedule)
    onetime_extra_amount       : one-time extra payment amount

    Returns
    -------
    pd.DataFrame — combined pre+post refi schedule with same columns as
                   build_amortization_schedule output
    """
    if original_schedule_pre_refi.empty:
        return pd.DataFrame()

    # Slice to rows up to and including refi_month
    pre_refi = original_schedule_pre_refi[
        original_schedule_pre_refi["Payment_Number"] <= refi_month
    ].copy()

    if pre_refi.empty:
        return pd.DataFrame()

    # Remaining balance at end of refi_month
    remaining_balance = pre_refi.iloc[-1]["Ending_Balance"]
    refi_start_date = pre_refi.iloc[-1]["Payment_Date"] + relativedelta(months=1)

    new_principal = remaining_balance
    if roll_closing_costs:
        new_principal += closing_costs

    # Build post-refi schedule
    post_refi = build_amortization_schedule(
        principal=new_principal,
        annual_rate=refi_annual_rate,
        term_months=refi_term_months,
        start_date=refi_start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate_annual,
        tax_insurance_annual_pct=tax_insurance_annual_pct,
        monthly_extra=monthly_extra,
        onetime_extra_month=onetime_extra_month,
        onetime_extra_amount=onetime_extra_amount,
        starting_month_number=refi_month + 1,
    )

    if post_refi.empty:
        return pre_refi

    # Re-accumulate cumulative columns across the combined schedule
    pre_cum_int = pre_refi.iloc[-1]["Cumulative_Interest"]
    pre_cum_prin = pre_refi.iloc[-1]["Cumulative_Principal"]
    pre_cum_total = pre_refi.iloc[-1]["Cumulative_Total_Paid"]

    post_refi = post_refi.copy()
    post_refi["Cumulative_Interest"] = (
        post_refi["Cumulative_Interest"] + pre_cum_int
    ).round(2)
    post_refi["Cumulative_Principal"] = (
        post_refi["Cumulative_Principal"] + pre_cum_prin
    ).round(2)
    post_refi["Cumulative_Total_Paid"] = (
        post_refi["Cumulative_Total_Paid"] + pre_cum_total
    ).round(2)

    combined = pd.concat([pre_refi, post_refi], ignore_index=True)
    return combined


def summarize_schedule(
    schedule: pd.DataFrame,
    original_term_months: int,
    label: str = "",
) -> dict:
    """
    Compute summary metrics from a completed amortization schedule.

    Parameters
    ----------
    schedule            : output of build_amortization_schedule or build_refinance_schedule
    original_term_months: contractual term of the original loan in months
    label               : scenario name (e.g. "Current Loan", "Refinance")

    Returns
    -------
    dict with summary statistics
    """
    if schedule.empty:
        return {}

    actual_payoff_month = int(schedule["Payment_Number"].max())
    total_interest = round(schedule["Interest"].sum(), 2)
    total_paid = round(schedule["Cumulative_Total_Paid"].iloc[-1], 2)
    time_saved_months = original_term_months - actual_payoff_month
    time_saved_years = round(time_saved_months / 12, 2)

    # Original total interest = what would have been paid with no extras (already schedule)
    # For a "stay in original loan" scenario the original interest is computed externally.
    # Here we return actual values; comparison logic handles differences.

    return {
        "Scenario": label,
        "Original_Term_Months": original_term_months,
        "Actual_Payoff_Month": actual_payoff_month,
        "Time_Saved_Months": time_saved_months,
        "Time_Saved_Years": time_saved_years,
        "Actual_Interest_Paid": total_interest,
        "Total_Paid": total_paid,
    }


def compare_scenarios(
    summary_current: dict,
    summary_refi: dict,
    baseline_interest_current: float,
    baseline_interest_refi: float,
    closing_costs: float = 0.0,
    roll_closing_costs: bool = True,
) -> dict:
    """
    Compute the comparison metrics between staying in the current loan
    and refinancing.

    Parameters
    ----------
    summary_current         : summarize_schedule output for current scenario
    summary_refi            : summarize_schedule output for refi scenario
    baseline_interest_current: total interest on original loan with no extras
    baseline_interest_refi  : total interest on original loan with no extras
    closing_costs           : refinance closing costs
    roll_closing_costs      : True = rolled into loan; False = upfront cash

    Returns
    -------
    dict with comparison statistics and recommendation
    """
    interest_diff = (
        summary_current["Actual_Interest_Paid"]
        - summary_refi["Actual_Interest_Paid"]
    )
    total_paid_diff = (
        summary_current["Total_Paid"] - summary_refi["Total_Paid"]
    )
    if not roll_closing_costs:
        # Upfront cash cost reduces the refinance benefit
        total_paid_diff -= closing_costs

    payoff_diff_months = (
        summary_current["Actual_Payoff_Month"]
        - summary_refi["Actual_Payoff_Month"]
    )

    # Break-even estimate: months until closing cost savings equal monthly savings
    # Monthly savings ≈ difference in monthly payment
    # We compare total paid differences on a per-month basis as proxy
    breakeven_month: int | str = "N/A"
    if closing_costs > 0:
        monthly_saving_estimate = (
            interest_diff / summary_refi["Actual_Payoff_Month"]
            if summary_refi["Actual_Payoff_Month"] > 0
            else 0
        )
        if monthly_saving_estimate > 0:
            breakeven_month = math.ceil(closing_costs / monthly_saving_estimate)
        else:
            breakeven_month = "N/A (no monthly savings)"

    refi_beneficial = interest_diff > 0 or payoff_diff_months > 0

    return {
        "Interest_Saved": round(interest_diff, 2),
        "Total_Paid_Saved": round(total_paid_diff, 2),
        "Payoff_Diff_Months": payoff_diff_months,
        "Payoff_Diff_Years": round(payoff_diff_months / 12, 2),
        "Breakeven_Month": breakeven_month,
        "Refi_Beneficial": refi_beneficial,
        "Interest_Saved_Pct": round(
            interest_diff / baseline_interest_current * 100, 2
        )
        if baseline_interest_current > 0
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_remaining_balance(
    df_current: pd.DataFrame, df_refi: pd.DataFrame
) -> go.Figure:
    """Line chart: Remaining loan balance over time for both scenarios."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_current["Payment_Number"],
            y=df_current["Ending_Balance"],
            mode="lines",
            name="Current Loan",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Month %{x}<br>Balance: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_refi["Payment_Number"],
            y=df_refi["Ending_Balance"],
            mode="lines",
            name="Refinance Scenario",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="Month %{x}<br>Balance: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Remaining Loan Balance Over Time",
        xaxis_title="Payment Month",
        yaxis_title="Remaining Balance ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    return fig


def chart_cumulative_interest(
    df_current: pd.DataFrame, df_refi: pd.DataFrame
) -> go.Figure:
    """Line chart: Cumulative interest paid over time for both scenarios."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_current["Payment_Number"],
            y=df_current["Cumulative_Interest"],
            mode="lines",
            name="Current Loan",
            line=dict(color="#d62728", width=2),
            hovertemplate="Month %{x}<br>Cum. Interest: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_refi["Payment_Number"],
            y=df_refi["Cumulative_Interest"],
            mode="lines",
            name="Refinance Scenario",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            hovertemplate="Month %{x}<br>Cum. Interest: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cumulative Interest Paid Over Time",
        xaxis_title="Payment Month",
        yaxis_title="Cumulative Interest ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    return fig


def chart_payment_composition(df: pd.DataFrame, title: str = "Payment Composition") -> go.Figure:
    """Stacked bar chart: monthly payment breakdown (principal, interest, PMI, tax/ins)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Payment_Number"],
            y=df["Principal"],
            name="Principal",
            marker_color="#1f77b4",
            hovertemplate="Month %{x}<br>Principal: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["Payment_Number"],
            y=df["Interest"],
            name="Interest",
            marker_color="#d62728",
            hovertemplate="Month %{x}<br>Interest: $%{y:,.2f}<extra></extra>",
        )
    )
    if df["PMI"].sum() > 0:
        fig.add_trace(
            go.Bar(
                x=df["Payment_Number"],
                y=df["PMI"],
                name="PMI",
                marker_color="#9467bd",
                hovertemplate="Month %{x}<br>PMI: $%{y:,.2f}<extra></extra>",
            )
        )
    if df["Tax_Insurance"].sum() > 0:
        fig.add_trace(
            go.Bar(
                x=df["Payment_Number"],
                y=df["Tax_Insurance"],
                name="Tax & Insurance",
                marker_color="#8c564b",
                hovertemplate="Month %{x}<br>Tax & Ins: $%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Payment Month",
        yaxis_title="Monthly Amount ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    return fig


def chart_equity_growth(
    df_current: pd.DataFrame, df_refi: pd.DataFrame, property_value: float
) -> go.Figure:
    """Line chart: Equity (property value minus remaining balance) over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_current["Payment_Number"],
            y=property_value - df_current["Ending_Balance"],
            mode="lines",
            name="Current Loan Equity",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Month %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_refi["Payment_Number"],
            y=property_value - df_refi["Ending_Balance"],
            mode="lines",
            name="Refinance Equity",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="Month %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=property_value,
        line_dash="dot",
        annotation_text="Full Equity",
        annotation_position="bottom right",
        line_color="green",
    )
    fig.update_layout(
        title="Home Equity Growth Over Time",
        xaxis_title="Payment Month",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = "", positive_good: bool = True):
    """Render a styled metric card using st.metric."""
    st.metric(label=label, value=value, delta=delta if delta else None)


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar / Input section
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """
    Render all input controls and return a dict of validated inputs.
    All defaults are chosen to represent a realistic mortgage scenario.
    """
    st.sidebar.header("🏠 Loan Parameters")

    with st.sidebar.expander("📋 Current Loan", expanded=True):
        loan_amount = st.number_input(
            "Original Loan Amount ($)",
            min_value=10_000.0,
            max_value=10_000_000.0,
            value=400_000.0,
            step=1_000.0,
            format="%.2f",
            help="The original principal balance of your current mortgage.",
        )
        current_rate = st.number_input(
            "Current Interest Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=7.0,
            step=0.125,
            format="%.3f",
            help="Annual interest rate of your current mortgage.",
        )
        loan_term_years = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            max_value=50,
            value=30,
            step=1,
            help="Original loan term in years.",
        )
        start_date = st.date_input(
            "Loan Start Date",
            value=date(2023, 1, 1),
            help="Date of the first payment.",
        )

    with st.sidebar.expander("🏡 Property & Escrow", expanded=True):
        property_value = st.number_input(
            "Property Value ($)",
            min_value=10_000.0,
            max_value=10_000_000.0,
            value=500_000.0,
            step=1_000.0,
            format="%.2f",
            help="Current appraised property value. Used for LTV and PMI calculation.",
        )
        pmi_rate = st.number_input(
            "PMI Rate (% per year)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help=(
                "Annual PMI rate as a % of the outstanding loan balance. "
                "PMI is automatically removed when LTV ≤ 80%."
            ),
        )
        tax_ins_pct = st.number_input(
            "Annual Tax & Insurance (% of Property Value)",
            min_value=0.0,
            max_value=10.0,
            value=1.25,
            step=0.05,
            format="%.2f",
            help="Annual property taxes + homeowners insurance as a % of property value.",
        )

    with st.sidebar.expander("🔄 Refinance Options", expanded=True):
        refi_rate = st.number_input(
            "Refinance Interest Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=5.5,
            step=0.125,
            format="%.3f",
        )
        refi_month = st.number_input(
            "Refinance at Month #",
            min_value=1,
            max_value=600,
            value=36,
            step=1,
            help=(
                "The month in your current loan at which you refinance. "
                "Month 36 means you refinance after 3 years of payments."
            ),
        )
        refi_term_years = st.number_input(
            "Refinance Loan Term (Years)",
            min_value=1,
            max_value=50,
            value=30,
            step=1,
        )
        closing_costs = st.number_input(
            "Refinance Closing Costs ($)",
            min_value=0.0,
            max_value=100_000.0,
            value=5_000.0,
            step=500.0,
            format="%.2f",
            help="Total closing costs for the refinance.",
        )
        roll_closing_costs = st.checkbox(
            "Roll closing costs into new loan balance",
            value=True,
            help=(
                "If checked, closing costs are added to the new loan principal. "
                "If unchecked, they are treated as an upfront cash expense."
            ),
        )

    with st.sidebar.expander("💰 Extra Payments", expanded=False):
        apply_extra_both = st.checkbox(
            "Apply extra payments to both scenarios",
            value=True,
            help=(
                "If checked, the same extra payment settings apply to both "
                "the current loan and refinance scenarios."
            ),
        )
        monthly_extra = st.number_input(
            "Monthly Extra Principal Payment ($)",
            min_value=0.0,
            max_value=100_000.0,
            value=0.0,
            step=50.0,
            format="%.2f",
        )
        onetime_month = st.number_input(
            "One-Time Extra Payment at Month #",
            min_value=0,
            max_value=600,
            value=0,
            step=1,
            help="Enter 0 to disable. Month is relative to the start of the loan.",
        )
        onetime_amount = st.number_input(
            "One-Time Extra Payment Amount ($)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=0.0,
            step=500.0,
            format="%.2f",
        )

    # ---- Input validation ----
    errors = []
    if loan_amount <= 0:
        errors.append("Loan amount must be greater than zero.")
    if property_value <= 0:
        errors.append("Property value must be greater than zero.")
    if refi_month >= loan_term_years * 12:
        errors.append(
            f"Refinance month ({refi_month}) must be before the original loan term "
            f"({loan_term_years * 12} months)."
        )
    if current_rate < 0 or refi_rate < 0:
        errors.append("Interest rates cannot be negative.")

    return {
        "loan_amount": loan_amount,
        "current_rate": current_rate,
        "loan_term_months": int(loan_term_years * 12),
        "start_date": start_date,
        "property_value": property_value,
        "pmi_rate": pmi_rate,
        "tax_ins_pct": tax_ins_pct,
        "refi_rate": refi_rate,
        "refi_month": int(refi_month),
        "refi_term_months": int(refi_term_years * 12),
        "closing_costs": closing_costs,
        "roll_closing_costs": roll_closing_costs,
        "monthly_extra": monthly_extra,
        "onetime_month": int(onetime_month),
        "onetime_amount": onetime_amount,
        "apply_extra_both": apply_extra_both,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    st.title("🏠 Mortgage Amortization & Refinance Comparison Tool")
    st.caption(
        "Compare your current loan against a refinance scenario — with charts, "
        "tables, and a clear recommendation. For informational purposes only."
    )

    inputs = render_sidebar()

    # ---- Validation errors ----
    if inputs["errors"]:
        for err in inputs["errors"]:
            st.error(f"⚠️ {err}")
        st.stop()

    # ---- Unpack inputs ----
    loan_amount = inputs["loan_amount"]
    current_rate = inputs["current_rate"]
    loan_term_months = inputs["loan_term_months"]
    start_date = inputs["start_date"]
    property_value = inputs["property_value"]
    pmi_rate = inputs["pmi_rate"]
    tax_ins_pct = inputs["tax_ins_pct"]
    refi_rate = inputs["refi_rate"]
    refi_month = inputs["refi_month"]
    refi_term_months = inputs["refi_term_months"]
    closing_costs = inputs["closing_costs"]
    roll_closing_costs = inputs["roll_closing_costs"]
    monthly_extra = inputs["monthly_extra"]
    onetime_month = inputs["onetime_month"]
    onetime_amount = inputs["onetime_amount"]
    apply_extra_both = inputs["apply_extra_both"]

    # ---- Baseline schedule (no extra payments) for comparison reference ----
    df_baseline = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=current_rate,
        term_months=loan_term_months,
        start_date=start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        monthly_extra=0.0,
        onetime_extra_month=0,
        onetime_extra_amount=0.0,
    )
    baseline_total_interest = df_baseline["Interest"].sum()

    # ---- Current loan schedule (with extra payments) ----
    df_current = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=current_rate,
        term_months=loan_term_months,
        start_date=start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        monthly_extra=monthly_extra,
        onetime_extra_month=onetime_month,
        onetime_extra_amount=onetime_amount,
    )

    # Monthly P&I for current loan
    current_monthly_pi = monthly_payment(loan_amount, current_rate, loan_term_months)
    monthly_tax_ins = property_value * (tax_ins_pct / 100) / 12
    initial_pmi = (
        (loan_amount * (pmi_rate / 100) / 12)
        if (loan_amount / property_value) > 0.80
        else 0.0
    )
    current_total_monthly = current_monthly_pi + initial_pmi + monthly_tax_ins

    # ---- Refinance scenario ----
    # Extra payments for refi scenario
    refi_monthly_extra = monthly_extra if apply_extra_both else 0.0
    refi_onetime_month = onetime_month if apply_extra_both else 0
    refi_onetime_amount = onetime_amount if apply_extra_both else 0.0

    # Pre-refi portion of original schedule (no extras for pre-refi period)
    df_pre_refi_original = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=current_rate,
        term_months=loan_term_months,
        start_date=start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        monthly_extra=0.0,
        onetime_extra_month=0,
        onetime_extra_amount=0.0,
    )

    df_refi = build_refinance_schedule(
        original_schedule_pre_refi=df_pre_refi_original,
        refi_month=refi_month,
        refi_annual_rate=refi_rate,
        refi_term_months=refi_term_months,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        closing_costs=closing_costs,
        roll_closing_costs=roll_closing_costs,
        monthly_extra=refi_monthly_extra,
        onetime_extra_month=refi_onetime_month,
        onetime_extra_amount=refi_onetime_amount,
    )

    # Remaining principal at refi_month (for display)
    refi_balance_at_refi = df_pre_refi_original[
        df_pre_refi_original["Payment_Number"] == refi_month
    ]["Ending_Balance"].values
    refi_balance_at_refi = refi_balance_at_refi[0] if len(refi_balance_at_refi) > 0 else loan_amount
    new_refi_principal = refi_balance_at_refi + (closing_costs if roll_closing_costs else 0)

    # Refinance monthly P&I
    refi_monthly_pi = monthly_payment(new_refi_principal, refi_rate, refi_term_months)
    refi_initial_pmi = (
        (new_refi_principal * (pmi_rate / 100) / 12)
        if (new_refi_principal / property_value) > 0.80
        else 0.0
    )
    refi_total_monthly = refi_monthly_pi + refi_initial_pmi + monthly_tax_ins

    # ---- Summaries ----
    summary_current = summarize_schedule(df_current, loan_term_months, "Current Loan")
    summary_refi = summarize_schedule(df_refi, loan_term_months, "Refinance Scenario")

    # Baseline refi interest (no extras) for savings % calc
    df_refi_baseline = build_refinance_schedule(
        original_schedule_pre_refi=df_pre_refi_original,
        refi_month=refi_month,
        refi_annual_rate=refi_rate,
        refi_term_months=refi_term_months,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        closing_costs=closing_costs,
        roll_closing_costs=roll_closing_costs,
        monthly_extra=0.0,
        onetime_extra_month=0,
        onetime_extra_amount=0.0,
    )
    baseline_refi_interest = df_refi_baseline["Interest"].sum()

    comparison = compare_scenarios(
        summary_current=summary_current,
        summary_refi=summary_refi,
        baseline_interest_current=baseline_total_interest,
        baseline_interest_refi=baseline_refi_interest,
        closing_costs=closing_costs,
        roll_closing_costs=roll_closing_costs,
    )

    # PMI drop-off months
    def pmi_dropoff_month(df: pd.DataFrame) -> str:
        pmi_rows = df[df["PMI"] > 0]
        if pmi_rows.empty:
            return "No PMI"
        return f"Month {int(pmi_rows['Payment_Number'].max() + 1)}"

    pmi_off_current = pmi_dropoff_month(df_current)
    pmi_off_refi = pmi_dropoff_month(df_refi)

    # =========================================================================
    # SECTION 1 — Summary KPI Cards
    # =========================================================================
    section_header("📊 Summary", "Key metrics at a glance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current Monthly Payment",
            fmt_currency(current_total_monthly),
            help="P&I + PMI + Tax & Insurance (month 1)",
        )
        st.metric(
            "Current Lifetime Interest",
            fmt_currency(summary_current["Actual_Interest_Paid"]),
        )
    with col2:
        st.metric(
            "Refi Monthly Payment",
            fmt_currency(refi_total_monthly),
            delta=fmt_currency(refi_total_monthly - current_total_monthly),
            delta_color="inverse",
            help="P&I + PMI + Tax & Insurance after refinancing",
        )
        st.metric(
            "Refi Lifetime Interest",
            fmt_currency(summary_refi["Actual_Interest_Paid"]),
            delta=fmt_currency(
                summary_refi["Actual_Interest_Paid"]
                - summary_current["Actual_Interest_Paid"]
            ),
            delta_color="inverse",
        )
    with col3:
        interest_saved = comparison["Interest_Saved"]
        st.metric(
            "Interest Savings (Refi vs Stay)",
            fmt_currency(abs(interest_saved)),
            delta="You save" if interest_saved > 0 else "You pay more",
            delta_color="normal" if interest_saved > 0 else "inverse",
        )
        st.metric(
            "Total Paid Difference",
            fmt_currency(abs(comparison["Total_Paid_Saved"])),
            delta="Refi saves" if comparison["Total_Paid_Saved"] > 0 else "Refi costs more",
            delta_color="normal" if comparison["Total_Paid_Saved"] > 0 else "inverse",
        )
    with col4:
        payoff_diff = comparison["Payoff_Diff_Months"]
        st.metric(
            "Payoff Difference (Months)",
            f"{abs(payoff_diff)} months",
            delta="Refi pays off sooner" if payoff_diff > 0 else ("Same" if payoff_diff == 0 else "Refi pays off later"),
            delta_color="normal" if payoff_diff >= 0 else "inverse",
        )
        bkeven = comparison["Breakeven_Month"]
        st.metric(
            "Est. Break-Even",
            str(bkeven) if isinstance(bkeven, str) else f"Month {bkeven}",
            help=(
                "Estimated month at which the cumulative monthly savings from "
                "refinancing exceed the closing costs paid."
            ),
        )

    # ---- Quick context cards ----
    st.markdown("---")
    info_cols = st.columns(4)
    info_cols[0].info(f"**New Refi Balance:** {fmt_currency(new_refi_principal)}\n\n"
                      f"*(at month {refi_month})*")
    info_cols[1].info(f"**PMI Drops Off**\n\nCurrent: {pmi_off_current}\n\nRefi: {pmi_off_refi}")
    info_cols[2].info(
        f"**Current Payoff:** Month {summary_current['Actual_Payoff_Month']}\n\n"
        f"**Refi Payoff:** Month {summary_refi['Actual_Payoff_Month']}"
    )
    info_cols[3].info(
        f"**Baseline Interest (no extras)**\n\n"
        f"Current: {fmt_currency(baseline_total_interest)}\n\n"
        f"Refi: {fmt_currency(baseline_refi_interest)}"
    )

    # =========================================================================
    # SECTION 2 — Comparison Table
    # =========================================================================
    section_header("🔍 Scenario Comparison", "Side-by-side loan outcome comparison")

    comparison_data = {
        "Metric": [
            "Scenario",
            "Original Term (months)",
            "Actual Payoff Month",
            "Time Saved (months)",
            "Time Saved (years)",
            "Actual Interest Paid",
            "Total Paid (P+I+PMI+T&I)",
            "Monthly P&I Payment",
            "Monthly Total Payment (est.)",
        ],
        "Current Loan": [
            "Stay in Current Loan",
            fmt_months(summary_current["Original_Term_Months"]),
            fmt_months(summary_current["Actual_Payoff_Month"]),
            f"{summary_current['Time_Saved_Months']} months",
            f"{summary_current['Time_Saved_Years']} yrs",
            fmt_currency(summary_current["Actual_Interest_Paid"]),
            fmt_currency(summary_current["Total_Paid"]),
            fmt_currency(current_monthly_pi),
            fmt_currency(current_total_monthly),
        ],
        "Refinance Scenario": [
            "Refinance at Month " + str(refi_month),
            fmt_months(summary_refi["Original_Term_Months"]),
            fmt_months(summary_refi["Actual_Payoff_Month"]),
            f"{summary_refi['Time_Saved_Months']} months",
            f"{summary_refi['Time_Saved_Years']} yrs",
            fmt_currency(summary_refi["Actual_Interest_Paid"]),
            fmt_currency(summary_refi["Total_Paid"]),
            fmt_currency(refi_monthly_pi),
            fmt_currency(refi_total_monthly),
        ],
    }
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.set_index("Metric"), use_container_width=True)

    csv_comparison = df_comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Comparison CSV",
        data=csv_comparison,
        file_name="mortgage_comparison.csv",
        mime="text/csv",
    )

    # =========================================================================
    # SECTION 3 — Charts
    # =========================================================================
    section_header("📈 Charts", "Interactive visualizations of loan scenarios")

    tab_bal, tab_int, tab_comp_cur, tab_comp_refi, tab_equity = st.tabs([
        "Remaining Balance",
        "Cumulative Interest",
        "Payment Composition — Current",
        "Payment Composition — Refi",
        "Equity Growth",
    ])

    with tab_bal:
        st.plotly_chart(
            chart_remaining_balance(df_current, df_refi),
            use_container_width=True,
        )

    with tab_int:
        st.plotly_chart(
            chart_cumulative_interest(df_current, df_refi),
            use_container_width=True,
        )

    with tab_comp_cur:
        st.plotly_chart(
            chart_payment_composition(df_current, "Payment Composition — Current Loan"),
            use_container_width=True,
        )

    with tab_comp_refi:
        st.plotly_chart(
            chart_payment_composition(df_refi, "Payment Composition — Refinance Scenario"),
            use_container_width=True,
        )

    with tab_equity:
        st.plotly_chart(
            chart_equity_growth(df_current, df_refi, property_value),
            use_container_width=True,
        )

    # =========================================================================
    # SECTION 4 — Amortization Tables
    # =========================================================================
    section_header("📋 Amortization Tables", "Month-by-month payment detail")

    table_choice = st.radio(
        "View schedule for:",
        ["Current Loan", "Refinance Scenario"],
        horizontal=True,
    )

    def format_schedule_for_display(df: pd.DataFrame) -> pd.DataFrame:
        """Return a display-friendly version of the amortization schedule."""
        display = df.copy()
        display["Payment_Date"] = display["Payment_Date"].apply(
            lambda d: d.strftime("%b %Y") if hasattr(d, "strftime") else str(d)
        )
        currency_cols = [
            "Beginning_Balance", "Scheduled_Payment", "Principal", "Interest",
            "Extra_Payment", "Ending_Balance", "PMI", "Tax_Insurance",
            "Total_Monthly_Outflow", "Cumulative_Interest", "Cumulative_Principal",
            "Cumulative_Total_Paid",
        ]
        for col in currency_cols:
            if col in display.columns:
                display[col] = display[col].apply(lambda v: f"${v:,.2f}")
        return display

    if table_choice == "Current Loan":
        st.caption(f"Showing {len(df_current)} payment rows for current loan scenario.")
        st.dataframe(
            format_schedule_for_display(df_current),
            use_container_width=True,
            height=400,
        )
        csv_current = df_current.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Current Loan Schedule CSV",
            data=csv_current,
            file_name="current_loan_schedule.csv",
            mime="text/csv",
        )
    else:
        st.caption(
            f"Showing {len(df_refi)} payment rows for refinance scenario "
            f"(refinance occurs at month {refi_month})."
        )
        st.dataframe(
            format_schedule_for_display(df_refi),
            use_container_width=True,
            height=400,
        )
        csv_refi = df_refi.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Refinance Schedule CSV",
            data=csv_refi,
            file_name="refinance_schedule.csv",
            mime="text/csv",
        )

    # =========================================================================
    # SECTION 5 — Recommendation
    # =========================================================================
    section_header("💡 Recommendation", "Data-based analysis of your refinance decision")

    reasons_for = []
    reasons_against = []

    monthly_savings = current_total_monthly - refi_total_monthly
    if monthly_savings > 0:
        reasons_for.append(
            f"Lower monthly payment by **{fmt_currency(monthly_savings)}/month** "
            f"({fmt_currency(monthly_savings * 12)}/year)."
        )
    else:
        reasons_against.append(
            f"Monthly payment increases by **{fmt_currency(abs(monthly_savings))}/month**."
        )

    if comparison["Interest_Saved"] > 0:
        reasons_for.append(
            f"Saves **{fmt_currency(comparison['Interest_Saved'])}** in lifetime interest "
            f"(**{fmt_percent(comparison['Interest_Saved_Pct'])}** savings)."
        )
    else:
        reasons_against.append(
            f"Costs **{fmt_currency(abs(comparison['Interest_Saved']))}** more in lifetime interest."
        )

    if comparison["Payoff_Diff_Months"] > 0:
        reasons_for.append(
            f"Pays off **{comparison['Payoff_Diff_Months']} months sooner** "
            f"({comparison['Payoff_Diff_Years']:.1f} years)."
        )
    elif comparison["Payoff_Diff_Months"] < 0:
        reasons_against.append(
            f"Takes **{abs(comparison['Payoff_Diff_Months'])} months longer** to pay off."
        )

    if closing_costs > 0 and isinstance(bkeven, int):
        if bkeven <= 36:
            reasons_for.append(
                f"Closing costs recover quickly — break-even in **{bkeven} months**."
            )
        elif bkeven <= 60:
            reasons_for.append(
                f"Closing costs recover in **{bkeven} months** — moderate payback period."
            )
        else:
            reasons_against.append(
                f"Closing costs take **{bkeven} months** to recover — long payback period."
            )
    elif closing_costs == 0:
        reasons_for.append("No closing costs — refinance is low-risk financially.")

    if comparison["Refi_Beneficial"]:
        st.success("✅ **Refinancing appears to be BENEFICIAL under these assumptions.**")
    else:
        st.warning("⚠️ **Refinancing may NOT be beneficial under these assumptions.**")

    rec_cols = st.columns(2)
    with rec_cols[0]:
        if reasons_for:
            st.markdown("**Reasons to refinance:**")
            for r in reasons_for:
                st.markdown(f"- {r}")
    with rec_cols[1]:
        if reasons_against:
            st.markdown("**Reasons to consider staying:**")
            for r in reasons_against:
                st.markdown(f"- {r}")

    st.markdown("---")
    st.caption(
        "⚠️ This tool is for informational and educational purposes only. "
        "It does not constitute financial, tax, or legal advice. "
        "Consult a licensed mortgage professional before making any decisions."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
