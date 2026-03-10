"""
Mortgage Amortization & Refinance Comparison Tool
==================================================
A production-quality Streamlit app for mortgage brokers and borrowers to:
  - Calculate full amortization schedules (current and refinance scenarios)
  - Compare lifetime costs, interest, and payoff timelines
  - Evaluate the impact of advanced, flexible extra payment strategies
  - Model irregular and conditional extra payment behaviour:
      * Fixed monthly extra starting at a chosen month (optionally ending at another)
      * One-time lump sum payments at multiple selected payment months
      * Constant total monthly payment target (e.g. always pay $3,200/month)
      * Combination strategies
  - Compare multiple what-if scenarios side by side
  - Generate interactive charts and exportable tables

Timing note
-----------
All payment-period numbers (month numbers) refer to *monthly* mortgage payment
periods.  Some users (particularly those with biweekly payment habits) may
refer to "week 10" or "week 19" when they actually mean payment period 10 or 19.
This app treats all period numbers as monthly payment numbers.
This assumption is noted in the UI and in relevant function docstrings.

Run with:  streamlit run app.py
"""

import math
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional, Tuple

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
# Payment Strategy Data Structures & Helper Functions
# ---------------------------------------------------------------------------

# Supported payment strategy modes
STRATEGY_MODES: Dict[str, str] = {
    "none": "No extra payments",
    "fixed_monthly": "Fixed monthly extra payment",
    "lump_sum": "Lump sum payment(s) only",
    "target_total": "Constant total monthly payment target",
    "combination": "Combination: fixed monthly + lump sums",
}


def build_payment_strategy(
    mode: str = "none",
    fixed_monthly_extra: float = 0.0,
    fixed_start_month: int = 1,
    fixed_end_month: Optional[int] = None,
    lump_sums: Optional[List[Dict[str, Any]]] = None,
    target_total_payment: float = 0.0,
    target_includes_escrow: bool = False,
) -> Dict[str, Any]:
    """
    Build a payment strategy dict that can be passed to the amortization engine.

    Parameters
    ----------
    mode                 : strategy mode key (see STRATEGY_MODES)
    fixed_monthly_extra  : additional principal per month (used in fixed_monthly / combination)
    fixed_start_month    : 1-based payment month at which fixed extra begins
                           (relative to this schedule segment).
                           Timing note: month 19 = 19th monthly payment period.
    fixed_end_month      : 1-based month at which fixed extra stops (None = no end)
    lump_sums            : list of {"month": int, "amount": float}
    target_total_payment : desired total P&I outflow per month (target_total mode)
    target_includes_escrow: if True, target includes PMI + tax/insurance in its
                            calculation; otherwise targets P&I only

    Returns
    -------
    dict — strategy object for use in build_amortization_schedule()
    """
    return {
        "mode": mode,
        "fixed_monthly_extra": float(fixed_monthly_extra),
        "fixed_start_month": int(fixed_start_month),
        "fixed_end_month": int(fixed_end_month) if fixed_end_month is not None else None,
        "lump_sums": lump_sums if lump_sums is not None else [],
        "target_total_payment": float(target_total_payment),
        "target_includes_escrow": bool(target_includes_escrow),
    }


def parse_lump_sum_inputs(
    text: str, max_month: int = 600
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse lump sum input text into a list of {"month": int, "amount": float} dicts.

    Accepted formats (comma- or newline-separated):
      ``10:1000, 20:1500``   (colon separator)
      ``10=1000, 20=1500``   (equals separator)
      ``10 1000``            (space separator)

    Parameters
    ----------
    text      : raw text input from the user
    max_month : largest valid month number

    Returns
    -------
    (lump_sums, errors) where errors is a list of human-readable problem strings
    """
    if not text or not text.strip():
        return [], []

    lump_sums: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_months: set = set()

    entries = [e.strip() for e in text.replace("\n", ",").split(",") if e.strip()]

    for entry in entries:
        parsed = False
        for sep in [":", "=", " "]:
            if sep in entry:
                parts = entry.split(sep, 1)
                parsed = True
                break
        if not parsed:
            errors.append(f"Cannot parse '{entry}' — use format month:amount (e.g. 10:1000)")
            continue

        try:
            month = int(parts[0].strip())
            amount = float(parts[1].strip().replace("$", "").replace(",", ""))
        except (ValueError, IndexError):
            errors.append(
                f"Cannot parse '{entry}' — month must be an integer, amount must be a number"
            )
            continue

        if month < 1:
            errors.append(f"Month {month} is invalid — must be ≥ 1")
            continue
        if month > max_month:
            errors.append(f"Month {month} exceeds maximum loan term ({max_month} months)")
            continue
        if amount < 0:
            errors.append(f"Amount ${amount:,.2f} is negative — must be ≥ $0")
            continue
        if month in seen_months:
            errors.append(f"Duplicate month {month} — only the first occurrence is used")
            continue

        seen_months.add(month)
        lump_sums.append({"month": month, "amount": amount})

    lump_sums.sort(key=lambda x: x["month"])
    return lump_sums, errors


def validate_payment_strategy(
    strategy: Dict[str, Any],
    scheduled_pi: float,
    loan_term_months: int,
) -> List[str]:
    """
    Validate a payment strategy and return a list of human-readable warning strings.
    An empty list means the strategy is valid.
    """
    warnings: List[str] = []
    mode = strategy.get("mode", "none")

    if mode in ("fixed_monthly", "combination"):
        start = strategy.get("fixed_start_month", 1)
        end = strategy.get("fixed_end_month")
        if start < 1:
            warnings.append("Fixed start month must be ≥ 1")
        if end is not None and end < start:
            warnings.append(f"Fixed end month ({end}) must be ≥ start month ({start})")
        if end is not None and end > loan_term_months:
            warnings.append(
                f"Fixed end month ({end}) exceeds loan term ({loan_term_months} months)"
            )

    if mode in ("lump_sum", "combination"):
        for ls in strategy.get("lump_sums", []):
            if ls["month"] > loan_term_months:
                warnings.append(
                    f"Lump sum at month {ls['month']} is beyond loan term "
                    f"({loan_term_months} months) and will not apply"
                )

    if mode == "target_total":
        target = strategy.get("target_total_payment", 0.0)
        if 0 < target < scheduled_pi:
            warnings.append(
                f"Target payment {fmt_currency(target)} is less than the scheduled "
                f"P&I {fmt_currency(scheduled_pi)} — extra payment will be $0 each month"
            )

    return warnings


def get_extra_payment_for_month(
    month: int,
    strategy: Dict[str, Any],
    scheduled_pi: float,
    pmi: float,
    tax_ins: float,
) -> Tuple[float, float, float]:
    """
    Compute the three extra-payment components for a given payment month.

    Timing note: ``month`` is the 1-based payment number *within this schedule
    segment* (not an absolute calendar month).  Month 1 = first payment of this
    segment; month 19 = the 19th monthly payment.

    Parameters
    ----------
    month        : 1-based payment number within this schedule segment
    strategy     : strategy dict from build_payment_strategy()
    scheduled_pi : scheduled P&I payment amount for this month
    pmi          : PMI charge for this month
    tax_ins      : tax + insurance charge for this month

    Returns
    -------
    (extra_fixed, extra_lump, extra_target)
      extra_fixed  — from fixed monthly strategy
      extra_lump   — from a lump sum event on this month
      extra_target — from target total payment mode
    """
    mode = strategy.get("mode", "none")
    extra_fixed = 0.0
    extra_lump = 0.0
    extra_target = 0.0

    # Fixed monthly extra component
    if mode in ("fixed_monthly", "combination"):
        start = strategy.get("fixed_start_month", 1)
        end = strategy.get("fixed_end_month")
        if month >= start and (end is None or month <= end):
            extra_fixed = float(strategy.get("fixed_monthly_extra", 0.0))

    # Lump sum component (at most one event per month, de-duped in parse step)
    if mode in ("lump_sum", "combination"):
        for ls in strategy.get("lump_sums", []):
            if ls["month"] == month:
                extra_lump += float(ls["amount"])

    # Target total payment component
    if mode == "target_total":
        target = float(strategy.get("target_total_payment", 0.0))
        if strategy.get("target_includes_escrow", False):
            # Target encompasses all cash outflow (P&I + PMI + tax/ins)
            required = scheduled_pi + pmi + tax_ins
        else:
            # Target is P&I only
            required = scheduled_pi
        if target > required:
            extra_target = target - required

    return extra_fixed, extra_lump, extra_target


def strategy_summary_text(strategy: Dict[str, Any]) -> str:
    """Return a concise, human-readable description of the payment strategy."""
    mode = strategy.get("mode", "none")
    if mode == "none":
        return "No extra payments"

    parts: List[str] = []

    if mode in ("fixed_monthly", "combination"):
        amt = strategy.get("fixed_monthly_extra", 0.0)
        start = strategy.get("fixed_start_month", 1)
        end = strategy.get("fixed_end_month")
        if end:
            parts.append(f"{fmt_currency(amt)}/month extra (months {start}–{end})")
        else:
            parts.append(f"{fmt_currency(amt)}/month extra starting month {start}")

    if mode in ("lump_sum", "combination"):
        lumps = strategy.get("lump_sums", [])
        if lumps:
            lump_strs = [
                f"month {ls['month']}: {fmt_currency(ls['amount'])}" for ls in lumps
            ]
            parts.append("Lump sums — " + ", ".join(lump_strs))
        else:
            parts.append("Lump sums: none entered")

    if mode == "target_total":
        target = strategy.get("target_total_payment", 0.0)
        incl = " (incl. escrow)" if strategy.get("target_includes_escrow") else " (P&I only)"
        parts.append(f"Target total {fmt_currency(target)}/month{incl}")

    return " + ".join(parts) if parts else STRATEGY_MODES.get(mode, mode)


# ---------------------------------------------------------------------------
# Core calculation helpers
# ---------------------------------------------------------------------------

def monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Calculate the fixed monthly P&I payment using the standard amortization formula.

    M = P * [r(1+r)^n] / [(1+r)^n - 1]

    Parameters
    ----------
    principal   : loan amount in dollars
    annual_rate : annual interest rate as a percentage (e.g. 6.5 for 6.5 %)
    term_months : loan term in months

    Returns
    -------
    float : monthly P&I payment
    """
    if principal <= 0 or term_months <= 0:
        return 0.0
    if annual_rate == 0:
        # Zero-interest edge case
        return principal / term_months
    r = annual_rate / 100 / 12
    return principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)


def build_amortization_schedule(
    principal: float,
    annual_rate: float,
    term_months: int,
    start_date: date,
    property_value: float,
    pmi_rate_annual: float,
    tax_insurance_annual_pct: float,
    strategy: Optional[Dict[str, Any]] = None,
    starting_month_number: int = 1,
) -> pd.DataFrame:
    """
    Build a month-by-month amortization schedule as a pandas DataFrame.

    PMI is charged when the current balance > 80 % of property_value and
    is automatically removed once that threshold is crossed.

    Extra payments are driven entirely by the ``strategy`` dict; see
    build_payment_strategy() for the supported modes.

    Timing note: ``month`` inside this function is the 1-based payment number
    within *this* schedule segment.  Month 1 = first payment.  ``starting_month_number``
    offsets the ``Payment_Number`` column for combined pre/post-refi schedules.

    Parameters
    ----------
    principal               : opening loan balance
    annual_rate             : annual interest rate (%)
    term_months             : loan term in months
    start_date              : first payment date
    property_value          : current appraised value (for LTV / PMI)
    pmi_rate_annual         : annual PMI rate (%)
    tax_insurance_annual_pct: annual tax + insurance as % of property value
    strategy                : payment strategy dict; None → no extra payments
    starting_month_number   : first Payment_Number value (for combined schedules)

    Returns
    -------
    pd.DataFrame with columns:
        Payment_Number, Payment_Date, Beginning_Balance, Scheduled_Payment,
        Principal, Interest,
        Extra_Payment_Fixed, Extra_Payment_Lump, Extra_Payment_Target,
        Total_Extra_Payment, Extra_Payment (alias for Total_Extra_Payment),
        Ending_Balance, PMI, Tax_Insurance, Total_Monthly_Outflow,
        Cumulative_Interest, Cumulative_Principal,
        Cumulative_Extra_Payments, Cumulative_Total_Paid
    """
    if principal <= 0 or term_months <= 0:
        return pd.DataFrame()

    if strategy is None:
        strategy = build_payment_strategy(mode="none")

    r = annual_rate / 100 / 12
    scheduled_pi = monthly_payment(principal, annual_rate, term_months)
    monthly_tax_ins = property_value * (tax_insurance_annual_pct / 100) / 12

    rows: List[Dict[str, Any]] = []
    balance = principal
    cum_interest = 0.0
    cum_principal = 0.0
    cum_extra = 0.0
    cum_total = 0.0

    for i in range(1, term_months + 1):
        if balance <= 0:
            break

        payment_number = starting_month_number + i - 1
        payment_date = start_date + relativedelta(months=i - 1)
        beginning_balance = balance

        # Interest accrued this month
        interest = beginning_balance * r if annual_rate > 0 else 0.0

        # Scheduled principal portion (floor at 0 for floating-point safety)
        principal_portion = max(0.0, scheduled_pi - interest)

        # PMI — charged when LTV > 80 %; based on beginning balance
        ltv = beginning_balance / property_value if property_value > 0 else 0.0
        pmi = (beginning_balance * (pmi_rate_annual / 100) / 12) if ltv > 0.80 else 0.0

        # Extra payments for this month
        # 'i' is the local month index (1-based within this segment)
        extra_fixed, extra_lump, extra_target = get_extra_payment_for_month(
            month=i,
            strategy=strategy,
            scheduled_pi=scheduled_pi,
            pmi=pmi,
            tax_ins=monthly_tax_ins,
        )
        total_extra = extra_fixed + extra_lump + extra_target

        # Cap total principal reduction so balance never drops below zero
        total_principal = principal_portion + total_extra
        if beginning_balance - total_principal < 0:
            available_extra = max(0.0, beginning_balance - principal_portion)
            if total_extra > 0:
                scale = available_extra / total_extra
                extra_fixed *= scale
                extra_lump *= scale
                extra_target *= scale
                total_extra = available_extra
            else:
                total_extra = 0.0
            total_principal = principal_portion + total_extra

        ending_balance = max(0.0, beginning_balance - total_principal)

        # Actual payment (may be less than scheduled on the final month)
        actual_payment = min(scheduled_pi, beginning_balance + interest)

        cum_interest += interest
        cum_principal += principal_portion + total_extra
        cum_extra += total_extra
        cum_total += actual_payment + total_extra + pmi + monthly_tax_ins

        rows.append(
            {
                "Payment_Number": payment_number,
                "Payment_Date": payment_date,
                "Beginning_Balance": round(beginning_balance, 2),
                "Scheduled_Payment": round(scheduled_pi, 2),
                "Principal": round(principal_portion, 2),
                "Interest": round(interest, 2),
                "Extra_Payment_Fixed": round(extra_fixed, 2),
                "Extra_Payment_Lump": round(extra_lump, 2),
                "Extra_Payment_Target": round(extra_target, 2),
                "Total_Extra_Payment": round(total_extra, 2),
                "Extra_Payment": round(total_extra, 2),  # backward-compat alias
                "Ending_Balance": round(ending_balance, 2),
                "PMI": round(pmi, 2),
                "Tax_Insurance": round(monthly_tax_ins, 2),
                "Total_Monthly_Outflow": round(
                    actual_payment + total_extra + pmi + monthly_tax_ins, 2
                ),
                "Cumulative_Interest": round(cum_interest, 2),
                "Cumulative_Principal": round(cum_principal, 2),
                "Cumulative_Extra_Payments": round(cum_extra, 2),
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
    strategy: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build the combined amortization schedule for the refinance scenario:
      - Pre-refi months  : taken directly from original_schedule_pre_refi (rows ≤ refi_month)
      - Post-refi months : new schedule from remaining principal, using ``strategy``

    Closing costs can be rolled into the new loan balance or treated as upfront cash.

    Parameters
    ----------
    original_schedule_pre_refi : pre-refi schedule (no extra payments applied)
    refi_month                 : month number at which refinancing occurs
    refi_annual_rate           : new interest rate (%)
    refi_term_months           : new loan term in months
    property_value             : appraised property value (unchanged)
    pmi_rate_annual            : PMI rate for the new loan (%)
    tax_insurance_annual_pct   : annual tax + insurance as % of property value
    closing_costs              : dollar amount of closing costs
    roll_closing_costs         : if True, closing costs are added to new principal
    strategy                   : payment strategy for the POST-refi period

    Returns
    -------
    pd.DataFrame — combined pre + post refi schedule (same columns as
                   build_amortization_schedule output)
    """
    if original_schedule_pre_refi.empty:
        return pd.DataFrame()

    pre_refi = original_schedule_pre_refi[
        original_schedule_pre_refi["Payment_Number"] <= refi_month
    ].copy()

    if pre_refi.empty:
        return pd.DataFrame()

    remaining_balance = pre_refi.iloc[-1]["Ending_Balance"]
    refi_start_date = pre_refi.iloc[-1]["Payment_Date"] + relativedelta(months=1)

    new_principal = remaining_balance
    if roll_closing_costs:
        new_principal += closing_costs

    post_refi = build_amortization_schedule(
        principal=new_principal,
        annual_rate=refi_annual_rate,
        term_months=refi_term_months,
        start_date=refi_start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate_annual,
        tax_insurance_annual_pct=tax_insurance_annual_pct,
        strategy=strategy,
        starting_month_number=refi_month + 1,
    )

    if post_refi.empty:
        return pre_refi

    # Re-accumulate cumulative columns across the combined schedule
    pre_cum_int = pre_refi.iloc[-1]["Cumulative_Interest"]
    pre_cum_prin = pre_refi.iloc[-1]["Cumulative_Principal"]
    pre_cum_extra = pre_refi.iloc[-1].get("Cumulative_Extra_Payments", 0.0)
    pre_cum_total = pre_refi.iloc[-1]["Cumulative_Total_Paid"]

    post_refi = post_refi.copy()
    post_refi["Cumulative_Interest"] = (
        post_refi["Cumulative_Interest"] + pre_cum_int
    ).round(2)
    post_refi["Cumulative_Principal"] = (
        post_refi["Cumulative_Principal"] + pre_cum_prin
    ).round(2)
    post_refi["Cumulative_Extra_Payments"] = (
        post_refi["Cumulative_Extra_Payments"] + pre_cum_extra
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
    schedule             : output of build_amortization_schedule or build_refinance_schedule
    original_term_months : contractual term of the original loan in months
    label                : scenario name (e.g. "Current Loan", "Refinance")

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

    # Total extra payments
    if "Cumulative_Extra_Payments" in schedule.columns:
        total_extra = round(schedule["Cumulative_Extra_Payments"].iloc[-1], 2)
    else:
        total_extra = 0.0

    # PMI drop-off month
    pmi_rows = schedule[schedule["PMI"] > 0]
    if pmi_rows.empty:
        pmi_dropoff: str = "No PMI"
    else:
        pmi_dropoff = f"Month {int(pmi_rows['Payment_Number'].max()) + 1}"

    # First month an extra payment was made
    extra_col = "Total_Extra_Payment" if "Total_Extra_Payment" in schedule.columns else "Extra_Payment"
    if extra_col in schedule.columns:
        extra_rows = schedule[schedule[extra_col] > 0]
        first_extra_month: Optional[int] = (
            int(extra_rows["Payment_Number"].min()) if not extra_rows.empty else None
        )
    else:
        first_extra_month = None

    return {
        "Scenario": label,
        "Original_Term_Months": original_term_months,
        "Actual_Payoff_Month": actual_payoff_month,
        "Time_Saved_Months": time_saved_months,
        "Time_Saved_Years": time_saved_years,
        "Actual_Interest_Paid": total_interest,
        "Total_Paid": total_paid,
        "Total_Extra_Payments": total_extra,
        "PMI_Dropoff": pmi_dropoff,
        "First_Extra_Month": first_extra_month,
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
    Compute the comparison metrics between staying in the current loan and refinancing.

    Parameters
    ----------
    summary_current          : summarize_schedule output for the current loan scenario
    summary_refi             : summarize_schedule output for the refi scenario
    baseline_interest_current: total interest on current loan with no extra payments
    baseline_interest_refi   : total interest on refi loan with no extra payments
    closing_costs            : refinance closing costs
    roll_closing_costs       : True = rolled into loan; False = upfront cash

    Returns
    -------
    dict with comparison statistics
    """
    interest_diff = (
        summary_current["Actual_Interest_Paid"] - summary_refi["Actual_Interest_Paid"]
    )
    total_paid_diff = summary_current["Total_Paid"] - summary_refi["Total_Paid"]
    if not roll_closing_costs:
        total_paid_diff -= closing_costs

    payoff_diff_months = (
        summary_current["Actual_Payoff_Month"] - summary_refi["Actual_Payoff_Month"]
    )

    # Break-even estimate: months until cumulative savings cover closing costs
    breakeven_month: int | str = "N/A"
    if closing_costs > 0:
        monthly_saving_estimate = (
            interest_diff / summary_refi["Actual_Payoff_Month"]
            if summary_refi.get("Actual_Payoff_Month", 0) > 0
            else 0.0
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
        "Interest_Saved_Pct": (
            round(interest_diff / baseline_interest_current * 100, 2)
            if baseline_interest_current > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _scenario_traces(
    scenarios: List[Tuple[str, pd.DataFrame]],
    y_col: str,
    hover_label: str,
    colors: Optional[List[str]] = None,
    dashes: Optional[List[str]] = None,
) -> List[go.Scatter]:
    """Build Scatter traces for a list of (name, DataFrame) scenario pairs."""
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]
    default_dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    traces = []
    for idx, (name, df) in enumerate(scenarios):
        if df.empty or y_col not in df.columns:
            continue
        color = (colors or default_colors)[idx % len(default_colors)]
        dash = (dashes or default_dashes)[idx % len(default_dashes)]
        traces.append(
            go.Scatter(
                x=df["Payment_Number"],
                y=df[y_col],
                mode="lines",
                name=name,
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=(
                    f"Month %{{x}}<br>{hover_label}: $%{{y:,.2f}}<extra></extra>"
                ),
            )
        )
    return traces


def _base_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    """Return a common Plotly layout dict."""
    return dict(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )


def chart_remaining_balance(
    scenarios: List[Tuple[str, pd.DataFrame]],
) -> go.Figure:
    """Line chart: Remaining loan balance over time for multiple scenarios."""
    fig = go.Figure(data=_scenario_traces(scenarios, "Ending_Balance", "Balance"))
    fig.update_layout(**_base_layout(
        "Remaining Loan Balance Over Time", "Payment Month", "Remaining Balance ($)"
    ))
    return fig


def chart_cumulative_interest(
    scenarios: List[Tuple[str, pd.DataFrame]],
) -> go.Figure:
    """Line chart: Cumulative interest paid over time for multiple scenarios."""
    fig = go.Figure(
        data=_scenario_traces(scenarios, "Cumulative_Interest", "Cum. Interest")
    )
    fig.update_layout(**_base_layout(
        "Cumulative Interest Paid Over Time", "Payment Month", "Cumulative Interest ($)"
    ))
    return fig


def chart_cumulative_extra_payments(
    scenarios: List[Tuple[str, pd.DataFrame]],
) -> go.Figure:
    """Line chart: Cumulative extra principal payments over time."""
    fig = go.Figure(
        data=_scenario_traces(
            scenarios, "Cumulative_Extra_Payments", "Cum. Extra Paid"
        )
    )
    fig.update_layout(**_base_layout(
        "Cumulative Extra Payments Over Time",
        "Payment Month",
        "Cumulative Extra Payments ($)",
    ))
    return fig


def chart_monthly_total_payment(
    scenarios: List[Tuple[str, pd.DataFrame]],
) -> go.Figure:
    """Line chart: Total monthly outflow over time for multiple scenarios."""
    fig = go.Figure(
        data=_scenario_traces(scenarios, "Total_Monthly_Outflow", "Total Payment")
    )
    fig.update_layout(**_base_layout(
        "Monthly Total Payment Over Time", "Payment Month", "Monthly Total Payment ($)"
    ))
    return fig


def chart_equity_growth(
    scenarios: List[Tuple[str, pd.DataFrame]],
    property_value: float,
) -> go.Figure:
    """Line chart: Equity growth over time for multiple scenarios."""
    eq_scenarios: List[Tuple[str, pd.DataFrame]] = []
    for name, df in scenarios:
        if df.empty:
            continue
        df_eq = df.copy()
        df_eq["Equity"] = property_value - df_eq["Ending_Balance"]
        eq_scenarios.append((name, df_eq))

    fig = go.Figure(data=_scenario_traces(eq_scenarios, "Equity", "Equity"))
    fig.add_hline(
        y=property_value,
        line_dash="dot",
        annotation_text="Full Equity",
        annotation_position="bottom right",
        line_color="green",
    )
    fig.update_layout(**_base_layout(
        "Home Equity Growth Over Time", "Payment Month", "Equity ($)"
    ))
    return fig


def chart_payment_composition(
    df: pd.DataFrame, title: str = "Payment Composition"
) -> go.Figure:
    """Stacked bar chart: monthly payment breakdown (principal, interest, extras, PMI, tax/ins)."""
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
    if "Total_Extra_Payment" in df.columns and df["Total_Extra_Payment"].sum() > 0:
        fig.add_trace(
            go.Bar(
                x=df["Payment_Number"],
                y=df["Total_Extra_Payment"],
                name="Extra Payment",
                marker_color="#17becf",
                hovertemplate="Month %{x}<br>Extra: $%{y:,.2f}<extra></extra>",
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


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def render_strategy_ui(
    prefix: str,
    loan_term_months: int,
    default_mode: str = "none",
) -> Dict[str, Any]:
    """
    Render the Advanced Extra Payment Strategy controls for one scenario.

    Parameters
    ----------
    prefix           : unique Streamlit widget-key prefix (e.g. "cur", "refi", "wi")
    loan_term_months : maximum valid payment month for this scenario
    default_mode     : pre-selected strategy mode

    Returns
    -------
    strategy dict (from build_payment_strategy)
    """
    mode = st.selectbox(
        "Extra payment mode",
        options=list(STRATEGY_MODES.keys()),
        format_func=lambda k: STRATEGY_MODES[k],
        index=list(STRATEGY_MODES.keys()).index(default_mode),
        key=f"{prefix}_mode",
        help="Choose how extra payments are applied to this loan scenario.",
    )

    fixed_monthly_extra = 0.0
    fixed_start_month = 1
    fixed_end_month: Optional[int] = None
    lump_sums: List[Dict[str, Any]] = []
    target_total_payment = 0.0
    target_includes_escrow = False

    # ---- Fixed monthly extra section ----
    if mode in ("fixed_monthly", "combination"):
        st.markdown("**Fixed Monthly Extra Payment**")
        fixed_monthly_extra = st.number_input(
            "Extra amount per month ($)",
            min_value=0.0,
            max_value=100_000.0,
            value=500.0,
            step=50.0,
            format="%.2f",
            key=f"{prefix}_fixed_amt",
            help="Additional principal payment each month.",
        )
        col_start, col_end = st.columns(2)
        with col_start:
            fixed_start_month = int(
                st.number_input(
                    "Start extra payments in month",
                    min_value=1,
                    max_value=loan_term_months,
                    value=1,
                    step=1,
                    key=f"{prefix}_fixed_start",
                    help=(
                        "Payment month number when you begin making extra payments. "
                        "Month 1 = first payment period. "
                        "Example: enter 19 to start extra payments on the 19th monthly payment."
                    ),
                )
            )
        with col_end:
            use_end = st.checkbox(
                "Set an end month",
                value=False,
                key=f"{prefix}_use_end",
                help="Optional: stop making this fixed extra payment after a specific month.",
            )
            if use_end:
                fixed_end_month = int(
                    st.number_input(
                        "Stop extra payments after month",
                        min_value=fixed_start_month,
                        max_value=loan_term_months,
                        value=min(fixed_start_month + 11, loan_term_months),
                        step=1,
                        key=f"{prefix}_fixed_end",
                    )
                )

    # ---- Lump sum section ----
    if mode in ("lump_sum", "combination"):
        st.markdown("**Lump Sum Payment Events**")
        st.caption(
            "Enter one-time extra payments as: `month:amount, month:amount`\n\n"
            "Example: `10:1000, 20:1500, 36:5000`\n\n"
            "Month numbers are monthly payment periods (period 10 = 10th monthly payment).  "
            "Accepted separators: `:` `=` or space."
        )
        lump_text = st.text_area(
            "Lump sum events",
            value="",
            placeholder="10:1000, 20:1500, 36:5000",
            key=f"{prefix}_lumps",
            height=80,
        )
        lump_sums, lump_errors = parse_lump_sum_inputs(
            lump_text, max_month=loan_term_months
        )
        for err in lump_errors:
            st.warning(f"⚠️ {err}")
        if lump_sums:
            st.success(
                "✅ Parsed: "
                + ", ".join(
                    f"Month {ls['month']}: {fmt_currency(ls['amount'])}"
                    for ls in lump_sums
                )
            )

    # ---- Target total payment section ----
    if mode == "target_total":
        st.markdown("**Constant Total Monthly Payment Target**")
        st.caption(
            "You pay a fixed total amount each month.  "
            "Extra principal = target − scheduled P&I (or total outflow if escrow is included).  "
            "If the target is less than the scheduled P&I, no extra payment is applied."
        )
        target_total_payment = st.number_input(
            "Target total monthly payment ($)",
            min_value=0.0,
            max_value=100_000.0,
            value=3_200.0,
            step=50.0,
            format="%.2f",
            key=f"{prefix}_target",
            help="Example: 3200 means you always put $3,200 toward this loan each month.",
        )
        target_includes_escrow = st.checkbox(
            "Target includes PMI, tax & insurance",
            value=False,
            key=f"{prefix}_target_escrow",
            help=(
                "Checked: extra = target − (P&I + PMI + tax/ins).  "
                "Unchecked: extra = target − P&I only."
            ),
        )

    return build_payment_strategy(
        mode=mode,
        fixed_monthly_extra=fixed_monthly_extra,
        fixed_start_month=fixed_start_month,
        fixed_end_month=fixed_end_month,
        lump_sums=lump_sums,
        target_total_payment=target_total_payment,
        target_includes_escrow=target_includes_escrow,
    )


# ---------------------------------------------------------------------------
# Sidebar / Input section
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """
    Render all loan-parameter input controls and return a validated inputs dict.
    Extra-payment strategies are rendered in the main area (not the sidebar).
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
            help="Current appraised property value (used for LTV and PMI).",
        )
        pmi_rate = st.number_input(
            "PMI Rate (% per year)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help="Annual PMI rate as a % of outstanding balance. Removed automatically when LTV ≤ 80 %.",
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
            help="The payment month at which you refinance (e.g. 36 = after 3 years).",
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
        )
        roll_closing_costs = st.checkbox(
            "Roll closing costs into new loan balance",
            value=True,
            help=(
                "Checked: closing costs are added to the new loan principal. "
                "Unchecked: treated as an upfront cash expense."
            ),
        )

    # ---- Input validation ----
    errors: List[str] = []
    if loan_amount <= 0:
        errors.append("Loan amount must be greater than zero.")
    if property_value <= 0:
        errors.append("Property value must be greater than zero.")
    if refi_month >= loan_term_years * 12:
        errors.append(
            f"Refinance month ({refi_month}) must be before the original loan term "
            f"({int(loan_term_years * 12)} months)."
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
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏠 Mortgage Amortization & Refinance Comparison Tool")
    st.caption(
        "Compare your current loan against a refinance scenario with advanced "
        "extra payment strategies. For informational purposes only."
    )

    inputs = render_sidebar()

    # ---- Validation errors ----
    if inputs["errors"]:
        for err in inputs["errors"]:
            st.error(f"⚠️ {err}")
        st.stop()

    # ---- Unpack loan parameters ----
    loan_amount: float = inputs["loan_amount"]
    current_rate: float = inputs["current_rate"]
    loan_term_months: int = inputs["loan_term_months"]
    start_date: date = inputs["start_date"]
    property_value: float = inputs["property_value"]
    pmi_rate: float = inputs["pmi_rate"]
    tax_ins_pct: float = inputs["tax_ins_pct"]
    refi_rate: float = inputs["refi_rate"]
    refi_month: int = inputs["refi_month"]
    refi_term_months: int = inputs["refi_term_months"]
    closing_costs: float = inputs["closing_costs"]
    roll_closing_costs: bool = inputs["roll_closing_costs"]

    # Computed monthly figures
    current_monthly_pi = monthly_payment(loan_amount, current_rate, loan_term_months)
    monthly_tax_ins = property_value * (tax_ins_pct / 100) / 12
    initial_pmi = (
        (loan_amount * (pmi_rate / 100) / 12)
        if (loan_amount / property_value) > 0.80
        else 0.0
    )
    current_total_monthly = current_monthly_pi + initial_pmi + monthly_tax_ins

    # Pre-compute the no-extras pre-refi baseline schedule (needed by both UI and calcs)
    df_pre_refi_original = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=current_rate,
        term_months=loan_term_months,
        start_date=start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        strategy=build_payment_strategy(mode="none"),
    )

    # Refi principal at the refinance month (no extras before refi for baseline display)
    row_at_refi = df_pre_refi_original[
        df_pre_refi_original["Payment_Number"] == refi_month
    ]
    refi_balance_at_refi = (
        float(row_at_refi["Ending_Balance"].values[0])
        if len(row_at_refi) > 0
        else loan_amount
    )
    new_refi_principal = refi_balance_at_refi + (closing_costs if roll_closing_costs else 0.0)
    refi_monthly_pi = monthly_payment(new_refi_principal, refi_rate, refi_term_months)
    refi_initial_pmi = (
        (new_refi_principal * (pmi_rate / 100) / 12)
        if (new_refi_principal / property_value) > 0.80
        else 0.0
    )
    refi_total_monthly = refi_monthly_pi + refi_initial_pmi + monthly_tax_ins

    # =========================================================================
    # SECTION 0 — Advanced Extra Payment Strategy (main area tabs)
    # =========================================================================
    section_header(
        "💰 Advanced Extra Payment Strategy",
        subtitle=(
            "Define independent payment strategies for each scenario. "
            "All month numbers are monthly mortgage payment periods "
            "(month 1 = first payment, month 19 = 19th payment, etc.)."
        ),
    )

    st.info(
        "📌 **Timing assumption:** Month numbers refer to your mortgage *payment period* "
        "numbers (monthly).  If you think in terms of 'week 10' or 'week 19,' treat those "
        "as payment months 10 and 19 respectively, unless you have a biweekly mortgage."
    )

    strat_tab_cur, strat_tab_refi, strat_tab_whatif = st.tabs([
        "📋 Current Loan Strategy",
        "🔄 Refinance Strategy",
        "🔍 What-If Scenario",
    ])

    with strat_tab_cur:
        st.markdown("### Payment strategy — staying in your current loan")
        strategy_current = render_strategy_ui(
            prefix="cur",
            loan_term_months=loan_term_months,
            default_mode="none",
        )
        st.markdown("**Strategy summary:**")
        st.success(f"📋 {strategy_summary_text(strategy_current)}")
        for w in validate_payment_strategy(strategy_current, current_monthly_pi, loan_term_months):
            st.warning(f"⚠️ {w}")

    with strat_tab_refi:
        st.markdown("### Payment strategy — post-refinance period")
        st.caption(
            "This strategy applies after the refinance date.  "
            "Month 1 here = the first payment on the *new* loan after refinancing."
        )
        st.info(
            f"Estimated new loan balance after refinancing at month {refi_month}: "
            f"**{fmt_currency(new_refi_principal)}** | "
            f"New P&I payment: **{fmt_currency(refi_monthly_pi)}/month**"
        )
        strategy_refi = render_strategy_ui(
            prefix="refi",
            loan_term_months=refi_term_months,
            default_mode="none",
        )
        st.markdown("**Strategy summary:**")
        st.success(f"📋 {strategy_summary_text(strategy_refi)}")
        for w in validate_payment_strategy(strategy_refi, refi_monthly_pi, refi_term_months):
            st.warning(f"⚠️ {w}")

    with strat_tab_whatif:
        st.markdown("### What-if scenario")
        st.caption(
            "Define an alternate payment strategy to compare against your main scenarios.  "
            "This adds a third line to all charts and a third column to the comparison table."
        )
        whatif_basis = st.radio(
            "What-if is based on:",
            ["Current loan (no refinance)", "Refinance scenario"],
            key="whatif_basis",
            horizontal=True,
        )
        wi_term = (
            loan_term_months
            if whatif_basis == "Current loan (no refinance)"
            else refi_term_months
        )
        strategy_whatif = render_strategy_ui(
            prefix="wi",
            loan_term_months=wi_term,
            default_mode="none",
        )
        st.markdown("**Strategy summary:**")
        st.success(f"📋 {strategy_summary_text(strategy_whatif)}")

    # =========================================================================
    # Build all amortization schedules
    # =========================================================================

    # Baseline (no extras) for reference interest calculation
    df_baseline = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=current_rate,
        term_months=loan_term_months,
        start_date=start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        strategy=build_payment_strategy(mode="none"),
    )
    baseline_total_interest = df_baseline["Interest"].sum()

    # Current loan with user's strategy
    df_current = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=current_rate,
        term_months=loan_term_months,
        start_date=start_date,
        property_value=property_value,
        pmi_rate_annual=pmi_rate,
        tax_insurance_annual_pct=tax_ins_pct,
        strategy=strategy_current,
    )

    # Refinance scenario (pre-refi: no extras; post-refi: strategy_refi)
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
        strategy=strategy_refi,
    )

    # What-if scenario
    if whatif_basis == "Current loan (no refinance)":
        df_whatif = build_amortization_schedule(
            principal=loan_amount,
            annual_rate=current_rate,
            term_months=loan_term_months,
            start_date=start_date,
            property_value=property_value,
            pmi_rate_annual=pmi_rate,
            tax_insurance_annual_pct=tax_ins_pct,
            strategy=strategy_whatif,
        )
        whatif_label = "What-If (Current)"
        whatif_term = loan_term_months
    else:
        df_whatif = build_refinance_schedule(
            original_schedule_pre_refi=df_pre_refi_original,
            refi_month=refi_month,
            refi_annual_rate=refi_rate,
            refi_term_months=refi_term_months,
            property_value=property_value,
            pmi_rate_annual=pmi_rate,
            tax_insurance_annual_pct=tax_ins_pct,
            closing_costs=closing_costs,
            roll_closing_costs=roll_closing_costs,
            strategy=strategy_whatif,
        )
        whatif_label = "What-If (Refi)"
        whatif_term = loan_term_months  # use original term for time-saved calc

    # Baseline refi interest (no extras) for % savings calc
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
        strategy=build_payment_strategy(mode="none"),
    )
    baseline_refi_interest = df_refi_baseline["Interest"].sum()

    # ---- Summaries ----
    summary_current = summarize_schedule(df_current, loan_term_months, "Current Loan")
    summary_refi = summarize_schedule(df_refi, loan_term_months, "Refinance Scenario")
    summary_whatif = summarize_schedule(df_whatif, whatif_term, whatif_label)

    comparison = compare_scenarios(
        summary_current=summary_current,
        summary_refi=summary_refi,
        baseline_interest_current=baseline_total_interest,
        baseline_interest_refi=baseline_refi_interest,
        closing_costs=closing_costs,
        roll_closing_costs=roll_closing_costs,
    )
    bkeven = comparison["Breakeven_Month"]

    # =========================================================================
    # SECTION 1 — Summary KPI Cards
    # =========================================================================
    section_header("📊 Summary", "Key metrics at a glance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current Monthly P&I",
            fmt_currency(current_monthly_pi),
            help="Monthly principal + interest payment (current loan, month 1).",
        )
        st.metric(
            "Current Lifetime Interest",
            fmt_currency(summary_current["Actual_Interest_Paid"]),
        )
    with col2:
        st.metric(
            "Refi Monthly P&I",
            fmt_currency(refi_monthly_pi),
            delta=fmt_currency(refi_monthly_pi - current_monthly_pi),
            delta_color="inverse",
            help="Monthly P&I after refinancing.",
        )
        st.metric(
            "Refi Lifetime Interest",
            fmt_currency(summary_refi["Actual_Interest_Paid"]),
            delta=fmt_currency(
                summary_refi["Actual_Interest_Paid"] - summary_current["Actual_Interest_Paid"]
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
            delta=(
                "Refi pays off sooner"
                if payoff_diff > 0
                else ("Same" if payoff_diff == 0 else "Refi pays off later")
            ),
            delta_color="normal" if payoff_diff >= 0 else "inverse",
        )
        st.metric(
            "Est. Break-Even",
            str(bkeven) if isinstance(bkeven, str) else f"Month {bkeven}",
            help="Estimated month at which cumulative savings exceed closing costs.",
        )

    st.markdown("---")
    info_cols = st.columns(4)
    info_cols[0].info(
        f"**New Refi Balance:** {fmt_currency(new_refi_principal)}\n\n"
        f"*(at month {refi_month})*"
    )
    info_cols[1].info(
        f"**PMI Drops Off**\n\n"
        f"Current: {summary_current.get('PMI_Dropoff', 'N/A')}\n\n"
        f"Refi: {summary_refi.get('PMI_Dropoff', 'N/A')}"
    )
    info_cols[2].info(
        f"**Payoff Month**\n\n"
        f"Current: Month {summary_current['Actual_Payoff_Month']}\n\n"
        f"Refi: Month {summary_refi['Actual_Payoff_Month']}"
    )
    info_cols[3].info(
        f"**Baseline Interest (no extras)**\n\n"
        f"Current: {fmt_currency(baseline_total_interest)}\n\n"
        f"Refi: {fmt_currency(baseline_refi_interest)}"
    )

    # =========================================================================
    # SECTION 2 — Scenario Comparison Table
    # =========================================================================
    section_header("🔍 Scenario Comparison", "Side-by-side loan outcome comparison")

    def _none_str(v: Any) -> str:
        return str(v) if v is not None else "N/A"

    wi_payoff = summary_whatif.get("Actual_Payoff_Month", 0)
    wi_time_saved = summary_whatif.get("Time_Saved_Months", 0)

    comparison_data = {
        "Metric": [
            "Scenario",
            "Original Term",
            "Actual Payoff Month",
            "Time Saved (months)",
            "Time Saved (years)",
            "Actual Interest Paid",
            "Total Paid (P+I+PMI+T&I)",
            "Total Extra Payments",
            "Monthly P&I Payment",
            "Monthly Total Payment (est.)",
            "PMI Drop-Off",
            "First Extra Payment Month",
            "% Interest Saved vs Baseline",
        ],
        "Current Loan": [
            "Stay in Current Loan",
            fmt_months(summary_current["Original_Term_Months"]),
            fmt_months(summary_current["Actual_Payoff_Month"]),
            f"{summary_current['Time_Saved_Months']} months",
            f"{summary_current['Time_Saved_Years']} yrs",
            fmt_currency(summary_current["Actual_Interest_Paid"]),
            fmt_currency(summary_current["Total_Paid"]),
            fmt_currency(summary_current.get("Total_Extra_Payments", 0.0)),
            fmt_currency(current_monthly_pi),
            fmt_currency(current_total_monthly),
            summary_current.get("PMI_Dropoff", "N/A"),
            _none_str(summary_current.get("First_Extra_Month")),
            fmt_percent(
                (baseline_total_interest - summary_current["Actual_Interest_Paid"])
                / baseline_total_interest * 100
                if baseline_total_interest > 0
                else 0.0
            ),
        ],
        "Refinance Scenario": [
            f"Refinance at Month {refi_month}",
            fmt_months(summary_refi["Original_Term_Months"]),
            fmt_months(summary_refi["Actual_Payoff_Month"]),
            f"{summary_refi['Time_Saved_Months']} months",
            f"{summary_refi['Time_Saved_Years']} yrs",
            fmt_currency(summary_refi["Actual_Interest_Paid"]),
            fmt_currency(summary_refi["Total_Paid"]),
            fmt_currency(summary_refi.get("Total_Extra_Payments", 0.0)),
            fmt_currency(refi_monthly_pi),
            fmt_currency(refi_total_monthly),
            summary_refi.get("PMI_Dropoff", "N/A"),
            _none_str(summary_refi.get("First_Extra_Month")),
            fmt_percent(
                (baseline_refi_interest - summary_refi["Actual_Interest_Paid"])
                / baseline_refi_interest * 100
                if baseline_refi_interest > 0
                else 0.0
            ),
        ],
        whatif_label: [
            whatif_label,
            fmt_months(summary_whatif.get("Original_Term_Months", whatif_term)),
            fmt_months(wi_payoff) if wi_payoff else "N/A",
            f"{wi_time_saved} months",
            f"{round(wi_time_saved / 12, 2)} yrs",
            fmt_currency(summary_whatif.get("Actual_Interest_Paid", 0.0)),
            fmt_currency(summary_whatif.get("Total_Paid", 0.0)),
            fmt_currency(summary_whatif.get("Total_Extra_Payments", 0.0)),
            "N/A",
            "N/A",
            summary_whatif.get("PMI_Dropoff", "N/A"),
            _none_str(summary_whatif.get("First_Extra_Month")),
            "N/A",
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

    # Build scenario list; include what-if only when a non-trivial strategy was chosen
    chart_scenarios: List[Tuple[str, pd.DataFrame]] = [
        ("Current Loan", df_current),
        ("Refinance Scenario", df_refi),
    ]
    if strategy_whatif.get("mode", "none") != "none":
        chart_scenarios.append((whatif_label, df_whatif))

    (
        tab_bal,
        tab_int,
        tab_extra,
        tab_monthly,
        tab_comp_cur,
        tab_comp_refi,
        tab_equity,
    ) = st.tabs([
        "Remaining Balance",
        "Cumulative Interest",
        "Cumulative Extra Payments",
        "Monthly Total Payment",
        "Payment Composition — Current",
        "Payment Composition — Refi",
        "Equity Growth",
    ])

    with tab_bal:
        st.plotly_chart(chart_remaining_balance(chart_scenarios), use_container_width=True)

    with tab_int:
        st.plotly_chart(chart_cumulative_interest(chart_scenarios), use_container_width=True)

    with tab_extra:
        st.plotly_chart(
            chart_cumulative_extra_payments(chart_scenarios), use_container_width=True
        )

    with tab_monthly:
        st.plotly_chart(
            chart_monthly_total_payment(chart_scenarios), use_container_width=True
        )

    with tab_comp_cur:
        st.plotly_chart(
            chart_payment_composition(df_current, "Payment Composition — Current Loan"),
            use_container_width=True,
        )

    with tab_comp_refi:
        st.plotly_chart(
            chart_payment_composition(
                df_refi, "Payment Composition — Refinance Scenario"
            ),
            use_container_width=True,
        )

    with tab_equity:
        st.plotly_chart(
            chart_equity_growth(chart_scenarios, property_value),
            use_container_width=True,
        )

    # =========================================================================
    # SECTION 4 — Amortization Tables
    # =========================================================================
    section_header("📋 Amortization Tables", "Month-by-month payment detail")

    table_choices = ["Current Loan", "Refinance Scenario", whatif_label]
    table_choice = st.radio(
        "View schedule for:", table_choices, horizontal=True
    )

    def format_schedule_for_display(df: pd.DataFrame) -> pd.DataFrame:
        """Return a display-friendly version of the amortization schedule."""
        display = df.copy()
        display["Payment_Date"] = display["Payment_Date"].apply(
            lambda d: d.strftime("%b %Y") if hasattr(d, "strftime") else str(d)
        )
        currency_cols = [
            "Beginning_Balance",
            "Scheduled_Payment",
            "Principal",
            "Interest",
            "Extra_Payment_Fixed",
            "Extra_Payment_Lump",
            "Extra_Payment_Target",
            "Total_Extra_Payment",
            "Extra_Payment",
            "Ending_Balance",
            "PMI",
            "Tax_Insurance",
            "Total_Monthly_Outflow",
            "Cumulative_Interest",
            "Cumulative_Principal",
            "Cumulative_Extra_Payments",
            "Cumulative_Total_Paid",
        ]
        for col in currency_cols:
            if col in display.columns:
                display[col] = display[col].apply(lambda v: f"${v:,.2f}")
        return display

    schedule_map = {
        "Current Loan": df_current,
        "Refinance Scenario": df_refi,
        whatif_label: df_whatif,
    }
    selected_df = schedule_map[table_choice]
    st.caption(f"Showing {len(selected_df)} payment rows for {table_choice}.")
    st.dataframe(
        format_schedule_for_display(selected_df),
        use_container_width=True,
        height=400,
    )
    csv_data = selected_df.to_csv(index=False).encode("utf-8")
    safe_name = table_choice.lower().replace(" ", "_").replace("(", "").replace(")", "")
    st.download_button(
        f"⬇️ Download {table_choice} Schedule CSV",
        data=csv_data,
        file_name=f"{safe_name}_schedule.csv",
        mime="text/csv",
    )

    # =========================================================================
    # SECTION 5 — Recommendation
    # =========================================================================
    section_header("💡 Recommendation", "Data-based analysis of your refinance decision")

    reasons_for: List[str] = []
    reasons_against: List[str] = []

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
        "⚠️ **Assumptions & Disclaimers:**\n\n"
        "- All month numbers are monthly mortgage payment periods (month 1 = first payment, month 19 = 19th payment)\n"
        "- Extra payments are capped so the balance never drops below zero\n"
        "- Target payment mode targets P&I only unless 'includes escrow' is checked\n"
        "- PMI is removed automatically when LTV ≤ 80 %\n"
        "- This tool is for informational and educational purposes only.  "
        "It does not constitute financial, tax, or legal advice.  "
        "Consult a licensed mortgage professional before making any decisions."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
