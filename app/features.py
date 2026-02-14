import numpy as np

# ─────────────────────────────────────────
# ODE — STRESS VELOCITY (Fast Closed-Form)
# ─────────────────────────────────────────

def compute_stress_velocity(
    debt_burden,
    income_volatility,
    lending_app_score,
    savings_drawdown,
    salary_delay,
    autodebit_fail,
    cash_hoarding
):
    """
    F8: Stress Velocity
    dΩ/dt = α·D(t) + β·L(t) - γ·R(t)

    D(t) = Debt Pressure
    L(t) = Load Markers
    R(t) = Recovery Force
    """
    alpha, beta, gamma = 0.4, 0.35, 0.25

    D = debt_burden * 0.6 + income_volatility * 0.4        # Debt pressure
    L = cash_hoarding * 0.4 + salary_delay * 0.3 + autodebit_fail * 0.3  # Load markers
    R = max(0.01, 1.0 - savings_drawdown)                  # Recovery force

    # Stress velocity = net rate of stress accumulation
    velocity = alpha * D + beta * L - gamma * R

    return float(np.clip(velocity, 0.0, 0.12))


# ─────────────────────────────────────────
# PAYMENT REGULARITY ENTROPY (F9)
# Shannon entropy of payment timing
# ─────────────────────────────────────────

def compute_payment_entropy(payment_days: list) -> float:
    """
    F9: Payment Regularity Entropy
    High entropy = chaotic payments = stressed customer
    Low entropy  = predictable payments = disciplined customer

    payment_days: list of day-of-month when bills were paid
    Example: [2, 3, 1, 2] → low entropy (predictable)
             [2, 15, 28, 7] → high entropy (chaotic)
    """
    if not payment_days or len(payment_days) < 2:
        return 0.3  # Default moderate entropy

    payment_days = np.array(payment_days, dtype=float)

    # Bin payments into weekly buckets (weeks 1-4 of month)
    bins = np.zeros(4)
    for day in payment_days:
        week = min(int((day - 1) / 7), 3)
        bins[week] += 1

    # Normalize to probabilities
    total = bins.sum()
    if total == 0:
        return 0.3

    probs = bins / total
    probs = probs[probs > 0]  # Remove zeros for log calculation

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))

    # Normalize: max entropy for 4 bins = log2(4) = 2.0
    normalized_entropy = entropy / 2.0

    return float(np.clip(normalized_entropy, 0.0, 1.0))


# ─────────────────────────────────────────
# MAIN FEATURE BUILDER — 10 FEATURES
# ─────────────────────────────────────────

def build_feature_vector(raw: dict) -> dict:
    """
    Builds the exact 10-feature vector per PS specification.

    Input:  raw customer data dict
    Output: F1-F10 feature dict
    """

    income = max(raw["income"], 1)

    # ──────────────────────────────────────
    # F1: Debt Burden Ratio
    # Formula: EMI ÷ Monthly Income
    # ──────────────────────────────────────
    emi = raw["emi"]
    F1_debt_burden = np.clip(emi / income, 0.01, 0.95)

    # ──────────────────────────────────────
    # F2: Savings Drawdown Rate
    # Formula: (Balance_30d_ago - Balance_now) ÷ Balance_30d_ago
    # ──────────────────────────────────────
    balance_now = max(raw.get("balance_now", income), 1)
    balance_30d = max(raw.get("balance_30d_ago", income), 1)
    F2_savings_drawdown = np.clip(
        max(0, balance_30d - balance_now) / balance_30d,
        0.0, 1.0
    )

    # ──────────────────────────────────────
    # F3: Lending App Activity Score
    # Formula: α × Num_Apps + β × (Total_Borrowed / Income)
    # α = 0.3, β = 0.7 (per PS specification)
    # ──────────────────────────────────────
    num_apps      = raw.get("num_lending_apps", 0)
    total_borrowed = raw.get("total_app_amount", 0)
    alpha_f3, beta_f3 = 0.3, 0.7
    raw_f3 = alpha_f3 * (num_apps / 5.0) + beta_f3 * (total_borrowed / income)
    F3_lending_app_score = np.clip(raw_f3, 0.0, 1.0)

    # ──────────────────────────────────────
    # F4: Discretionary Spending Decline
    # Formula: (Prev_spend - Current_spend) ÷ Prev_spend
    # ──────────────────────────────────────
    usual_spend = max(raw.get("usual_discretionary_spend", 1), 1)
    this_spend  = raw.get("this_month_discretionary_spend", usual_spend)
    F4_spending_decline = np.clip(
        max(0, usual_spend - this_spend) / usual_spend,
        0.0, 1.0
    )

    # ──────────────────────────────────────
    # F5: Salary Credit Delay
    # Formula: Days delayed ÷ 30 (normalized)
    # ──────────────────────────────────────
    expected_day = raw.get("expected_salary_day", 1)
    actual_day   = raw.get("actual_salary_day", 1)
    delay_days   = max(0, actual_day - expected_day)
    F5_salary_delay = np.clip(delay_days / 30.0, 0.0, 1.0)

    # ──────────────────────────────────────
    # F6: Auto-Debit Failure Rate
    # Formula: Failed ÷ Total attempts
    # ──────────────────────────────────────
    failed_debits = raw.get("failed_debits", 0)
    total_debits  = max(raw.get("total_debits", 1), 1)
    F6_autodebit_fail = np.clip(failed_debits / total_debits, 0.0, 1.0)

    # ──────────────────────────────────────
    # F7: Cash Hoarding Behavior
    # Formula: ATM_this_month ÷ ATM_avg_6months
    # Ratio > 1 = hoarding, normalized to 0-1
    # ──────────────────────────────────────
    avg_atm  = max(raw.get("usual_atm_withdrawal", 1), 1)
    this_atm = raw.get("this_month_atm_withdrawal", avg_atm)
    raw_f7   = this_atm / avg_atm  # e.g. 2.67 = 267% of normal
    F7_cash_hoarding = np.clip(
        (raw_f7 - 1.0) / 2.0,   # Normalize: 1x=0, 3x=1
        0.0, 1.0
    )

    # ──────────────────────────────────────
    # F8: Stress Velocity (ODE)
    # dΩ/dt = α·D(t) + β·L(t) - γ·R(t)
    # ──────────────────────────────────────
    F8_stress_velocity = compute_stress_velocity(
        debt_burden    = float(F1_debt_burden),
        income_volatility = np.clip(
            raw.get("income_std", income * 0.05) / income,
            0.005, 0.80
        ),
        lending_app_score = float(F3_lending_app_score),
        savings_drawdown  = float(F2_savings_drawdown),
        salary_delay      = float(F5_salary_delay),
        autodebit_fail    = float(F6_autodebit_fail),
        cash_hoarding     = float(F7_cash_hoarding),
    )

    # ──────────────────────────────────────
    # F9: Payment Regularity Entropy
    # Shannon entropy of payment day distribution
    # ──────────────────────────────────────
    payment_days = raw.get("payment_days", [])

    # If no payment days provided, estimate from available data
    if not payment_days:
        base = actual_day
        # Simulate pattern based on delay behavior
        if delay_days > 5:
            # Chaotic pattern
            payment_days = [base, base + 8, base + 15, base + 3]
        else:
            # Regular pattern
            payment_days = [base, base + 1, base, base + 1]

    F9_payment_entropy = compute_payment_entropy(payment_days)

    # ──────────────────────────────────────
    # F10: Cohort Peer Stress
    # Default rate of similar customers in last 30 days
    # ──────────────────────────────────────
    F10_cohort_stress = np.clip(
        raw.get("cohort_default_rate", 0.05),
        0.0, 0.50
    )

    return {
        "F1_debt_burden":       round(float(F1_debt_burden), 4),
        "F2_savings_drawdown":  round(float(F2_savings_drawdown), 4),
        "F3_lending_app_score": round(float(F3_lending_app_score), 4),
        "F4_spending_decline":  round(float(F4_spending_decline), 4),
        "F5_salary_delay":      round(float(F5_salary_delay), 4),
        "F6_autodebit_fail":    round(float(F6_autodebit_fail), 4),
        "F7_cash_hoarding":     round(float(F7_cash_hoarding), 4),
        "F8_stress_velocity":   round(float(F8_stress_velocity), 5),
        "F9_payment_entropy":   round(float(F9_payment_entropy), 4),
        "F10_cohort_stress":    round(float(F10_cohort_stress), 4),
    }