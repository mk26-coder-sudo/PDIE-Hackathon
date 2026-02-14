import numpy as np
from scipy.integrate import odeint


# ─────────────────────────────────────────
# ODE DEFINITIONS
# ─────────────────────────────────────────

def stress_ode(S, t, debt_pressure, load_markers, recovery):
    """
    Feature 8: Stress Velocity
    dS/dt = alpha*debt_pressure + beta*load_markers - gamma*recovery
    """
    alpha, beta, gamma = 0.4, 0.35, 0.25
    dSdt = alpha * debt_pressure + beta * load_markers - gamma * recovery
    return dSdt


def foresight_ode(P, t, proactive_signals, stress_erosors):
    """
    Feature 9: Foresight Momentum
    dP/dt = delta*proactive_signals - epsilon*stress_erosors
    """
    delta, epsilon = 0.45, 0.55
    dPdt = delta * proactive_signals - epsilon * stress_erosors
    return dPdt


# ─────────────────────────────────────────
# ODE SOLVER
# ─────────────────────────────────────────

def compute_ode_features(
    debt_burden,
    income_volatility,
    lending_app_score,
    savings_depletion,
    salary_delay,
    autodebit_fail,
    spending_cuts
):
    """
    Solves ODEs over 12 months and returns
    F8 (stress_velocity) and F9 (foresight_momentum)
    """
    t = np.linspace(0, 12, 120)  # 12 months, 120 points

    # --- Stress ODE inputs ---
    debt_pressure = debt_burden * 0.6 + income_volatility * 0.4
    load_markers  = spending_cuts * 0.5 + salary_delay * 0.5
    recovery      = max(0.01, 1.0 - savings_depletion)

    S0 = debt_burden * 0.5  # initial stress
    S_trajectory = odeint(
        stress_ode, S0, t,
        args=(debt_pressure, load_markers, recovery)
    )
    S_arr = S_trajectory.flatten()

    # Velocity = slope of last 3 months
    stress_velocity = float(np.polyfit(t[-30:], S_arr[-30:], 1)[0])
    stress_velocity = np.clip(stress_velocity, 0.0, 0.12)

    # --- Foresight ODE inputs ---
    proactive_signals = max(0.01, 1.0 - autodebit_fail)
    stress_erosors    = lending_app_score * 0.6 + spending_cuts * 0.4

    P0 = 1.0 - debt_burden  # initial planning ability
    P_trajectory = odeint(
        foresight_ode, P0, t,
        args=(proactive_signals, stress_erosors)
    )
    P_arr = P_trajectory.flatten()

    foresight_momentum = float(np.polyfit(t[-30:], P_arr[-30:], 1)[0])
    foresight_momentum = np.clip(abs(foresight_momentum), 0.001, 0.08)

    return stress_velocity, foresight_momentum


# ─────────────────────────────────────────
# MAIN FEATURE BUILDER
# ─────────────────────────────────────────

def build_feature_vector(raw: dict) -> dict:
    """
    Input:  raw customer data dict
    Output: 10-feature vector dict + metadata
    """

    # ── Group 1: Basic Financial (F1-F4) ──
    income       = max(raw["income"], 1)
    emi          = raw["emi"]
    F1_debt_burden       = np.clip(emi / income, 0.01, 0.95)
    F2_income_volatility = np.clip(raw["income_std"] / income, 0.005, 0.80)

    num_apps     = raw.get("num_lending_apps", 0)
    total_app_amt= raw.get("total_app_amount", 0)
    F3_lending_app_score = np.clip(
        (num_apps / 5.0) * 0.6 + (total_app_amt / income) * 0.4,
        0.0, 1.0
    )

    balance_now  = raw.get("balance_now", income)
    balance_30d  = raw.get("balance_30d_ago", income)
    daily_burn   = max(0, (balance_30d - balance_now) / 30)
    daily_income = income / 30
    F4_savings_depletion = np.clip(daily_burn / max(daily_income, 1), 0.0, 1.0)

    # ── Group 2: Behavioral (F5-F7) ──
    expected_salary_day = raw.get("expected_salary_day", 5)
    actual_salary_day   = raw.get("actual_salary_day", 5)
    F5_salary_delay = np.clip(
        abs(actual_salary_day - expected_salary_day) / 30.0,
        0.0, 0.30
    )

    failed_debits = raw.get("failed_debits", 0)
    total_debits  = max(raw.get("total_debits", 1), 1)
    F6_autodebit_fail = np.clip(failed_debits / total_debits, 0.0, 1.0)

    usual_spend = max(raw.get("usual_discretionary_spend", 1), 1)
    this_spend  = raw.get("this_month_discretionary_spend", usual_spend)
    F7_spending_cuts = np.clip(
        max(0, usual_spend - this_spend) / usual_spend,
        0.0, 1.0
    )

    # ── Group 3: ODE Features (F8-F9) ──
    F8_stress_velocity, F9_foresight_momentum = compute_ode_features(
        F1_debt_burden, F2_income_volatility,
        F3_lending_app_score, F4_savings_depletion,
        F5_salary_delay, F6_autodebit_fail, F7_spending_cuts
    )

    # ── F10: Neighborhood Risk ──
    F10_neighborhood_risk = np.clip(
        raw.get("neighborhood_default_rate", 0.05),
        0.0, 0.50
    )

    return {
        "F1_debt_burden":        round(float(F1_debt_burden), 4),
        "F2_income_volatility":  round(float(F2_income_volatility), 4),
        "F3_lending_app_score":  round(float(F3_lending_app_score), 4),
        "F4_savings_depletion":  round(float(F4_savings_depletion), 4),
        "F5_salary_delay":       round(float(F5_salary_delay), 4),
        "F6_autodebit_fail":     round(float(F6_autodebit_fail), 4),
        "F7_spending_cuts":      round(float(F7_spending_cuts), 4),
        "F8_stress_velocity":    round(float(F8_stress_velocity), 5),
        "F9_foresight_momentum": round(float(F9_foresight_momentum), 5),
        "F10_neighborhood_risk": round(float(F10_neighborhood_risk), 4),
    }