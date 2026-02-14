import numpy as np
import joblib
import os

# ─────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────

TRACK1_FEATURES = [
    "F1_debt_burden",
    "F2_income_volatility",
    "F3_lending_app_score",
    "F4_savings_depletion",
    "F5_salary_delay",
    "F6_autodebit_fail",
    "F7_spending_cuts",
]

TRACK2_FEATURES = TRACK1_FEATURES + [
    "F8_stress_velocity",
    "F9_foresight_momentum",
    "F10_neighborhood_risk",
]

# ─────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

track1_model = joblib.load(os.path.join(BASE_DIR, "xgboost_track1.pkl"))
track2_model = joblib.load(os.path.join(BASE_DIR, "lightgbm_track2.pkl"))

print("[INFO] Models loaded successfully ✓")

# ─────────────────────────────────────────
# SHAP DRIVER DETECTION
# ─────────────────────────────────────────

def get_primary_driver(features: dict) -> str:
    """Find biggest risk contributor using SHAP-style weights"""
    weights = {
        "F1_debt_burden":        0.25,
        "F2_income_volatility":  0.10,
        "F3_lending_app_score":  0.20,
        "F4_savings_depletion":  0.15,
        "F5_salary_delay":       0.08,
        "F6_autodebit_fail":     0.12,
        "F7_spending_cuts":      0.10,
        "F8_stress_velocity":    2.50,  # ODE scaled up
        "F9_foresight_momentum": 0.50,
        "F10_neighborhood_risk": 0.30,
    }
    contributions = {k: weights[k] * features[k] for k in weights}
    return max(contributions, key=contributions.get)


# ─────────────────────────────────────────
# INTERVENTION LOGIC
# ─────────────────────────────────────────

def get_intervention(primary_driver: str, pd_pct: float, features: dict) -> dict:
    """Rule-based intervention — no AI hallucination risk"""

    if pd_pct <= 10:
        return {"recommended": False, "type": "none", "message": "No intervention needed."}

    interventions = {
        "F3_lending_app_score": {
            "type": "debt_consolidation",
            "urgency": "HIGH" if pd_pct > 20 else "MEDIUM",
            "message": (
                "You appear to be managing multiple loan apps. "
                "We can consolidate these into one lower EMI loan. "
                "Speak to your relationship manager today."
            ),
        },
        "F5_salary_delay": {
            "type": "emi_postponement",
            "urgency": "MEDIUM",
            "message": (
                "We noticed your salary credit is delayed this month. "
                "We can shift your EMI due date by 7 days at no extra cost."
            ),
        },
        "F4_savings_depletion": {
            "type": "tenure_extension",
            "urgency": "MEDIUM",
            "message": (
                "Your savings balance has been declining. "
                "We can extend your loan tenure to reduce your monthly burden."
            ),
        },
        "F8_stress_velocity": {
            "type": "financial_counseling",
            "urgency": "HIGH",
            "message": (
                "Our system has detected rapidly increasing financial stress. "
                "A free financial counseling session is available for you."
            ),
        },
        "F6_autodebit_fail": {
            "type": "balance_alert",
            "urgency": "MEDIUM",
            "message": (
                "Your auto-debit payments have been failing due to low balance. "
                "Set up a low-balance alert to avoid missed payments."
            ),
        },
    }

    default_intervention = {
        "type": "general_support",
        "urgency": "LOW",
        "message": "Please contact your relationship manager for support.",
    }

    intervention = interventions.get(primary_driver, default_intervention)
    return {"recommended": True, **intervention}


# ─────────────────────────────────────────
# BAND & LOAN TERMS
# ─────────────────────────────────────────

def get_loan_terms(pd_pct: float, requested_amount: float) -> dict:
    if pd_pct <= 10:
        return {
            "band": "BAND_1",
            "decision": "APPROVED",
            "approved_amount": requested_amount,
            "approval_pct": 100,
            "interest_rate": 12.0,
            "processing_fee": 0,
        }
    elif pd_pct <= 20:
        return {
            "band": "BAND_2",
            "decision": "APPROVED_WITH_CONDITIONS",
            "approved_amount": round(requested_amount * 0.70, 2),
            "approval_pct": 70,
            "interest_rate": 15.0,
            "processing_fee": 500,
        }
    else:
        return {
            "band": "BAND_3",
            "decision": "REJECTED",
            "approved_amount": 0,
            "approval_pct": 0,
            "interest_rate": 0,
            "processing_fee": 0,
        }


# ─────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────

def predict(features: dict, requested_amount: float = 300000) -> dict:
    """
    Full dual-track prediction pipeline.
    Input:  features dict (10 features)
    Output: complete risk assessment + decision + intervention
    """

    # ── Track 1: XGBoost (all customers) ──
    x1 = np.array([[features[f] for f in TRACK1_FEATURES]])
    risk_fast = float(track1_model.predict_proba(x1)[0][1])

    # ── Gate: Is Track 2 needed? ──
    risk_fast_pct = risk_fast * 100
    track2_activated = 15 <= risk_fast_pct <= 25

    if track2_activated:
        # ── Track 2: LightGBM (ambiguous band only) ──
        x2 = np.array([[features[f] for f in TRACK2_FEATURES]])
        risk_deep = float(track2_model.predict_proba(x2)[0][1])

        # ── Fusion: 70/30 weighted average ──
        final_pd = 0.70 * risk_fast + 0.30 * risk_deep
    else:
        risk_deep = None
        final_pd = risk_fast

    # Calibration nudge
    calibrated_pd = float(np.clip(0.05 + 0.90 * final_pd, 0.01, 0.99))
    pd_pct = round(calibrated_pd * 100, 1)

    # ── Decision ──
    loan_terms = get_loan_terms(pd_pct, requested_amount)

    # ── SHAP-style primary driver ──
    primary_driver = get_primary_driver(features)

    # ── Intervention ──
    intervention = get_intervention(primary_driver, pd_pct, features)

    return {
        "pd_pct": pd_pct,
        "risk_fast_pct": round(risk_fast_pct, 1),
        "risk_deep_pct": round(risk_deep * 100, 1) if risk_deep else None,
        "track2_activated": track2_activated,
        "primary_driver": primary_driver,
        "loan_terms": loan_terms,
        "intervention": intervention,
    }