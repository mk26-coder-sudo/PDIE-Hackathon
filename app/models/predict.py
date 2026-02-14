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
# RISK BAND CLASSIFICATION
# ─────────────────────────────────────────

def get_risk_band(pd_pct: float) -> dict:
    """Classify customer into risk bands for monitoring priority"""
    if pd_pct <= 10:
        return {
            "band": "LOW_RISK",
            "color": "green",
            "monitoring_frequency": "monthly",
            "priority": "LOW"
        }
    elif pd_pct <= 25:
        return {
            "band": "MEDIUM_RISK",
            "color": "yellow",
            "monitoring_frequency": "weekly",
            "priority": "MEDIUM"
        }
    elif pd_pct <= 50:
        return {
            "band": "HIGH_RISK",
            "color": "orange",
            "monitoring_frequency": "daily",
            "priority": "HIGH"
        }
    else:
        return {
            "band": "CRITICAL_RISK",
            "color": "red",
            "monitoring_frequency": "immediate",
            "priority": "CRITICAL"
        }


# ─────────────────────────────────────────
# INTERVENTION LOGIC
# ─────────────────────────────────────────

def get_intervention(primary_driver: str, pd_pct: float, features: dict) -> dict:
    """Determine intervention strategy based on risk drivers"""

    # No intervention needed for very low risk
    if pd_pct <= 10:
        return {
            "required": False,
            "urgency": "NONE",
            "type": "monitoring_only",
            "message": "Customer is in good financial health. Continue routine monitoring.",
            "actions": []
        }

    # Map drivers to specific interventions
    interventions = {
        "F3_lending_app_score": {
            "type": "debt_consolidation",
            "urgency": "HIGH" if pd_pct > 30 else "MEDIUM",
            "message": (
                "Customer is managing multiple loan apps totaling ₹{:.0f}. "
                "Recommend debt consolidation into single lower EMI loan."
            ).format(features.get("F3_lending_app_score", 0) * 30000),
            "actions": [
                "Schedule call with relationship manager within 48 hours",
                "Offer debt consolidation product",
                "Calculate consolidated EMI and savings"
            ]
        },
        "F5_salary_delay": {
            "type": "emi_restructuring",
            "urgency": "MEDIUM",
            "message": (
                "Salary credit delayed by {} days. "
                "Offer EMI date postponement or payment holiday."
            ).format(int(features.get("F5_salary_delay", 0) * 30)),
            "actions": [
                "Verify salary credit date with customer",
                "Offer 7-day EMI postponement",
                "Set up salary day alerts"
            ]
        },
        "F4_savings_depletion": {
            "type": "tenure_extension",
            "urgency": "MEDIUM" if pd_pct < 30 else "HIGH",
            "message": (
                "Savings depleted by {:.0f}% in last 30 days. "
                "Recommend loan tenure extension to reduce monthly burden."
            ).format(features.get("F4_savings_depletion", 0) * 100),
            "actions": [
                "Offer tenure extension (reduce EMI by 20-30%)",
                "Provide financial planning guidance",
                "Set up savings threshold alerts"
            ]
        },
        "F8_stress_velocity": {
            "type": "financial_counseling",
            "urgency": "HIGH" if pd_pct > 30 else "MEDIUM",
            "message": (
                "Financial stress is rapidly increasing (velocity: {:.3f}). "
                "Immediate counseling recommended to prevent default."
            ).format(features.get("F8_stress_velocity", 0)),
            "actions": [
                "Schedule urgent financial counseling session",
                "Review all loan obligations",
                "Create debt management plan",
                "Consider emergency restructuring options"
            ]
        },
        "F6_autodebit_fail": {
            "type": "payment_assistance",
            "urgency": "MEDIUM",
            "message": (
                "Auto-debit failure rate: {:.0f}%. "
                "Set up balance alerts and payment reminders."
            ).format(features.get("F6_autodebit_fail", 0) * 100),
            "actions": [
                "Enable low-balance SMS alerts",
                "Offer payment date flexibility",
                "Set up pre-debit reminders (3 days before)"
            ]
        },
        "F1_debt_burden": {
            "type": "debt_restructuring",
            "urgency": "HIGH",
            "message": (
                "Debt burden at {:.0f}% of income (healthy limit: 40%). "
                "Restructuring required to prevent default."
            ).format(features.get("F1_debt_burden", 0) * 100),
            "actions": [
                "Immediate review of all debt obligations",
                "Restructure high-interest loans first",
                "Consider top-up loan for consolidation"
            ]
        },
        "F7_spending_cuts": {
            "type": "budget_counseling",
            "urgency": "MEDIUM",
            "message": (
                "Discretionary spending cut by {:.0f}%. "
                "Customer may be facing cash flow issues."
            ).format(features.get("F7_spending_cuts", 0) * 100),
            "actions": [
                "Offer financial planning assistance",
                "Review income sources",
                "Discuss emergency fund creation"
            ]
        }
    }

    # Default intervention for unlisted drivers
    default_intervention = {
        "type": "relationship_manager_review",
        "urgency": "HIGH" if pd_pct > 30 else "MEDIUM",
        "message": (
            "Customer showing elevated default risk ({:.1f}%). "
            "Relationship manager review recommended."
        ).format(pd_pct),
        "actions": [
            "Schedule call with relationship manager",
            "Review account activity in detail",
            "Assess intervention options"
        ]
    }

    intervention = interventions.get(primary_driver, default_intervention)
    
    return {
        "required": True,
        **intervention
    }


# ─────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────

def predict(features: dict, requested_amount: float = 0) -> dict:
    """
    Pre-Delinquency Intervention Engine - Main Prediction Pipeline
    
    Purpose: Monitor existing customers and trigger early interventions
    NOT for new loan approvals - for existing loan portfolio monitoring
    
    Input:  features dict (10 features)
    Output: risk assessment + intervention recommendations
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

    # Calibration
    calibrated_pd = float(np.clip(0.05 + 0.90 * final_pd, 0.01, 0.99))
    pd_pct = round(calibrated_pd * 100, 1)

    # ── Risk Classification ──
    risk_band = get_risk_band(pd_pct)

    # ── SHAP-style primary driver ──
    primary_driver = get_primary_driver(features)

    # ── Intervention Strategy ──
    intervention = get_intervention(primary_driver, pd_pct, features)

    # ── Trend Analysis (comparing stress velocity) ──
    stress_velocity = features.get("F8_stress_velocity", 0)
    trend = "RAPIDLY_INCREASING" if stress_velocity > 0.08 else \
            "INCREASING" if stress_velocity > 0.04 else \
            "STABLE" if stress_velocity > -0.02 else \
            "IMPROVING"

    return {
        "pd_pct": pd_pct,
        "risk_fast_pct": round(risk_fast_pct, 1),
        "risk_deep_pct": round(risk_deep * 100, 1) if risk_deep else None,
        "track2_activated": track2_activated,
        "primary_driver": primary_driver,
        "risk_classification": risk_band,
        "stress_trend": trend,
        "intervention": intervention,
        # Legacy support (can be removed later)
        "loan_terms": {
            "band": risk_band["band"],
            "decision": "INTERVENTION_" + intervention["urgency"],
            "approved_amount": 0,
            "approval_pct": 0,
            "interest_rate": 0,
            "processing_fee": 0
        }
    }