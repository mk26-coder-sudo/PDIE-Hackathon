from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from pathlib import Path

from app.features import build_feature_vector
from app.models.predict import predict

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────

app = FastAPI(
    title="PDIE — Pre-Delinquency Intervention Engine",
    description="Real-time loan risk scoring with XGBoost + LightGBM + ODE features",
    version="2.5.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────

class CustomerRequest(BaseModel):
    customer_id: str
    customer_name: str

    # Core financials
    income: float
    emi: float
    income_std: Optional[float] = None

    # F2 — Savings
    balance_now: Optional[float] = None
    balance_30d_ago: Optional[float] = None

    # F3 — Lending apps
    num_lending_apps: Optional[int] = 0
    total_app_amount: Optional[float] = 0

    # F4 — Discretionary spending
    usual_discretionary_spend: Optional[float] = None
    this_month_discretionary_spend: Optional[float] = None

    # F5 — Salary timing
    expected_salary_day: Optional[int] = 1
    actual_salary_day: Optional[int] = 1

    # F6 — Auto-debit
    failed_debits: Optional[int] = 0
    total_debits: Optional[int] = 12

    # F7 — ATM / Cash hoarding
    usual_atm_withdrawal: Optional[float] = 5000
    this_month_atm_withdrawal: Optional[float] = 5000

    # F9 — Payment days (list of day-of-month)
    payment_days: Optional[list] = []

    # F10 — Cohort peer stress
    cohort_default_rate: Optional[float] = 0.05



# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "engine": "PDIE v2.5.1",
        "status": "running",
        "endpoints": {
            "dashboard": "/dashboard",
            "api_docs": "/docs",
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_risk(customer: CustomerRequest):
    try:
        # Step 1: Build 10-feature vector
        raw = customer.model_dump()
        features = build_feature_vector(raw)

        # Step 2: Run dual-track prediction
        result = predict(features, customer.requested_loan_amount)

        # Step 3: Return full response
        return {
            "customer_id": customer.customer_id,
            "customer_name": customer.customer_name,

            "risk_assessment": {
                "pd_pct": result["pd_pct"],
                "track1_xgboost_pct": result["risk_fast_pct"],
                "track2_lightgbm_pct": result["risk_deep_pct"],
                "track2_activated": result["track2_activated"],
                "primary_driver": result["primary_driver"],
            },

            "risk_classification": result["risk_classification"],
            "stress_trend": result["stress_trend"],
            "intervention": result["intervention"],

            "features": features,

            "compliance": {
                "model_version": "v2.5.1",
                "explanation_provided": True,
                "audit_trail": f"AUD_{hash(customer.customer_id) % 999999}",
            },
        }

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample-input")
def sample_input():
    """Returns a sample request body for testing"""
    return {
        "customer_id": "CUST_00001",
        "customer_name": "Rajesh Kumar",
        "income": 50000,
        "emi": 8000,
        "income_std": 1500,
        "requested_loan_amount": 300000,
        "balance_now": 28000,
        "balance_30d_ago": 45000,
        "expected_salary_day": 5,
        "actual_salary_day": 10,
        "failed_debits": 3,
        "total_debits": 10,
        "usual_discretionary_spend": 13000,
        "this_month_discretionary_spend": 7000,
        "num_lending_apps": 3,
        "total_app_amount": 12000,
        "neighborhood_default_rate": 0.09,
    }


# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)