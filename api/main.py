# ============================================================
# main.py — FastAPI Production Serving Layer
# ============================================================
import io
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

# Was hardcoded to "E:/attrition-agent/.env" — only ever worked on one
# specific Windows machine with that exact drive letter, and silently
# does nothing everywhere else (a fresh clone, CI, a teammate's laptop).
# override=True without a hardcoded path lets python-dotenv discover
# .env relative to cwd, same as agents/attrition_agent.py does it.
load_dotenv(override=True)

sys.path.append(".")
from agents.attrition_agent import run_agent  # noqa: E402, I001 — needs sys.path.append + load_dotenv above

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(
    title="Jio Attrition Intelligence API",
    description="Early attrition detection and intervention recommendation system",
    version="1.0.0"
)


# ============================================================
# Pure, testable functions
# ============================================================

def resolve_tier_filter(correction_type: str) -> list:
    """
    Maps a budget-simulation scope name to the risk tiers it covers.
    Unknown values fall back to CRITICAL-only, the most conservative
    scope — better to under-simulate than silently scope to everyone
    on a typo'd query param.
    """
    tier_map = {
        "CRITICAL": ["CRITICAL"],
        "HIGH":     ["CRITICAL", "HIGH"],
        "ALL":      ["CRITICAL", "HIGH", "MEDIUM"]
    }
    return tier_map.get(correction_type.upper(), ["CRITICAL"])


def compute_budget_summary(urgent_df: pd.DataFrame, replacement_cost: float) -> dict:
    """
    EV/ROI math for the weekly digest — same EV framework as
    ev_scoring.py, applied to the subset of employees worth
    budgeting an intervention for. Pure given a dataframe with
    'ev' and 'p_attrition' columns plus the cost constant.
    """
    total_ev             = urgent_df["ev"].sum()
    expected_replacement = urgent_df["p_attrition"].sum()
    replacement_avoided  = expected_replacement * replacement_cost

    return {
        "total_ev_urgent":          round(total_ev),
        "expected_replacements":    round(expected_replacement, 1),
        "replacement_cost_avoided": round(replacement_avoided),
        "net_roi":                  round(replacement_avoided / max(total_ev, 1), 2)
    }


def simulate_budget(subset: pd.DataFrame, intervention_cost: float,
                     replacement_cost: float, correction_type: str) -> dict:
    """
    "What if we intervened on this tier?" simulation. Assumes a 75%
    retention success rate per intervention (subset["p_attrition"] * 0.75)
    — pure given the subset dataframe and cost constants, so this
    assumption is easy to spot and change later without touching the
    endpoint itself.
    """
    total_cost          = len(subset) * intervention_cost
    expected_retentions = (subset["p_attrition"] * 0.75).sum()
    replacement_avoided = expected_retentions * replacement_cost
    net_roi             = replacement_avoided / max(total_cost, 1)

    return {
        "simulation_tier":          correction_type,
        "employees_affected":       len(subset),
        "total_intervention_cost":  round(total_cost),
        "expected_retentions":      round(expected_retentions, 1),
        "replacement_cost_avoided": round(replacement_avoided),
        "net_roi":                  round(net_roi, 2),
        "payback_months":           round(total_cost / max(replacement_avoided / 12, 1), 1)
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health_check():
    checks = {}
    checks["champion_model"] = Path("models/champion_model.pkl").exists()
    checks["cox_model"]      = Path("models/cox_model.pkl").exists()
    checks["scored_data"]    = Path("data/synthetic/scored_employees.csv").exists()
    checks["chroma_db"]      = Path(cfg["paths"]["chroma_db"]).exists()
    all_ok = all(checks.values())
    return {
        "status":    "healthy" if all_ok else "degraded",
        "checks":    checks,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/explain/{employee_id}")
def explain_employee(employee_id: str):
    try:
        result = run_agent(employee_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        try:
            rec = json.loads(result["llm_recommendation"])
        except Exception:
            rec = {"narrative": result["llm_recommendation"]}
        return {
            "employee_id":            result["employee_id"],
            "risk_tier":              result["risk_tier"],
            "p_attrition":            round(result["p_attrition"], 4),
            "median_survival_months": round(result["median_survival_months"], 1),
            "ev":                     round(result["ev"], 0),
            "shap_drivers":           result["shap_drivers"],
            "recommendation":         rec,
            "routed_to":              result["routed_to"],
            "employee_profile": {
                "band":               result["employee_data"].get("band"),
                "department":         result["employee_data"].get("department"),
                "circle":             result["employee_data"].get("circle"),
                "tenure_months":      result["employee_data"].get("tenure_months"),
                "compa_ratio":        result["employee_data"].get("compa_ratio"),
                "perf_rating":        result["employee_data"].get("perf_rating_current"),
                "months_since_promo": result["employee_data"].get("months_since_promotion"),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/weekly")
def weekly_digest():
    scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
    critical  = scored_df[scored_df["risk_tier"] == "CRITICAL"]
    high      = scored_df[scored_df["risk_tier"] == "HIGH"]
    medium    = scored_df[scored_df["risk_tier"] == "MEDIUM"]
    urgent    = scored_df[scored_df["risk_tier"].isin(["CRITICAL", "HIGH"])]

    replacement_cost = cfg["ev_framework"]["replacement_cost"]
    budget_summary    = compute_budget_summary(urgent, replacement_cost)

    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_employees": len(scored_df),
            "critical_count":  len(critical),
            "high_count":      len(high),
            "medium_count":    len(medium),
            "low_count":       len(scored_df) - len(critical) - len(high) - len(medium)
        },
        "budget_summary": budget_summary,
        "top_critical": critical.head(10)[
            ["employee_id", "band", "department", "p_attrition", "ev", "median_survival_months"]
        ].to_dict(orient="records"),
        "top_high": high.head(10)[
            ["employee_id", "band", "department", "p_attrition", "ev", "median_survival_months"]
        ].to_dict(orient="records")
    }


@app.get("/report/department/{dept_name}")
def department_report(dept_name: str):
    scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
    dept_df   = scored_df[scored_df["department"].str.lower() == dept_name.lower()]
    if dept_df.empty:
        raise HTTPException(status_code=404, detail=f"Department '{dept_name}' not found")
    heatmap = dept_df.groupby("band").agg(
        avg_p_attrition=("p_attrition", "mean"),
        high_risk_count=("risk_tier", lambda x: (x.isin(["CRITICAL", "HIGH"])).sum()),
        total_employees=("employee_id", "count"),
        total_ev=("ev", "sum")
    ).reset_index().to_dict(orient="records")
    return {
        "department":        dept_name,
        "total_employees":   len(dept_df),
        "avg_p_attrition":   round(dept_df["p_attrition"].mean(), 3),
        "heatmap_by_band":   heatmap,
        "critical_employees": dept_df[dept_df["risk_tier"] == "CRITICAL"][
            ["employee_id", "band", "p_attrition", "ev"]
        ].to_dict(orient="records")
    }


@app.post("/score/batch")
async def batch_score(file: UploadFile = File(...)):
    contents  = await file.read()
    upload_df = pd.read_csv(io.BytesIO(contents))
    scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
    if "employee_id" not in upload_df.columns:
        raise HTTPException(status_code=400, detail="CSV must have 'employee_id' column")
    result    = upload_df.merge(
        scored_df[["employee_id", "p_attrition", "ev", "risk_tier", "median_survival_months"]],
        on="employee_id", how="left"
    )
    not_found = result["risk_tier"].isna().sum()
    return {
        "total_submitted": len(upload_df),
        "scored":          len(result) - not_found,
        "not_found":       int(not_found),
        "results":         result.fillna("NOT_FOUND").to_dict(orient="records")
    }


@app.get("/report/budget-sim")
def budget_simulation(correction_type: str = "CRITICAL"):
    scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
    tiers     = resolve_tier_filter(correction_type)
    subset    = scored_df[scored_df["risk_tier"].isin(tiers)]

    intervention_cost = cfg["ev_framework"]["intervention_cost"]
    replacement_cost  = cfg["ev_framework"]["replacement_cost"]

    return simulate_budget(subset, intervention_cost, replacement_cost, correction_type)


@app.get("/audit/recent")
def recent_audit(limit: int = 20):
    db_path = cfg["paths"]["audit_db"]
    if not Path(db_path).exists():
        return {"message": "No audit log yet", "records": []}
    conn    = sqlite3.connect(db_path)
    # Parameterized rather than f-string interpolated — FastAPI already
    # coerces `limit` to int before this runs (a non-numeric query param
    # gets a 422 automatically), so this wasn't exploitable in practice,
    # but parameterized queries are the right default regardless.
    cursor  = conn.execute(
        "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    columns = [desc[0] for desc in cursor.description]
    rows    = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return {"total_records": len(rows), "records": rows}