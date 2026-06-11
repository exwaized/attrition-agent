# ============================================================
# main.py — FastAPI Production Serving Layer
# ============================================================
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path("E:/attrition-agent/.env"), override=True)
import os
print(f"KEY LOADED: {os.environ.get('GROQ_API_KEY')[:20]}...")

import json
import sqlite3
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import yaml
import io

import sys
sys.path.append(".")
from agents.attrition_agent import run_agent, run_batch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(
    title="Jio Attrition Intelligence API",
    description="Early attrition detection and intervention recommendation system",
    version="1.0.0"
)

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
        except:
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
    total_ev              = urgent["ev"].sum()
    replacement_cost      = cfg["ev_framework"]["replacement_cost"]
    expected_replacement  = urgent["p_attrition"].sum()
    replacement_avoided   = expected_replacement * replacement_cost
    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_employees": len(scored_df),
            "critical_count":  len(critical),
            "high_count":      len(high),
            "medium_count":    len(medium),
            "low_count":       len(scored_df) - len(critical) - len(high) - len(medium)
        },
        "budget_summary": {
            "total_ev_urgent":          round(total_ev),
            "expected_replacements":    round(expected_replacement, 1),
            "replacement_cost_avoided": round(replacement_avoided),
            "net_roi":                  round(replacement_avoided / max(total_ev, 1), 2)
        },
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
    tier_map  = {
        "CRITICAL": ["CRITICAL"],
        "HIGH":     ["CRITICAL", "HIGH"],
        "ALL":      ["CRITICAL", "HIGH", "MEDIUM"]
    }
    tiers               = tier_map.get(correction_type.upper(), ["CRITICAL"])
    subset              = scored_df[scored_df["risk_tier"].isin(tiers)]
    intervention_cost   = cfg["ev_framework"]["intervention_cost"]
    replacement_cost    = cfg["ev_framework"]["replacement_cost"]
    total_cost          = len(subset) * intervention_cost
    expected_retentions = (subset["p_attrition"] * 0.75).sum()
    replacement_avoided = expected_retentions * replacement_cost
    net_roi             = replacement_avoided / max(total_cost, 1)
    return {
        "simulation_tier":           correction_type,
        "employees_affected":        len(subset),
        "total_intervention_cost":   round(total_cost),
        "expected_retentions":       round(expected_retentions, 1),
        "replacement_cost_avoided":  round(replacement_avoided),
        "net_roi":                   round(net_roi, 2),
        "payback_months":            round(total_cost / max(replacement_avoided / 12, 1), 1)
    }

@app.get("/audit/recent")
def recent_audit(limit: int = 20):
    db_path = cfg["paths"]["audit_db"]
    if not Path(db_path).exists():
        return {"message": "No audit log yet", "records": []}
    conn    = sqlite3.connect(db_path)
    cursor  = conn.execute(
        f"SELECT * FROM audit_log ORDER BY created_at DESC LIMIT {limit}"
    )
    columns = [desc[0] for desc in cursor.description]
    rows    = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return {"total_records": len(rows), "records": rows}