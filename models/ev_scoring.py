# ============================================================
# ev_scoring.py — Full Dataset EV Scoring + Risk Tiering
# ============================================================
# PURPOSE: Scores all 1800 employees with P(attrition) + EV
# FLOW: features.csv + champion_model → scored_employees.csv
# CONNECTED TO: train.py (input models) → agents/ (input data)
# ============================================================

import pandas as pd
import numpy as np
import pickle
import yaml
import json
from pathlib import Path

# --- Step 1: Load config, model, data ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

with open("models/champion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/cox_model.pkl", "rb") as f:
    cph = pickle.load(f)

with open("models/explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

with open("models/metadata.json") as f:
    meta = json.load(f)

feature_cols = Path("data/synthetic/feature_cols.txt").read_text().strip().split("\n")
df           = pd.read_csv("data/synthetic/features.csv")
raw_df       = pd.read_csv("data/synthetic/employees.csv")

print(f"Scoring {len(df)} employees...")
print(f"Champion model: {meta['champion']} | PR-AUC: {meta['pr_auc']}")

# --- Step 2: Score all employees ---
# predict_proba returns [P(stay), P(leave)]
# Column index 1 = P(leave) = attrition probability
X_all        = df[feature_cols]
probabilities = model.predict_proba(X_all)[:, 1]
df["p_attrition"] = probabilities

# --- Step 3: Cox PH survival prediction ---
# Median survival = predicted months before employee leaves
# Complements P(attrition) — tells HR HOW URGENTLY to act
cox_features = [
    "tenure_months", "comp_compa_ratio", "career_tenure_at_band",
    "interact_high_performer_no_promo", "interact_underpaid_declining",
    "org_peer_attrition", "org_manager_attrition",
    "trend_login", "trend_performance", "band"
]

cox_input = df[cox_features].copy()
cox_input.columns = cox_features

# Predict median survival time per employee
# Lower survival months = more urgent intervention needed
survival_predictions = cph.predict_median(cox_input)
df["median_survival_months"] = survival_predictions.values

# --- Step 4: SHAP top drivers per employee ---
# Top 3 SHAP features per employee = LLM prompt context in agent
# Extracting here so agent doesn't recompute for every employee
print("Computing SHAP values for all employees...")
shap_values = explainer.shap_values(X_all)

def get_top_shap_drivers(shap_row, feature_names, n=3):
    """
    Returns top N features by absolute SHAP value.
    These become the 'why' in LLM recommendation narrative.
    Positive = pushing toward leaving, negative = pushing toward staying.
    """
    pairs = list(zip(feature_names, shap_row))
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "shap": round(v, 4)} for f, v in sorted_pairs[:n]]

df["top_shap_drivers"] = [
    json.dumps(get_top_shap_drivers(shap_values[i], feature_cols))
    for i in range(len(df))
]

# --- Step 5: Expected Value calculation ---
# EV = P(attrition) x replacement_cost - intervention_cost
# Positive EV = worth intervening. Negative = don't spend budget.
# This converts ML probability into capital allocation decision
REPLACEMENT_COST  = cfg["ev_framework"]["replacement_cost"]
INTERVENTION_COST = cfg["ev_framework"]["intervention_cost"]
HIGH_THRESHOLD    = cfg["ev_framework"]["high_risk_threshold"]
MEDIUM_THRESHOLD  = cfg["ev_framework"]["medium_risk_threshold"]

df["ev"] = (df["p_attrition"] * REPLACEMENT_COST) - INTERVENTION_COST
df["ev"] = df["ev"].round(0)

# --- Step 6: Risk tiering ---
# Tier drives routing in LangGraph agent
# CRITICAL/HIGH → Slack alert, MEDIUM → digest, LOW → log only
def assign_risk_tier(row):
    p    = row["p_attrition"]
    ev   = row["ev"]
    surv = row["median_survival_months"]

    if p >= 0.70 and ev >= HIGH_THRESHOLD:
        return "CRITICAL"
    elif p >= 0.55 and ev >= HIGH_THRESHOLD * 0.5:
        return "HIGH"
    elif p >= 0.40 and ev >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_tier"] = df.apply(assign_risk_tier, axis=1)

# --- Step 7: Merge with raw employee metadata ---
# Agent needs circle, department, band etc for LLM context
scored = df[["employee_id", "p_attrition", "median_survival_months",
             "ev", "risk_tier", "top_shap_drivers"]].copy()

scored = scored.merge(
    raw_df[["employee_id", "band", "circle", "department", "gender",
            "tenure_months", "compa_ratio", "perf_rating_current",
            "months_since_promotion", "months_since_hike"]],
    on="employee_id", how="left"
)

# Sort by EV descending — highest priority first
scored = scored.sort_values("ev", ascending=False).reset_index(drop=True)

# --- Step 8: Save ---
output_path = Path("data/synthetic/scored_employees.csv")
scored.to_csv(output_path, index=False)

# --- Step 9: Summary stats ---
tier_counts = scored["risk_tier"].value_counts()
print(f"\n[OK] Scoring complete — {len(scored)} employees")
print(f"\nRisk tier distribution:")
for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    count = tier_counts.get(tier, 0)
    pct   = count / len(scored) * 100
    print(f"  {tier:<10}: {count:>4} employees ({pct:.1f}%)")

print(f"\nEV summary:")
print(f"  Total intervention budget needed:  Rs {scored[scored['risk_tier'].isin(['CRITICAL','HIGH'])]['ev'].sum():,.0f}")
print(f"  Avg P(attrition) CRITICAL:         {scored[scored['risk_tier']=='CRITICAL']['p_attrition'].mean():.3f}")
print(f"  Avg survival months CRITICAL:      {scored[scored['risk_tier']=='CRITICAL']['median_survival_months'].mean():.1f}")
print(f"\nTop 5 highest priority employees:")
print(scored[["employee_id","band","department","p_attrition","ev","risk_tier"]].head())
print(f"\nSaved to: {output_path}")
