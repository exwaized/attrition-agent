# ============================================================
# feature_engineering.py — Feature Engineering Pipeline
# ============================================================
# PURPOSE: Transforms raw employee records into ML-ready features
# FLOW: reads employees.csv → engineers 18+ features → saves features.csv
# CONNECTED TO: generate.py (input) → train.py (output)
# ============================================================

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# --- Step 1: Load config and raw data ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["data"]["output_path"])
print(f"Loaded {len(df)} employee records")

# --- Step 2: Recency Features ---
# How recently did the employee do something meaningful
# Higher values = longer since last action = disengagement signal
df["recency_promotion"]       = df["months_since_promotion"] * 30
df["recency_hike"]            = df["months_since_hike"] * 30
df["recency_manager_change"]  = df["months_since_manager_changed"] * 30

# --- Step 3: Trend Features ---
# Direction of change matters more than absolute value
# Negative slope = declining activity = strongest attrition precursor
df["trend_login"]       = df["login_freq_slope_30d"]
df["trend_performance"] = df["performance_rating_delta"]
df["trend_training"]    = df["training_completions_90d"].apply(
    lambda x: -1 if x == 0 else 1
)

# --- Step 4: Volatility Features ---
# Erratic behaviour is itself a signal — instability precedes departure
# Normalise by tenure — new joiners naturally take less leave
df["volatility_leave"]       = df["leave_days_30d"] / (df["tenure_months"] + 1)
df["volatility_performance"] = df["performance_rating_delta"].abs()

# --- Step 5: Org Context Features ---
# What's happening around the employee affects their decision to stay
# Peer attrition creates contagion — seeing others leave normalises leaving
df["org_peer_attrition"]   = df["peer_attrition_count_90d"]
df["org_manager_attrition"]= df["manager_attrition_rate_6m"]
df["org_team_shrink"]      = (df["team_size_change_pct"] < -0.10).astype(int)

# --- Step 6: Compensation Features ---
# Pay dissatisfaction is the most actionable and measurable driver
# Compa-ratio < 0.85 = underpaid 15%+ vs market = active flight risk
df["comp_compa_ratio"]    = df["compa_ratio"]
df["comp_hike_vs_median"] = df["hike_vs_band_median"]
df["comp_underpaid_flag"] = (df["compa_ratio"] < 0.85).astype(int)
df["comp_months_no_hike"] = df["months_since_hike"]

# --- Step 7: Career Progression Features ---
# Stagnation is a slow but reliable attrition driver
df["career_tenure_at_band"]      = df["months_since_promotion"]
df["career_band_normalized"]     = df["band"] / 7
df["career_promotion_velocity"]  = df["tenure_months"] / (df["band"] - 2)

# --- Step 8: Interaction Features ---
# Combinations more predictive than individual features alone
# These are the features that surprise interviewers

# Strongest predictor: high performer + no promotion > 18 months
df["interact_high_performer_no_promo"] = (
    (df["perf_rating_current"] >= 4.0) &
    (df["months_since_promotion"] > 18)
).astype(int)

# Double signal: underpaid AND activity declining simultaneously
df["interact_underpaid_declining"] = (
    (df["compa_ratio"] < 0.85) &
    (df["login_freq_slope_30d"] < 0)
).astype(int)

# Instability + contagion: new manager AND peer left recently
df["interact_new_mgr_peer_left"] = (
    (df["months_since_manager_changed"] < 3) &
    (df["peer_attrition_count_90d"] > 1)
).astype(int)

# Post-appraisal dissatisfaction window — highest voluntary exit risk
df["interact_post_appraisal_dissatisfied"] = (
    (df["hike_vs_band_median"] < -1.5) &
    (df["months_since_hike"] < 3)
).astype(int)

# --- Step 9: Encode categoricals ---
# ML models need numeric input — get_dummies avoids ordinal assumption
df["gender_encoded"] = (df["gender"] == "M").astype(int)
dept_dummies   = pd.get_dummies(df["department"], prefix="dept")
circle_dummies = pd.get_dummies(df["circle"], prefix="circle")
df = pd.concat([df, dept_dummies, circle_dummies], axis=1)

# --- Step 10: Define final feature set ---
# Explicit list — no implicit column selection
FEATURE_COLS = [
    "recency_promotion", "recency_hike", "recency_manager_change",
    "trend_login", "trend_performance", "trend_training",
    "volatility_leave", "volatility_performance",
    "org_peer_attrition", "org_manager_attrition", "org_team_shrink",
    "comp_compa_ratio", "comp_hike_vs_median", "comp_underpaid_flag", "comp_months_no_hike",
    "career_tenure_at_band", "career_band_normalized", "career_promotion_velocity",
    "interact_high_performer_no_promo", "interact_underpaid_declining",
    "interact_new_mgr_peer_left", "interact_post_appraisal_dissatisfied",
    "band", "tenure_months", "perf_rating_current", "login_freq_30d",
    "training_completions_90d", "leave_days_30d", "gender_encoded",
] + list(dept_dummies.columns) + list(circle_dummies.columns)

# --- Step 11: Save features + labels ---
feature_df = df[["employee_id"] + FEATURE_COLS + ["attrition_flag", "attrition_prob", "risk_profile"]]
output_path = Path("data/synthetic/features.csv")
feature_df.to_csv(output_path, index=False)

feature_cols_path = Path("data/synthetic/feature_cols.txt")
feature_cols_path.write_text("\n".join(FEATURE_COLS))

# --- Step 12: Sanity checks ---
print(f"\n✅ Features engineered: {len(FEATURE_COLS)} features")
print(f"Dataset shape: {feature_df.shape}")
print(f"Attrition rate: {feature_df['attrition_flag'].mean():.1%}")
print(f"Null check: {feature_df.isnull().sum().sum()} nulls")
print(f"\nInteraction feature counts:")
print(f"  high_performer_no_promo:     {feature_df['interact_high_performer_no_promo'].sum()}")
print(f"  underpaid_declining:         {feature_df['interact_underpaid_declining'].sum()}")
print(f"  new_mgr_peer_left:           {feature_df['interact_new_mgr_peer_left'].sum()}")
print(f"  post_appraisal_dissatisfied: {feature_df['interact_post_appraisal_dissatisfied'].sum()}")
print(f"Saved to: {output_path}")
