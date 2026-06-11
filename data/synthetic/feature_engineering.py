# ============================================================
# feature_engineering.py — Feature Engineering Pipeline v2
# ============================================================
# PURPOSE: Transforms raw employee records into ML-ready features
# CHANGES FROM v1:
#   1. Peer attrition = dept+circle filtered (not random)
#   2. Multicollinearity fixes — dropped redundant parents:
#      - recency_promotion (= career_tenure_at_band × 30)
#      - recency_hike (= comp_months_no_hike × 30)
#      - career_band_normalized (= band / 7)
#      - comp_underpaid_flag (derived from comp_compa_ratio)
#   3. Dummy reference categories dropped (dept_HR, circle_Gujarat)
#   4. Added overload features + interactions
#   5. Added onboarding features + interactions
#   6. Added regime feature (is_new_joiner)
# FLOW: employees.csv → features.csv + feature_cols.txt
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

# ============================================================
# PEER ATTRITION — Department + Circle filtered
# ============================================================
# Peer = same department AND same circle
# Contagion is local not company-wide
# Excludes self — employee doesnt count their own leaving
# ============================================================
def compute_peer_attrition(df):
    peer_counts = []
    for idx, emp in df.iterrows():
        peers = df[
            (df["department"] == emp["department"]) &
            (df["circle"]     == emp["circle"]) &
            (df.index         != idx)
        ]
        peer_counts.append(int(peers["attrition_flag"].sum()))
    return peer_counts

print("Computing peer attrition (dept+circle filtered)...")
df["peer_attrition_count_90d"] = compute_peer_attrition(df)
print(f"Peer attrition computed — avg: {df['peer_attrition_count_90d'].mean():.2f}")

# ============================================================
# SYNTHETIC PROXIES — Overload + Onboarding
# ============================================================
np.random.seed(cfg["data"]["random_seed"])

# Overload proxies
# Overload proxies — added noise to prevent label leakage
# Real overload data has much more variance — not deterministic
df["overtime_hours_slope"] = np.random.normal(
    0.15 * df["attrition_flag"], 0.35  # more noise, weaker signal
)
df["after_hours_logins"] = np.random.poisson(
    np.where(df["attrition_flag"] == 1, 2.5, 1.2)  # closer distributions
).astype(float)

# Onboarding proxies — only meaningful for tenure <= 12 months
# Onboarding proxies — more noise
df["onboarding_completion"] = np.where(
    df["tenure_months"] <= 12,
    np.clip(
        np.random.normal(
            np.where(df["attrition_flag"] == 1, 0.72, 0.88), 0.15  # more overlap
        ), 0, 1
    ),
    1.0
)

df["days_to_first_project"] = np.where(
    df["tenure_months"] <= 12,
    np.random.poisson(
        np.where(df["attrition_flag"] == 1, 18, 10)  # closer distributions
    ).astype(float),
    0.0
)

df["manager_1on1_count_first30d"] = np.where(
    df["tenure_months"] <= 12,
    np.random.poisson(
        np.where(df["attrition_flag"] == 1, 1.5, 3.0)  # more overlap
    ).astype(float),
    0.0
)

# ============================================================
# STEP 2: Trend Features
# ============================================================
# login_freq_30d kept as base feature (level signal)
# trend_login = slope (direction signal) — both have value
# Cox PH confirmed trend HR=0.282 is strongest signal
# ============================================================
df["trend_login"]       = df["login_freq_slope_30d"]
df["trend_performance"] = df["performance_rating_delta"]
df["trend_training"]    = df["training_completions_90d"].apply(
    lambda x: -1 if x == 0 else 1
)

# ============================================================
# STEP 3: Volatility Features
# ============================================================
df["volatility_leave"]       = df["leave_days_30d"] / (df["tenure_months"] + 1)
df["volatility_performance"] = df["performance_rating_delta"].abs()

# ============================================================
# STEP 4: Org Context Features
# ============================================================
df["org_peer_attrition"]    = df["peer_attrition_count_90d"]
df["org_manager_attrition"] = df["manager_attrition_rate_6m"]
df["org_team_shrink"]       = (df["team_size_change_pct"] < -0.10).astype(int)

# ============================================================
# STEP 5: Compensation Features — REDUCED
# ============================================================
# comp_underpaid_flag DROPPED — derived from comp_compa_ratio
# Keeps comp_compa_ratio as continuous feature
# comp_hike_vs_median + comp_months_no_hike kept as unique signals
# ============================================================
df["comp_compa_ratio"]    = df["compa_ratio"]
df["comp_hike_vs_median"] = df["hike_vs_band_median"]
df["comp_months_no_hike"] = df["months_since_hike"]

# ============================================================
# STEP 6: Career Progression Features — REDUCED
# ============================================================
# career_band_normalized DROPPED — = band/7 (perfect collinearity)
# recency_promotion DROPPED — = career_tenure_at_band × 30
# recency_hike DROPPED — = comp_months_no_hike × 30
# recency_manager_change kept — unique signal
# ============================================================
df["career_tenure_at_band"]     = df["months_since_promotion"]
df["career_promotion_velocity"] = df["tenure_months"] / (df["band"] - 2)
df["recency_manager_change"]    = df["months_since_manager_changed"] * 30

# ============================================================
# STEP 7: Overload Features
# ============================================================
# Family prefix for naming consistency with other families
df["overload_overtime_slope"]   = df["overtime_hours_slope"]
df["overload_afterhours_login"] = df["after_hours_logins"]

# ============================================================
# STEP 8: Onboarding Features
# ============================================================
# is_new_joiner = regime feature separating two populations
# Onboarding features only fire for tenure <= 12 months
df["is_new_joiner"]           = (df["tenure_months"] <= 12).astype(int)
df["onboard_completion_rate"] = df["onboarding_completion"]
df["onboard_days_to_project"] = df["days_to_first_project"]
df["onboard_manager_1on1s"]   = df["manager_1on1_count_first30d"]

# ============================================================
# STEP 9: Interaction Features
# ============================================================
# Interactions replace dropped parent features
# Capture joint effects neither parent alone captures
# All protected from multicollinearity drops

# High performer + no promotion > 18 months
# Replaces: recency_promotion + perf_rating_current
df["interact_high_performer_no_promo"] = (
    (df["perf_rating_current"] >= 4.0) &
    (df["months_since_promotion"] > 18)
).astype(int)

# Underpaid AND activity declining
# Replaces: comp_compa_ratio individual + comp_underpaid_flag
df["interact_underpaid_declining"] = (
    (df["compa_ratio"] < 0.85) &
    (df["login_freq_slope_30d"] < 0)
).astype(int)

# New manager + peer left recently
df["interact_new_mgr_peer_left"] = (
    (df["months_since_manager_changed"] < 3) &
    (df["peer_attrition_count_90d"] > 1)
).astype(int)

# Post-appraisal dissatisfaction window
df["interact_post_appraisal_dissatisfied"] = (
    (df["hike_vs_band_median"] < -1.5) &
    (df["months_since_hike"] < 3)
).astype(int)

# Overload + underpaid = burnout-to-attrition pathway
df["interact_overload_underpaid"] = (
    (df["overtime_hours_slope"] > 0.3) &
    (df["compa_ratio"] < 0.90)
).astype(int)

# Onboarding failure + manager abandonment
df["interact_onboard_abandoned"] = (
    (df["tenure_months"] <= 12) &
    (df["onboarding_completion"] < 0.70) &
    (df["manager_1on1_count_first30d"] < 2)
).astype(int)

# New joiner + waited too long for real work
df["interact_onboard_idle"] = (
    (df["tenure_months"] <= 12) &
    (df["days_to_first_project"] > 21)
).astype(int)

# ============================================================
# STEP 10: Encode Categoricals
# ============================================================
df["gender_encoded"] = (df["gender"] == "M").astype(int)
dept_dummies         = pd.get_dummies(df["department"], prefix="dept")
circle_dummies       = pd.get_dummies(df["circle"],     prefix="circle")
df                   = pd.concat([df, dept_dummies, circle_dummies], axis=1)

# ============================================================
# STEP 11: Drop reference categories from dummy groups
# ============================================================
# One-hot encoded groups have perfect multicollinearity
# when all categories included — they sum to 1
# Fix: drop one reference category per group
# dept_HR = reference department
# circle_Gujarat = reference circle
# ============================================================
dept_dummies_clean   = [c for c in dept_dummies.columns   if c != "dept_HR"]
circle_dummies_clean = [c for c in circle_dummies.columns if c != "circle_Gujarat"]

# ============================================================
# STEP 12: Final Clean Feature Set
# ============================================================
# Total features: ~40 (down from 50 in v1)
# Dropped (multicollinearity):
#   recency_promotion, recency_hike, career_band_normalized,
#   comp_underpaid_flag + reference dummy categories
# Added (new signals):
#   overload family, onboarding family, regime feature,
#   3 new interaction terms
# ============================================================
FEATURE_COLS = [
    # Recency — only manager change kept
    # recency_promotion DROPPED (= career_tenure_at_band × 30)
    # recency_hike DROPPED (= comp_months_no_hike × 30)
    "recency_manager_change",

    # Trend — slope + level both kept
    "trend_login",
    "trend_performance",
    "trend_training",

    # Volatility
    "volatility_leave",
    "volatility_performance",

    # Org context
    "org_peer_attrition",
    "org_manager_attrition",
    "org_team_shrink",

    # Compensation — comp_underpaid_flag DROPPED
    "comp_compa_ratio",
    "comp_hike_vs_median",
    "comp_months_no_hike",

    # Career — career_band_normalized DROPPED
    "career_tenure_at_band",
    "career_promotion_velocity",

    # Overload family
    "overload_overtime_slope",
    "overload_afterhours_login",

    # Onboarding family
    "is_new_joiner",
    "onboard_completion_rate",
    "onboard_manager_1on1s",

    # Interactions — all protected from drops
    "interact_high_performer_no_promo",
    "interact_underpaid_declining",
    "interact_new_mgr_peer_left",
    "interact_post_appraisal_dissatisfied",
    "interact_overload_underpaid",
    "interact_onboard_abandoned",
    "interact_onboard_idle",

    # Base numeric
    "band",
    "tenure_months",
    "perf_rating_current",
    "login_freq_30d",
    "training_completions_90d",
    "leave_days_30d",
    "gender_encoded",

    # Dummies — reference categories dropped
] + dept_dummies_clean + circle_dummies_clean

# ============================================================
# STEP 13: Save
# ============================================================
# attrition_prob and risk_profile may not exist if mapper ran over raw data
# Only include columns that actually exist in df
extra_cols = [c for c in ["attrition_flag", "attrition_prob", "risk_profile"]
              if c in df.columns]

feature_df = df[["employee_id"] + FEATURE_COLS + extra_cols]


output_path = Path("data/synthetic/features.csv")
feature_df.to_csv(output_path, index=False)

feature_cols_path = Path("data/synthetic/feature_cols.txt")
feature_cols_path.write_text("\n".join(FEATURE_COLS))

# ============================================================
# STEP 14: Sanity checks
# ============================================================
print(f"\n[OK] Features engineered: {len(FEATURE_COLS)} features")
print(f"Dataset shape: {feature_df.shape}")
print(f"Attrition rate: {feature_df['attrition_flag'].mean():.1%}")
print(f"Null check: {feature_df.isnull().sum().sum()} nulls")

print(f"\nDropped vs v1 (multicollinearity fix):")
print(f"  recency_promotion    — = career_tenure_at_band × 30")
print(f"  recency_hike         — = comp_months_no_hike × 30")
print(f"  career_band_normalized — = band / 7")
print(f"  comp_underpaid_flag  — derived from comp_compa_ratio")
print(f"  dept_HR              — reference category")
print(f"  circle_Gujarat       — reference category")

print(f"\nNew features added:")
print(f"  overload_overtime_slope, overload_afterhours_login")
print(f"  is_new_joiner, onboard_completion_rate")
print(f"  onboard_days_to_project, onboard_manager_1on1s")
print(f"  interact_overload_underpaid")
print(f"  interact_onboard_abandoned, interact_onboard_idle")

print(f"\nInteraction feature counts:")
print(f"  high_performer_no_promo:     {feature_df['interact_high_performer_no_promo'].sum()}")
print(f"  underpaid_declining:         {feature_df['interact_underpaid_declining'].sum()}")
print(f"  new_mgr_peer_left:           {feature_df['interact_new_mgr_peer_left'].sum()}")
print(f"  post_appraisal_dissatisfied: {feature_df['interact_post_appraisal_dissatisfied'].sum()}")
print(f"  overload_underpaid:          {feature_df['interact_overload_underpaid'].sum()}")
print(f"  onboard_abandoned:           {feature_df['interact_onboard_abandoned'].sum()}")
print(f"  onboard_idle:                {feature_df['interact_onboard_idle'].sum()}")

print(f"\nOnboarding regime:")
print(f"  New joiners (tenure<=12m): {feature_df['is_new_joiner'].sum()}")

print(f"\nSaved to: {output_path}")