# ============================================================
# feature_engineering.py — Feature Engineering Pipeline
# ============================================================
# PURPOSE: Transforms raw employee records into ML-ready features
# FLOW: reads employees.csv → engineers 50 features → saves features.csv
# CONNECTED TO: generate.py (input) → train.py (output)
# KEY UPDATE: peer_attrition now computed department + circle filtered
#             not random synthetic count
# ============================================================
!pip install statsmodels
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
# Not company-wide — contagion is local, not global
# Each person counts how many dept+circle colleagues
# are leavers (attrition_flag=1) excluding themselves
# ============================================================

def compute_peer_attrition(df):
    """
    For each employee counts leavers in same dept + circle.
    This is the real contagion signal — not random noise.
    Same dept + circle = colleagues you actually see daily.
    Excludes self — employee doesn't count their own leaving.
    """
    peer_counts = []

    for idx, emp in df.iterrows():
        # Filter: same department AND same circle, not self
        peers = df[
            (df["department"] == emp["department"]) &
            (df["circle"]     == emp["circle"]) &
            (df.index         != idx)
        ]
        # Count peers who are flagged as leavers
        # In real data: filter by attrition_date within 90 days
        # In synthetic: attrition_flag is our proxy
        peer_attrition_count = peers["attrition_flag"].sum()
        peer_counts.append(int(peer_attrition_count))

    return peer_counts

print("Computing peer attrition (department + circle filtered)...")
df["peer_attrition_count_90d"] = compute_peer_attrition(df)
print(f"Peer attrition computed — avg: {df['peer_attrition_count_90d'].mean():.2f}")

# --- Step 2: Recency Features ---
# How recently did the employee do something meaningful
# Higher values = longer since last positive event = disengagement
df["recency_promotion"]      = df["months_since_promotion"] * 30
df["recency_hike"]           = df["months_since_hike"] * 30
df["recency_manager_change"] = df["months_since_manager_changed"] * 30

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
# Normalise leave by tenure — new joiners naturally take less leave
df["volatility_leave"]       = df["leave_days_30d"] / (df["tenure_months"] + 1)
df["volatility_performance"] = df["performance_rating_delta"].abs()

# --- Step 5: Org Context Features ---
# What's happening AROUND the employee affects their decision to stay
# Peer attrition now uses real dept+circle filtered counts
# Manager attrition rate = how often managers leave this team
df["org_peer_attrition"]    = df["peer_attrition_count_90d"]
df["org_manager_attrition"] = df["manager_attrition_rate_6m"]
df["org_team_shrink"]       = (df["team_size_change_pct"] < -0.10).astype(int)

# --- Step 6: Compensation Features ---
# Pay dissatisfaction is most actionable and measurable driver
# Compa-ratio < 0.85 = underpaid 15%+ vs market = active flight risk
df["comp_compa_ratio"]    = df["compa_ratio"]
df["comp_hike_vs_median"] = df["hike_vs_band_median"]
df["comp_underpaid_flag"] = (df["compa_ratio"] < 0.85).astype(int)
df["comp_months_no_hike"] = df["months_since_hike"]

# --- Step 7: Career Progression Features ---
# Stagnation is slow but reliable attrition driver
# Promotion velocity proxy: tenure / band — high = moving slowly
df["career_tenure_at_band"]     = df["months_since_promotion"]
df["career_band_normalized"]    = df["band"] / 7
df["career_promotion_velocity"] = df["tenure_months"] / (df["band"] - 2)

# --- Step 8: Interaction Features ---
# Combinations more predictive than individual features alone
# These are the features that surprise interviewers

# Strongest predictor: high performer + no promotion > 18 months
# High contribution + zero recognition = maximum flight risk
df["interact_high_performer_no_promo"] = (
    (df["perf_rating_current"] >= 4.0) &
    (df["months_since_promotion"] > 18)
).astype(int)

# Double signal: underpaid AND activity declining simultaneously
# Pay gap converting to leaving behaviour = most urgent case
df["interact_underpaid_declining"] = (
    (df["compa_ratio"] < 0.85) &
    (df["login_freq_slope_30d"] < 0)
).astype(int)

# Instability + contagion: new manager AND peer left recently
# Destabilisation + social proof = volatile departure window
df["interact_new_mgr_peer_left"] = (
    (df["months_since_manager_changed"] < 3) &
    (df["peer_attrition_count_90d"] > 1)
).astype(int)

# Post-appraisal dissatisfaction window
# 60 days post-appraisal = highest voluntary resignation rate
df["interact_post_appraisal_dissatisfied"] = (
    (df["hike_vs_band_median"] < -1.5) &
    (df["months_since_hike"] < 3)
).astype(int)

# --- Step 9: Encode categoricals ---
# ML models need numeric input
# get_dummies avoids ordinal assumption on nominal variables
df["gender_encoded"] = (df["gender"] == "M").astype(int)
dept_dummies         = pd.get_dummies(df["department"], prefix="dept")
circle_dummies       = pd.get_dummies(df["circle"],     prefix="circle")
df                   = pd.concat([df, dept_dummies, circle_dummies], axis=1)

# --- Step 10: Define final feature set ---
# Explicit list — no implicit column selection
# Adding a feature = add here. Removing = remove here.
FEATURE_COLS = [
    # Recency family
    "recency_promotion", "recency_hike", "recency_manager_change",
    # Trend family
    "trend_login", "trend_performance", "trend_training",
    # Volatility family
    "volatility_leave", "volatility_performance",
    # Org context family — peer_attrition now dept+circle filtered
    "org_peer_attrition", "org_manager_attrition", "org_team_shrink",
    # Compensation family
    "comp_compa_ratio", "comp_hike_vs_median", "comp_underpaid_flag",
    "comp_months_no_hike",
    # Career family
    "career_tenure_at_band", "career_band_normalized",
    "career_promotion_velocity",
    # Interaction features
    "interact_high_performer_no_promo", "interact_underpaid_declining",
    "interact_new_mgr_peer_left", "interact_post_appraisal_dissatisfied",
    # Base numeric features
    "band", "tenure_months", "perf_rating_current", "login_freq_30d",
    "training_completions_90d", "leave_days_30d", "gender_encoded",
] + list(dept_dummies.columns) + list(circle_dummies.columns)

# --- Step 11: Save features + labels ---
# employee_id kept for agent layer to look up individuals
feature_df = df[
    ["employee_id"] + FEATURE_COLS +
    ["attrition_flag", "attrition_prob", "risk_profile"]
]

output_path = Path("data/synthetic/features.csv")
feature_df.to_csv(output_path, index=False)

# Save feature column list — train.py reads this
feature_cols_path = Path("data/synthetic/feature_cols.txt")
feature_cols_path.write_text("\n".join(FEATURE_COLS))

# --- Step 12: Sanity checks ---
print(f"\n✅ Features engineered: {len(FEATURE_COLS)} features")
print(f"Dataset shape: {feature_df.shape}")
print(f"Attrition rate: {feature_df['attrition_flag'].mean():.1%}")
print(f"Null check: {feature_df.isnull().sum().sum()} nulls")
print(f"\nPeer attrition (dept+circle filtered):")
print(f"  Mean:   {df['peer_attrition_count_90d'].mean():.2f}")
print(f"  Max:    {df['peer_attrition_count_90d'].max()}")
print(f"  Median: {df['peer_attrition_count_90d'].median():.0f}")
print(f"\nInteraction feature counts:")
print(f"  high_performer_no_promo:     {feature_df['interact_high_performer_no_promo'].sum()}")
print(f"  underpaid_declining:         {feature_df['interact_underpaid_declining'].sum()}")
print(f"  new_mgr_peer_left:           {feature_df['interact_new_mgr_peer_left'].sum()}")
print(f"  post_appraisal_dissatisfied: {feature_df['interact_post_appraisal_dissatisfied'].sum()}")
print(f"\nSaved to: {output_path}")