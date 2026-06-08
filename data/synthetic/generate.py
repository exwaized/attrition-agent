# ============================================================
# generate.py — Synthetic Employee Dataset Generator
# ============================================================
# PURPOSE: Creates realistic Jio-like employee event log
# FLOW: config → faker profiles → temporal events → CSV output
# All downstream files read from this CSV
# ============================================================

import pandas as pd
import numpy as np
from faker import Faker
import yaml
import random
from pathlib import Path

# --- Step 1: Load config ---
# Single source of truth — all params come from config.yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Fix random seeds so dataset is identical every run
np.random.seed(cfg["data"]["random_seed"])
random.seed(cfg["data"]["random_seed"])
fake = Faker("en_IN")
Faker.seed(cfg["data"]["random_seed"])

N = cfg["data"]["n_employees"]
ATTRITION_RATE = cfg["data"]["attrition_rate"]

# --- Step 2: Define Jio org structure ---
CIRCLES = ["Mumbai", "Delhi", "Gujarat", "Rajasthan", "Maharashtra",
           "Karnataka", "Tamil Nadu", "West Bengal", "Punjab", "UP East"]
DEPARTMENTS = ["Network", "Technology", "Sales", "HR", "Finance", "Operations", "IT"]
BANDS = [3, 4, 5, 6, 7]
BAND_WEIGHTS = [0.30, 0.35, 0.20, 0.10, 0.05]

# --- Step 3: Define attrition risk profiles ---
# Each profile has a base_risk multiplier
# Mirrors real org dynamics — not all employees have equal flight risk
# Noise added per employee so labels aren't perfectly deterministic
RISK_PROFILES = {
    "high_performer_stuck":  0.45,
    "underpaid_active":      0.38,
    "disengaged_declining":  0.52,
    "new_joiner":            0.22,
    "stable_tenured":        0.06,
}

def assign_profile(band, tenure_months, compa_ratio, perf_rating):
    """
    Rule-based profile assignment mirrors how HR categorizes employees.
    Profile drives base_risk which drives attrition label generation.
    """
    if perf_rating >= 4.0 and tenure_months > 18 and compa_ratio < 0.90:
        return "high_performer_stuck"
    elif compa_ratio < 0.85 and perf_rating >= 3.5:
        return "underpaid_active"
    elif tenure_months < 12:
        return "new_joiner"
    elif perf_rating < 3.0:
        return "disengaged_declining"
    else:
        return "stable_tenured"

# --- Step 4: Generate base employee records ---
print(f"Generating {N} employee records...")
records = []

for i in range(N):
    band          = np.random.choice(BANDS, p=BAND_WEIGHTS)
    tenure_months = np.random.randint(1, 180)
    compa_ratio   = np.clip(np.random.normal(0.97, 0.12), 0.65, 1.35)
    perf_rating   = np.clip(np.random.normal(3.4, 0.7), 1.0, 5.0)

    months_since_promotion     = np.random.randint(0, min(tenure_months + 1, 48))
    months_since_hike          = np.random.randint(0, 24)
    manager_changed_months_ago = np.random.choice(
        [np.random.randint(1, 6), np.random.randint(6, 36)],
        p=[0.25, 0.75]
    )

    profile   = assign_profile(band, tenure_months, compa_ratio, perf_rating)
    base_risk = RISK_PROFILES[profile]

    # --- Realistic noise injection ---
    # Larger noise (0.15) vs original (0.05) makes labels less deterministic
    # Real attrition has unexplained variance — model shouldn't be too perfect
    noise          = np.random.normal(0, 0.15)
    attrition_prob = np.clip(base_risk + noise, 0.01, 0.95)

    # 8% random label flip — mirrors real-world mislabeling
    # Even HR experts can't perfectly label ~10% of borderline cases
    if np.random.random() < 0.08:
        attrition_prob = np.clip(1 - attrition_prob, 0.01, 0.95)

    attrition_flag = int(np.random.random() < attrition_prob)

    # --- Behavioral features ---
    # Leavers show lower activity — but not perfectly so (noise included)
    activity_multiplier = 0.60 if attrition_flag else 1.0

    login_freq_30d           = np.clip(np.random.normal(18 * activity_multiplier, 4), 0, 25)
    training_completions_90d = np.random.poisson(max(0.5, 2.5 * activity_multiplier))
    leave_days_30d           = np.clip(np.random.normal(2.5 / activity_multiplier, 1.5), 0, 10)

    # Trend features — negative slope for leavers but with noise
    login_slope        = np.random.normal(-0.6 * attrition_flag + 0.15 * (1 - attrition_flag), 0.4)
    performance_delta  = np.random.normal(-0.3 * attrition_flag + 0.1  * (1 - attrition_flag), 0.35)

    # Org context features
    peer_attrition_count  = np.random.poisson(2.0 if attrition_flag else 0.8)
    manager_attrition_rate= np.random.beta(2 if attrition_flag else 1, 5)
    team_size_change      = np.random.normal(-0.10 * attrition_flag, 0.09)

    hike_pct         = np.random.normal(7.5, 2.5)
    band_median_hike = 8.0
    hike_vs_median   = hike_pct - band_median_hike

    records.append({
        "employee_id":                  f"JIO{10000 + i}",
        "band":                         band,
        "circle":                       np.random.choice(CIRCLES),
        "department":                   np.random.choice(DEPARTMENTS),
        "gender":                       np.random.choice(["M", "F"], p=[0.62, 0.38]),
        "tenure_months":                tenure_months,
        "months_since_promotion":       months_since_promotion,
        "months_since_hike":            months_since_hike,
        "months_since_manager_changed": manager_changed_months_ago,
        "compa_ratio":                  round(compa_ratio, 3),
        "perf_rating_current":          round(perf_rating, 2),
        "hike_pct":                     round(hike_pct, 2),
        "hike_vs_band_median":          round(hike_vs_median, 2),
        "login_freq_30d":               round(login_freq_30d, 2),
        "training_completions_90d":     training_completions_90d,
        "leave_days_30d":               round(leave_days_30d, 2),
        "login_freq_slope_30d":         round(login_slope, 3),
        "performance_rating_delta":     round(performance_delta, 3),
        "peer_attrition_count_90d":     peer_attrition_count,
        "manager_attrition_rate_6m":    round(manager_attrition_rate, 3),
        "team_size_change_pct":         round(team_size_change, 3),
        "risk_profile":                 profile,
        "attrition_prob":               round(attrition_prob, 3),
        "attrition_flag":               attrition_flag,
    })

# --- Step 5: Save ---
df = pd.DataFrame(records)
output_path = Path(cfg["data"]["output_path"])
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# --- Step 6: Sanity checks ---
print(f"\n✅ Dataset generated: {len(df)} employees")
print(f"Attrition rate: {df['attrition_flag'].mean():.1%} (target: ~{ATTRITION_RATE:.0%})")
print(f"Band distribution:\n{df['band'].value_counts().sort_index()}")
print(f"Risk profiles:\n{df['risk_profile'].value_counts()}")
print(f"Attrition by profile:")
print(df.groupby("risk_profile")["attrition_flag"].mean().round(3))
print(f"\nSaved to: {output_path}")