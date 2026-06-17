# ============================================================
# generate.py — Synthetic Jio-like HR Dataset Generator
# ============================================================
# PURPOSE: Generates the synthetic employee dataset that the rest
#          of the pipeline (mapper.py, eda.py, feature_engineering.py)
#          expects as raw input.
# FLOW: config.yaml → synthetic employees.csv → data/raw/mapper.py
#
# RECONSTRUCTION NOTE: the file at this path had feature_engineering.py
# v1's content saved into it by mistake, so the original generation
# logic (exact distributions, exact parameters) wasn't recoverable —
# this is a clean rebuild matching the canonical schema mapper.py
# validates against and the department/circle reference categories
# feature_engineering.py drops (dept_HR, circle_Gujarat), with
# attrition_flag driven by a risk score in the same direction as the
# Cox PH findings already established for this project (manager
# attrition strongest driver, declining login trend risky, compa_ratio
# alone weak — it only really bites through the underpaid+declining
# interaction). If a real employees.csv already exists and your
# trained models/SHAP results are based on it, don't re-run this —
# it will produce a different dataset, not reproduce the old one.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Matches dashboard.py's department selectbox exactly, and dept_HR is
# the reference category feature_engineering.py drops from the dummies.
DEPARTMENTS = ["Network", "Technology", "Sales", "HR", "Finance", "Operations", "IT"]

# circle_Gujarat is the reference category feature_engineering.py drops,
# so Gujarat needs to be one of the circles.
CIRCLES = ["Mumbai", "Gujarat", "Karnataka", "Delhi NCR", "Tamil Nadu",
           "UP West", "Punjab", "West Bengal"]

# mapper.py's validate_final flags any band outside this set
BANDS        = [3, 4, 5, 6, 7]
BAND_WEIGHTS = [0.10, 0.32, 0.34, 0.18, 0.06]  # most headcount sits in bands 4-5


def generate_employees(n: int, attrition_rate: float, seed: int) -> pd.DataFrame:
    """
    Generates n synthetic employee records. attrition_flag is driven by
    a weighted risk score, not pure noise — manager attrition rate is
    the strongest term (matches HR=2.907), declining login trend adds
    risk (matches HR=0.282 per unit increase being protective), and
    compa_ratio only contributes through the underpaid+declining
    interaction rather than as a standalone term, since the project's
    own Cox PH fit found compa_ratio alone non-significant (p=0.896).
    Thresholding on the score (rather than a sigmoid) hits the exact
    target rate while still giving SHAP/Cox PH real signal to recover.
    """
    rng = np.random.default_rng(seed)

    band       = rng.choice(BANDS, size=n, p=BAND_WEIGHTS)
    department = rng.choice(DEPARTMENTS, size=n)
    circle     = rng.choice(CIRCLES, size=n)
    gender     = rng.choice(["M", "F"], size=n, p=[0.68, 0.32])

    tenure_months = np.clip(rng.exponential(scale=42, size=n), 1, 300).round(1)

    # Recency features can't exceed how long the person has even been here
    months_since_promotion = np.clip(
        rng.exponential(scale=18, size=n), 0, tenure_months
    ).round(1)
    months_since_hike = np.clip(
        rng.exponential(scale=10, size=n), 0, tenure_months
    ).round(1)
    months_since_manager_changed = np.clip(
        rng.exponential(scale=14, size=n), 0, tenure_months
    ).round(1)

    compa_ratio          = np.clip(rng.normal(1.0, 0.15, size=n), 0.5, 1.6).round(3)
    hike_vs_band_median  = np.clip(rng.normal(0, 1.2, size=n), -5, 5).round(2)
    perf_rating_current  = np.clip(rng.normal(3.3, 0.7, size=n), 1, 5).round(1)

    login_freq_30d            = np.clip(rng.poisson(19, size=n), 0, 31).astype(float)
    login_freq_slope_30d      = np.clip(rng.normal(0, 0.6, size=n), -3, 3).round(2)
    training_completions_90d  = rng.poisson(1.5, size=n).astype(float)
    leave_days_30d            = np.clip(rng.poisson(1.2, size=n), 0, 15).astype(float)
    performance_rating_delta  = np.clip(rng.normal(0, 0.4, size=n), -2, 2).round(2)

    manager_attrition_rate_6m = np.clip(rng.beta(1.5, 8, size=n), 0, 1).round(3)
    team_size_change_pct      = np.clip(rng.normal(0, 0.12, size=n), -0.6, 0.6).round(3)

    # Placeholder — feature_engineering.py recomputes this properly via
    # the dept+circle filtered compute_peer_attrition(); whatever value
    # sits here gets overwritten downstream, it just needs to exist.
    peer_attrition_count_90d = rng.poisson(1.0, size=n).astype(float)

    # ---------- Latent risk score driving attrition_flag ----------
    high_performer_no_promo = (
        (perf_rating_current >= 4.0) & (months_since_promotion > 18)
    ).astype(float)
    underpaid_declining = (
        (compa_ratio < 0.85) & (login_freq_slope_30d < 0)
    ).astype(float)
    post_appraisal_dissatisfied = (
        (hike_vs_band_median < -1.5) & (months_since_hike < 3)
    ).astype(float)
    new_joiner = (tenure_months < 12).astype(float)

    risk_score = (
        2.9 * manager_attrition_rate_6m
        - 0.8 * login_freq_slope_30d
        + 0.9 * high_performer_no_promo
        + 0.7 * underpaid_declining
        + 0.6 * post_appraisal_dissatisfied
        + 0.015 * months_since_hike
        + 0.10 * new_joiner
        + rng.normal(0, 1.0, size=n)  # noise — keeps this from being a deterministic rule
    )

    # Threshold to hit the exact target rate rather than just an
    # expected one — sort descending, cut at the n-th position.
    n_attrition = round(n * attrition_rate)
    threshold   = np.sort(risk_score)[::-1][max(n_attrition - 1, 0)]
    attrition_flag = (risk_score >= threshold).astype(int)

    # Continuous risk score rescaled to [0,1] — optional columns,
    # feature_engineering.py only includes them if present.
    attrition_prob = (
        (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    ).round(4)
    risk_profile = pd.cut(
        attrition_prob, bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Stable", "At Risk", "Critical"]
    ).astype(str)

    employee_id = [f"JIO{10000 + i}" for i in range(n)]

    return pd.DataFrame({
        "employee_id":                  employee_id,
        "band":                         band,
        "department":                   department,
        "circle":                       circle,
        "gender":                       gender,
        "tenure_months":                tenure_months,
        "months_since_promotion":       months_since_promotion,
        "months_since_hike":            months_since_hike,
        "months_since_manager_changed": months_since_manager_changed,
        "compa_ratio":                  compa_ratio,
        "hike_vs_band_median":          hike_vs_band_median,
        "perf_rating_current":          perf_rating_current,
        "login_freq_30d":               login_freq_30d,
        "login_freq_slope_30d":         login_freq_slope_30d,
        "training_completions_90d":     training_completions_90d,
        "leave_days_30d":               leave_days_30d,
        "performance_rating_delta":     performance_rating_delta,
        "peer_attrition_count_90d":     peer_attrition_count_90d,
        "manager_attrition_rate_6m":    manager_attrition_rate_6m,
        "team_size_change_pct":         team_size_change_pct,
        "attrition_flag":               attrition_flag,
        "attrition_prob":               attrition_prob,
        "risk_profile":                 risk_profile,
    })


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    n              = cfg["data"].get("n_employees", 1800)
    attrition_rate = cfg["data"]["attrition_rate"]
    seed           = cfg["data"]["random_seed"]
    output_path    = Path(cfg["data"]["output_path"])

    print(f"Generating {n} synthetic employee records "
          f"(seed={seed}, target attrition={attrition_rate:.1%})...")
    df = generate_employees(n, attrition_rate, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Actual attrition rate: {df['attrition_flag'].mean():.1%}")
    print(f"Saved {len(df)} records to: {output_path}")


if __name__ == "__main__":
    main()