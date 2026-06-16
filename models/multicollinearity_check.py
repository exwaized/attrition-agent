import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json


# ============================================================
# Pure, testable functions
# ============================================================
# These three are the actual DECISION logic — everything else in
# this file (VIF computation, the correlation heatmap, the L1 Lasso
# fit) needs real data and a real model fit, and stays inside main().
# Mocking statsmodels/sklearn internals to unit test those would
# mostly test the mocks, not your code.
# ============================================================

def find_high_correlation_pairs(corr_matrix, feature_names, threshold=0.7):
    """
    Method 2 — flags feature pairs with |Pearson r| above `threshold`.
    |r| > 0.8 is a multicollinearity risk; |r| > 0.9 is high enough to
    definitely drop one of the pair.
    """
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "r": round(r, 3),
                    "abs_r": round(abs(r), 3),
                })
    return pairs


def select_correlated_feature_to_drop(high_corr_df, vif_df, abs_r_threshold=0.85):
    """
    For each pair correlated above `abs_r_threshold`, flags whichever
    feature has the HIGHER VIF for dropping — the one contributing
    more to overall multicollinearity, not just to this one pairwise
    correlation.
    """
    corr_flagged = set()
    if high_corr_df.empty:
        return corr_flagged

    for _, row in high_corr_df[high_corr_df["abs_r"] > abs_r_threshold].iterrows():
        f1_vif = vif_df[vif_df["feature"] == row["feature_1"]]["VIF"].values
        f2_vif = vif_df[vif_df["feature"] == row["feature_2"]]["VIF"].values
        f1_vif = f1_vif[0] if len(f1_vif) > 0 else 0
        f2_vif = f2_vif[0] if len(f2_vif) > 0 else 0
        if f1_vif > f2_vif:
            corr_flagged.add(row["feature_1"])
        else:
            corr_flagged.add(row["feature_2"])
    return corr_flagged


def recommend_drops(vif_flagged, l1_flagged, corr_flagged, interaction_terms):
    """
    Combined drop recommendation — drop a feature only if 2+ of the
    three methods (VIF, L1, correlation) agree it's redundant, AND
    it isn't an interaction term. Interaction terms are always
    protected, since they capture genuine joint effects rather than
    redundancy — this is the rule that kept interact_underpaid_declining
    and interact_high_performer_no_promo out of your 7 dropped features.
    """
    return ((vif_flagged & l1_flagged) | (vif_flagged & corr_flagged)) - interaction_terms


# ============================================================
# Pipeline orchestration
# ============================================================

def main():
    # --- Step 1: Load features ---
    df = pd.read_csv("data/synthetic/features.csv")
    feature_cols = Path("data/synthetic/feature_cols.txt").read_text().strip().split("\n")
    X = df[feature_cols].copy()
    y = df["attrition_flag"]

    print(f"Loaded {len(df)} records, {len(feature_cols)} features")

    # --- Step 2: Scale features ---
    # VIF and L1 both need scaled features — StandardScaler: mean=0, std=1
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # ============================================================
    # METHOD 1: VIF (Variance Inflation Factor)
    # ============================================================
    # VIF = 1/(1-R²) where R² = how well other features predict this one
    # VIF = 1     : no correlation
    # VIF = 1-5   : moderate — acceptable
    # VIF = 5-10  : high — investigate
    # VIF > 10    : severe — must drop one of collinear pair
    # VIF = inf   : perfect collinearity — two features identical
    # ============================================================
    print("\n" + "="*50)
    print("METHOD 1: VIF Analysis")
    print("="*50)

    vif_scores = []
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(X_scaled.values, i)
            vif_scores.append({"feature": col, "VIF": round(vif, 2)})
        except Exception:
            vif_scores.append({"feature": col, "VIF": float('inf')})

    vif_df = pd.DataFrame(vif_scores).sort_values("VIF", ascending=False)

    print("\nAll VIF scores (sorted):")
    print(vif_df.to_string(index=False))

    critical = vif_df[vif_df["VIF"] > 10]
    high = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    moderate = vif_df[(vif_df["VIF"] > 2) & (vif_df["VIF"] <= 5)]
    clean = vif_df[vif_df["VIF"] <= 2]

    print(f"\nVIF Summary:")
    print(f"  Critical (>10):  {len(critical)} features — MUST FIX")
    print(f"  High (5-10):     {len(high)} features — investigate")
    print(f"  Moderate (2-5):  {len(moderate)} features — acceptable")
    print(f"  Clean (<=2):     {len(clean)} features — good")

    if len(critical) > 0:
        print(f"\nCritical VIF features:")
        print(critical.to_string(index=False))

    # ============================================================
    # METHOD 2: Pearson Correlation Matrix
    # ============================================================
    print("\n" + "="*50)
    print("METHOD 2: Pearson Correlation Analysis")
    print("="*50)

    # Exclude dummy columns for readability
    core_features = [f for f in feature_cols
                     if not f.startswith("dept_")
                     and not f.startswith("circle_")]

    corr_matrix = X[core_features].corr()

    high_corr_pairs = find_high_correlation_pairs(corr_matrix, core_features, threshold=0.7)
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        "abs_r", ascending=False
    ) if high_corr_pairs else pd.DataFrame()

    if not high_corr_df.empty:
        print(f"\nHighly correlated pairs (|r| > 0.7):")
        print(high_corr_df.to_string(index=False))
    else:
        print("\nNo highly correlated pairs (|r| > 0.7)")

    # Plot heatmap
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.3,
        annot_kws={"size": 6}
    )
    plt.title("Feature Correlation Matrix", fontsize=12)
    plt.tight_layout()
    Path("models").mkdir(exist_ok=True)
    plt.savefig("models/correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Correlation heatmap saved: models/correlation_matrix.png")

    # ============================================================
    # METHOD 3: L1 Lasso Feature Selection
    # ============================================================
    # L1 zeroes out redundant features automatically — if two features
    # are collinear, Lasso keeps one, zeroes the other.
    # ============================================================
    print("\n" + "="*50)
    print("METHOD 3: L1 Lasso Feature Selection")
    print("="*50)

    lr_lasso = LogisticRegressionCV(
        cv=5,
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
        Cs=10
    )
    lr_lasso.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "l1_coef": lr_lasso.coef_[0],
        "abs_coef": np.abs(lr_lasso.coef_[0])
    }).sort_values("abs_coef", ascending=False)

    zeroed = coef_df[coef_df["l1_coef"] == 0.0]
    kept = coef_df[coef_df["l1_coef"] != 0.0]

    print(f"\nBest C: {lr_lasso.C_[0]:.4f}")
    print(f"Kept:   {len(kept)} features")
    print(f"Zeroed: {len(zeroed)} features")

    print(f"\nTop 20 by L1 coefficient:")
    print(kept.head(20)[["feature", "l1_coef", "abs_coef"]].to_string(index=False))

    print(f"\nZeroed features (redundant/weak):")
    print(zeroed["feature"].tolist())

    # ============================================================
    # COMBINED DROP RECOMMENDATION
    # ============================================================
    # Drop if flagged by 2+ methods AND not an interaction term.
    # Interaction terms always protected — capture joint effects.
    # ============================================================
    print("\n" + "="*50)
    print("COMBINED DROP RECOMMENDATION")
    print("="*50)

    vif_flagged = set(critical["feature"].tolist())
    l1_flagged = set(zeroed["feature"].tolist())
    corr_flagged = select_correlated_feature_to_drop(high_corr_df, vif_df, abs_r_threshold=0.85)
    interaction_terms = {f for f in feature_cols if f.startswith("interact_")}

    drop_recommended = recommend_drops(vif_flagged, l1_flagged, corr_flagged, interaction_terms)

    print(f"\nFlagged by VIF:         {vif_flagged - interaction_terms}")
    print(f"Flagged by L1:          {l1_flagged - interaction_terms}")
    print(f"Flagged by correlation: {corr_flagged - interaction_terms}")
    print(f"\n[OK] Recommended drops (2+ methods agree):")
    print(f"   {drop_recommended}")
    print(f"\n[WARN]  Interaction terms protected:")
    print(f"   {interaction_terms}")

    features_after_drops = [f for f in feature_cols if f not in drop_recommended]

    recommendations = {
        "vif_critical": list(vif_flagged - interaction_terms),
        "l1_zeroed": list(l1_flagged - interaction_terms),
        "corr_flagged": list(corr_flagged - interaction_terms),
        "recommended_drops": list(drop_recommended),
        "features_after_drops": features_after_drops
    }

    with open("models/multicollinearity_report.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"\nReport saved: models/multicollinearity_report.json")
    print(f"Features before: {len(feature_cols)}")
    print(f"Features after:  {len(features_after_drops)}")
    print(f"\n[OK] Multicollinearity check complete")
    print(f"Run python models/train.py next — it will auto-read the drop list")

    return recommendations


if __name__ == "__main__":
    main()
