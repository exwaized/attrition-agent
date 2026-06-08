# ============================================================
# train.py — Champion-Challenger Training Pipeline
# ============================================================
# PURPOSE: Trains XGBoost + Cox PH, validates both, saves best model
# FLOW: features.csv → train/test split → XGBoost → Cox PH → champion
# CONNECTED TO: feature_engineering.py (input) → agents/ (output)
# ============================================================

import pandas as pd
import numpy as np
import yaml
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Step 1: Load config and features ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv("data/synthetic/features.csv")
feature_cols = Path("data/synthetic/feature_cols.txt").read_text().strip().split("\n")
print(f"Loaded {len(df)} records, {len(feature_cols)} features")

# --- Step 2: Temporal train/test split ---
# CRITICAL: split on index order not randomly
# Random split leaks future employees into training — inflates metrics
# Time-based split replicates how model works in real deployment
split_idx = int(len(df) * 0.80)
train_df  = df.iloc[:split_idx].copy()
test_df   = df.iloc[split_idx:].copy()

X_train = train_df[feature_cols]
y_train = train_df["attrition_flag"]
X_test  = test_df[feature_cols]
y_test  = test_df["attrition_flag"]

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Train attrition: {y_train.mean():.1%} | Test attrition: {y_test.mean():.1%}")

# --- Step 3: Scale features for Logistic Regression ---
# LR needs scaled features to converge properly
# XGBoost is tree-based — scaling doesn't affect it
# We scale for LR but keep unscaled for XGBoost
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Convert back to DataFrame — keeps feature names for SHAP
X_train_sc = pd.DataFrame(X_train_sc, columns=feature_cols)
X_test_sc  = pd.DataFrame(X_test_sc,  columns=feature_cols)

# --- Step 4: Champion — XGBoost ---
# scale_pos_weight handles class imbalance without SMOTE
# value = n_negative / n_positive
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: {pos_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=cfg["model"]["xgboost"]["n_estimators"],
    max_depth=cfg["model"]["xgboost"]["max_depth"],
    learning_rate=cfg["model"]["xgboost"]["learning_rate"],
    scale_pos_weight=pos_weight,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)

# --- Step 5: Challenger — Logistic Regression ---
# Uses scaled features — fixes convergence warning
# max_iter=3000 gives LR enough room to converge
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=3000,        # increased from 1000 — fixes convergence warning
    solver="saga",        # saga handles large datasets better than lbfgs
    random_state=42
)
lr.fit(X_train_sc, y_train)

# --- Step 6: Evaluate both ---
# PR-AUC is primary metric not ROC-AUC
# At 23% attrition ROC-AUC is optimistic — PR-AUC penalises correctly
def evaluate_model(model, X, y, name, threshold=0.40):
    proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, proba)
    pr_auc = auc(recall, precision)
    roc    = roc_auc_score(y, proba)
    preds  = (proba >= threshold).astype(int)

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y, preds, target_names=["Stayed", "Left"]))
    return {"name": name, "pr_auc": pr_auc, "roc_auc": roc, "proba": proba}

xgb_metrics = evaluate_model(xgb, X_test,    y_test, "XGBoost (Champion)")
lr_metrics  = evaluate_model(lr,  X_test_sc, y_test, "LogisticRegression (Challenger)")

# --- Step 7: Champion selection ---
# XGBoost must beat LR by >2% PR-AUC to justify added complexity
pr_gap = xgb_metrics["pr_auc"] - lr_metrics["pr_auc"]
print(f"\nPR-AUC gap: {pr_gap:.4f}")

if pr_gap > 0.02:
    champion       = xgb
    champion_name  = "XGBoost"
    champion_proba = xgb_metrics["proba"]
    X_train_champ  = X_train
    X_test_champ   = X_test
    print(f"✅ Champion: XGBoost")
else:
    champion       = lr
    champion_name  = "LogisticRegression"
    champion_proba = lr_metrics["proba"]
    X_train_champ  = X_train_sc
    X_test_champ   = X_test_sc
    print(f"✅ Champion: LogisticRegression")

# --- Step 8: SHAP values ---
# SHAP = why model made each prediction
# Force numpy float64 — fixes rint TypeError on newer numpy versions
print(f"\nComputing SHAP values...")

if champion_name == "XGBoost":
    explainer   = shap.TreeExplainer(champion)
    shap_values = explainer.shap_values(X_test_champ)
    shap_values = np.array(shap_values, dtype=np.float64)
else:
    explainer   = shap.LinearExplainer(champion, X_train_champ)
    shap_values = explainer.shap_values(X_test_champ)
    # Force float64 numpy array — fixes rint TypeError
    shap_values = np.array(shap_values, dtype=np.float64)
    # If 3D (binary classification returns 2 arrays) — take positive class
    if shap_values.ndim == 3:
        shap_values = shap_values[1]

print(f"SHAP values shape: {shap_values.shape}")
print(f"SHAP dtype: {shap_values.dtype}")

shap.summary_plot(
    shap_values, X_test_champ,
    feature_names=feature_cols,
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP summary saved")

# --- Step 9: Cox PH survival model ---
# Answers WHEN will employee leave not just IF
# Handles censoring — employees still employed contribute partial info
print(f"\nFitting Cox PH...")
cox_features = [
    "tenure_months", "comp_compa_ratio", "career_tenure_at_band",
    "interact_high_performer_no_promo", "interact_underpaid_declining",
    "org_peer_attrition", "org_manager_attrition",
    "trend_login", "trend_performance", "band"
]

cox_train = train_df[cox_features + ["tenure_months", "attrition_flag"]].copy()
cox_train.columns = list(cox_features) + ["duration", "event"]

cph = CoxPHFitter(penalizer=cfg["model"]["cox_ph"]["penalizer"])
cph.fit(cox_train, duration_col="duration", event_col="event")
print("\nCox PH Summary:")
cph.print_summary(decimals=3)

# --- Step 10: Kaplan-Meier curves ---
# KM shows survival probability over time per risk group
# Log-rank test proves curves are statistically different
print(f"\nFitting KM curves...")
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))

high_risk = train_df["risk_profile"].isin(["high_performer_stuck", "disengaged_declining"])
low_risk  = train_df["risk_profile"].isin(["stable_tenured"])

kmf.fit(train_df.loc[high_risk, "tenure_months"], train_df.loc[high_risk, "attrition_flag"], label="High Risk")
kmf.plot_survival_function(ax=ax, ci_show=True)
kmf.fit(train_df.loc[low_risk, "tenure_months"],  train_df.loc[low_risk,  "attrition_flag"], label="Low Risk")
kmf.plot_survival_function(ax=ax, ci_show=True)

results = logrank_test(
    train_df.loc[high_risk, "tenure_months"], train_df.loc[low_risk, "tenure_months"],
    train_df.loc[high_risk, "attrition_flag"], train_df.loc[low_risk, "attrition_flag"]
)
ax.set_title(f"Kaplan-Meier: High vs Low Risk (log-rank p={results.p_value:.4f})")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Survival Probability")
plt.tight_layout()
plt.savefig("models/km_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"KM curves saved | Log-rank p={results.p_value:.4f}")

# --- Step 11: Save all models ---
# Save scaler too — needed for LR predictions at inference time
Path("models").mkdir(exist_ok=True)

with open("models/champion_model.pkl", "wb") as f: pickle.dump(champion, f)
with open("models/cox_model.pkl",      "wb") as f: pickle.dump(cph, f)
with open("models/explainer.pkl",      "wb") as f: pickle.dump(explainer, f)
with open("models/scaler.pkl",         "wb") as f: pickle.dump(scaler, f)
with open("models/champion_name.txt",  "w") as f: f.write(champion_name)

metadata = {
    "champion":      champion_name,
    "pr_auc":        round(xgb_metrics["pr_auc"] if champion_name == "XGBoost" else lr_metrics["pr_auc"], 4),
    "roc_auc":       round(xgb_metrics["roc_auc"] if champion_name == "XGBoost" else lr_metrics["roc_auc"], 4),
    "threshold":     cfg["model"]["threshold"],
    "feature_count": len(feature_cols),
    "train_size":    len(train_df),
    "test_size":     len(test_df),
    "pr_auc_gap":    round(pr_gap, 4),
    "scaled":        champion_name == "LogisticRegression"
}
with open("models/metadata.json", "w") as f: json.dump(metadata, f, indent=2)

print(f"\n{'='*40}")
print(f"✅ Training complete")
print(f"Champion:  {champion_name}")
print(f"PR-AUC:    {metadata['pr_auc']}")
print(f"ROC-AUC:   {metadata['roc_auc']}")
print(f"Models saved to models/")