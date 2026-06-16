import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# ============================================================
# Pure, testable functions
# ============================================================
# Extracted from the original script body so they can be unit
# tested without loading real data, fitting real models, or
# running matplotlib/SHAP. Everything else lives inside main(),
# guarded by `if __name__ == "__main__"` below, so
# `from models.train import evaluate_model` (or a smoke-test
# `import models.train`) no longer triggers a full training run.
# ============================================================

def temporal_split(df, feature_cols, target_col="attrition_flag", train_frac=0.80):
    """
    Step 2 — Temporal train/test split.

    CRITICAL: split on index order, not randomly. A random split leaks
    future employees into training and inflates metrics — a time-based
    split replicates how the model actually behaves in deployment,
    where you're always scoring people whose outcome hasn't happened yet.
    """
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test, train_df, test_df


def evaluate_model(model, X, y, name, threshold=0.40):
    """
    Step 6 — Evaluate a fitted model.

    PR-AUC is the primary metric, not ROC-AUC — at this level of
    attrition imbalance, ROC-AUC is optimistic; PR-AUC penalises
    correctly for the skew (same reasoning as eval_metric="aucpr"
    on the XGBoost side).
    """
    proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, proba)
    pr_auc = auc(recall, precision)
    roc = roc_auc_score(y, proba)
    preds = (proba >= threshold).astype(int)

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y, preds, target_names=["Stayed", "Left"]))
    return {"name": name, "pr_auc": pr_auc, "roc_auc": roc, "proba": proba}


def select_champion(xgb_metrics, lr_metrics, gap_threshold=0.02):
    """
    Step 7 — Champion-challenger selection.

    XGBoost must beat Logistic Regression by more than `gap_threshold`
    PR-AUC to justify its added complexity (less interpretable, more
    hyperparameters, slower to retrain). Ties or near-ties default to
    LR — same governance logic as the champion-challenger framework in
    your Credit Default project, where the simpler model won whenever
    XGBoost didn't clear the Gini/PR-AUC bar.

    NOTE: strictly `>`, not `>=` — a gap of exactly gap_threshold does
    NOT promote XGBoost.
    """
    pr_gap = xgb_metrics["pr_auc"] - lr_metrics["pr_auc"]
    champion_name = "XGBoost" if pr_gap > gap_threshold else "LogisticRegression"
    return champion_name, pr_gap


# ============================================================
# Pipeline orchestration
# ============================================================
# Everything that needs real data, real model fitting, and disk
# I/O. Wrapped in main() so importing this module is safe and
# instant; only `python models/train.py` actually trains.
# ============================================================

def main():
    # --- Step 1: Load config and features ---
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv("data/synthetic/features.csv")
    feature_cols = Path("data/synthetic/feature_cols.txt").read_text().strip().split("\n")
    print(f"Loaded {len(df)} records, {len(feature_cols)} features")

    mlflow.set_experiment("attrition-agent")
    with mlflow.start_run(run_name=f"train_{datetime.now():%Y%m%d_%H%M%S}"):
        # Log the config that drove this run — same registry instinct
        # as the MLflow setup in your GCP Recommender project, just
        # pointed at this pipeline instead of ALS hyperparameters.
        mlflow.log_params({
            "n_estimators": cfg["model"]["xgboost"]["n_estimators"],
            "max_depth": cfg["model"]["xgboost"]["max_depth"],
            "learning_rate": cfg["model"]["xgboost"]["learning_rate"],
            "cox_ph_penalizer": cfg["model"]["cox_ph"]["penalizer"],
            "decision_threshold": cfg["model"]["threshold"],
            "train_frac": 0.80,
            "n_features": len(feature_cols),
        })

        # --- Step 2: Temporal train/test split ---
        X_train, y_train, X_test, y_test, train_df, test_df = temporal_split(df, feature_cols)
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"Train attrition: {y_train.mean():.1%} | Test attrition: {y_test.mean():.1%}")
        mlflow.log_metric("train_attrition_rate", y_train.mean())
        mlflow.log_metric("test_attrition_rate", y_test.mean())

        # --- Step 3: Scale features for Logistic Regression ---
        # LR needs scaled features to converge properly. XGBoost is
        # tree-based — scaling doesn't affect it. Scaler is fit on
        # X_train ONLY, then applied to X_test — fitting on combined
        # data would leak test-set statistics into training.
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        X_train_sc = pd.DataFrame(X_train_sc, columns=feature_cols)
        X_test_sc = pd.DataFrame(X_test_sc, columns=feature_cols)

        # --- Step 4: Champion — XGBoost ---
        # scale_pos_weight handles class imbalance without SMOTE —
        # value = n_negative / n_positive
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\nClass imbalance ratio: {pos_weight:.2f}")
        mlflow.log_metric("class_imbalance_ratio", pos_weight)

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
        # Uses scaled features — fixes convergence warning. max_iter=3000
        # gives LR enough room to converge; saga handles large datasets
        # better than lbfgs.
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="saga",
            random_state=42
        )
        lr.fit(X_train_sc, y_train)

        # --- Step 6: Evaluate both ---
        xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost (Champion)")
        lr_metrics = evaluate_model(lr, X_test_sc, y_test, "LogisticRegression (Challenger)")
        mlflow.log_metrics({
            "xgb_pr_auc": xgb_metrics["pr_auc"],
            "xgb_roc_auc": xgb_metrics["roc_auc"],
            "lr_pr_auc": lr_metrics["pr_auc"],
            "lr_roc_auc": lr_metrics["roc_auc"],
        })

        # --- Step 7: Champion selection ---
        champion_name, pr_gap = select_champion(xgb_metrics, lr_metrics)
        print(f"\nPR-AUC gap: {pr_gap:.4f}")
        mlflow.log_metric("pr_auc_gap", pr_gap)
        mlflow.log_param("champion", champion_name)

        if champion_name == "XGBoost":
            champion = xgb
            X_train_champ = X_train
            X_test_champ = X_test
            print("[OK] Champion: XGBoost")
        else:
            champion = lr
            X_train_champ = X_train_sc
            X_test_champ = X_test_sc
            print("[OK] Champion: LogisticRegression")

        # --- Step 8: SHAP values ---
        # SHAP = why the model made each prediction. Force float64 —
        # fixes the rint TypeError seen on newer numpy versions.
        print("\nComputing SHAP values...")
        if champion_name == "XGBoost":
            explainer = shap.TreeExplainer(champion)
            shap_values = explainer.shap_values(X_test_champ)
            shap_values = np.array(shap_values, dtype=np.float64)
        else:
            explainer = shap.LinearExplainer(champion, X_train_champ)
            shap_values = explainer.shap_values(X_test_champ)
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
        Path("models").mkdir(exist_ok=True)
        plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP summary saved")
        mlflow.log_artifact("models/shap_summary.png")

        # --- Step 9: Cox PH survival model ---
        # Answers WHEN will an employee leave, not just IF. Handles
        # censoring — employees still employed contribute partial info.
        print("\nFitting Cox PH...")
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
        mlflow.log_metric("cox_ph_concordance", cph.concordance_index_)

        # --- Step 10: Kaplan-Meier curves ---
        # KM shows survival probability over time per group. Log-rank
        # test proves the curves are statistically different.
        print("\nFitting KM curves...")
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(10, 6))

        # risk_profile dropped from features.csv — use attrition_flag as proxy
        high_risk = train_df["attrition_flag"] == 1
        low_risk = train_df["attrition_flag"] == 0

        kmf.fit(train_df.loc[high_risk, "tenure_months"], train_df.loc[high_risk, "attrition_flag"], label="High Risk")
        kmf.plot_survival_function(ax=ax, ci_show=True)
        kmf.fit(train_df.loc[low_risk, "tenure_months"], train_df.loc[low_risk, "attrition_flag"], label="Low Risk")
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
        mlflow.log_metric("km_logrank_p", results.p_value)
        mlflow.log_artifact("models/km_curves.png")

        # --- Step 11: Save all models ---
        # Save scaler too — needed for LR predictions at inference time
        with open("models/champion_model.pkl", "wb") as f:
            pickle.dump(champion, f)
        with open("models/cox_model.pkl", "wb") as f:
            pickle.dump(cph, f)
        with open("models/explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("models/champion_name.txt", "w") as f:
            f.write(champion_name)

        metadata = {
            "champion": champion_name,
            "pr_auc": round(xgb_metrics["pr_auc"] if champion_name == "XGBoost" else lr_metrics["pr_auc"], 4),
            "roc_auc": round(xgb_metrics["roc_auc"] if champion_name == "XGBoost" else lr_metrics["roc_auc"], 4),
            "threshold": cfg["model"]["threshold"],
            "feature_count": len(feature_cols),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "pr_auc_gap": round(pr_gap, 4),
            "scaled": champion_name == "LogisticRegression"
        }
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact("models/metadata.json")

        # Step 12 — Register the champion itself, not just its metrics.
        # This is what makes it a model REGISTRY rather than just a
        # metrics logger: mlflow.xgboost/mlflow.sklearn package the
        # model with its signature so it can be reloaded with
        # mlflow.pyfunc.load_model() later — same swap-readiness
        # pattern as MLflow standing in for Vertex AI Model Registry
        # in your GCP Recommender project.
        if champion_name == "XGBoost":
            mlflow.xgboost.log_model(champion, "champion_model")
        else:
            mlflow.sklearn.log_model(champion, "champion_model")

        print(f"\n{'='*40}")
        print("[OK] Training complete")
        print(f"Champion:  {champion_name}")
        print(f"PR-AUC:    {metadata['pr_auc']}")
        print(f"ROC-AUC:   {metadata['roc_auc']}")
        print("Models saved to models/")


if __name__ == "__main__":
    main()