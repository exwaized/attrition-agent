"""
Unit tests for the pure logic extracted from train.py:
temporal_split, evaluate_model, select_champion.

These run on tiny synthetic data, not the real feature set — fast,
and don't need features.csv or a real XGBoost/LR fit to exist.

What's deliberately NOT covered here: the actual XGBoost/LR fitting,
SHAP, Cox PH, and KM curve generation inside main(). Mocking
sklearn/SHAP/lifelines internals would mostly test the mocks, not
your code — that logic stays covered by actually running the
pipeline end-to-end (which you already do, and which run_pipeline.py's
checkpoint recovery protects).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from models.train import evaluate_model, select_champion, temporal_split

# ---------- temporal_split ----------

def _toy_df(n=20):
    return pd.DataFrame({
        "feature_a": np.arange(n),
        "feature_b": np.arange(n)[::-1],
        # ([0, 1] * k)[:n] instead of [0, 1] * (n // 2) — the latter
        # silently produces a column 1 element short whenever n is odd,
        # which pandas then rejects with "All arrays must be of the
        # same length" instead of a clear test-helper error.
        "attrition_flag": ([0, 1] * ((n // 2) + 1))[:n],
    })


def test_temporal_split_respects_train_frac():
    df = _toy_df(20)
    X_train, y_train, X_test, y_test, train_df, test_df = temporal_split(
        df, ["feature_a", "feature_b"], train_frac=0.80
    )
    assert len(train_df) == 16
    assert len(test_df) == 4
    assert len(X_train) == len(y_train) == 16
    assert len(X_test) == len(y_test) == 4


def test_temporal_split_is_chronological_not_random():
    # Train must be the FIRST rows by index order, test the LAST rows —
    # that's the entire point of this function: no shuffling, no
    # leakage from future employees into training.
    df = _toy_df(10)
    _, _, _, _, train_df, test_df = temporal_split(
        df, ["feature_a", "feature_b"], train_frac=0.80
    )
    assert train_df["feature_a"].tolist() == list(range(8))
    assert test_df["feature_a"].tolist() == list(range(8, 10))


def test_temporal_split_no_row_overlap():
    df = _toy_df(15)
    _, _, _, _, train_df, test_df = temporal_split(df, ["feature_a", "feature_b"])
    assert set(train_df.index).isdisjoint(set(test_df.index))


def test_temporal_split_uses_correct_target_column():
    df = _toy_df(10)
    _, y_train, _, y_test, _, _ = temporal_split(
        df, ["feature_a", "feature_b"], target_col="attrition_flag"
    )
    assert y_train.name == "attrition_flag"
    assert y_test.name == "attrition_flag"


# ---------- evaluate_model ----------

def test_evaluate_model_returns_expected_keys():
    # Not testing model quality here — just that evaluate_model's
    # PR-AUC / ROC-AUC / threshold plumbing is wired correctly.
    X = pd.DataFrame({"x": [0, 0, 0, 1, 1, 1] * 5})
    y = pd.Series([0, 0, 0, 1, 1, 1] * 5)
    model = LogisticRegression().fit(X, y)

    result = evaluate_model(model, X, y, "toy-model", threshold=0.40)

    assert set(result.keys()) == {"name", "pr_auc", "roc_auc", "proba"}
    assert result["name"] == "toy-model"
    assert 0 <= result["pr_auc"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert len(result["proba"]) == len(y)


def test_evaluate_model_perfect_separation_scores_high():
    # Perfectly separable toy data should produce near-1.0 PR-AUC/
    # ROC-AUC — a sanity check that the metric computation isn't
    # inverted or otherwise wired backwards.
    X = pd.DataFrame({"x": [0] * 10 + [1] * 10})
    y = pd.Series([0] * 10 + [1] * 10)
    model = LogisticRegression().fit(X, y)

    result = evaluate_model(model, X, y, "perfect-separation")
    assert result["pr_auc"] > 0.95
    assert result["roc_auc"] > 0.95


# ---------- select_champion ----------

def test_select_champion_xgboost_wins_clear_gap():
    champion_name, gap = select_champion({"pr_auc": 0.90}, {"pr_auc": 0.85})
    assert champion_name == "XGBoost"
    assert gap == pytest.approx(0.05)


def test_select_champion_lr_wins_when_gap_too_small():
    champion_name, _ = select_champion({"pr_auc": 0.86}, {"pr_auc": 0.85})
    assert champion_name == "LogisticRegression"


def test_select_champion_boundary_exactly_at_threshold_goes_to_lr():
    # pr_gap > gap_threshold is required, not >=, so a gap EXACTLY at
    # the threshold should NOT promote XGBoost. Constructing
    # gap_threshold from the same subtraction (rather than hardcoding
    # 0.02) avoids a floating-point trap: 0.87 - 0.85 == 0.02 is False
    # in IEEE754 (it's 0.020000000000000018), which would make this
    # test flaky for the wrong reason.
    xgb_pr_auc, lr_pr_auc = 0.87, 0.85
    gap_threshold = xgb_pr_auc - lr_pr_auc

    champion_name, gap = select_champion(
        {"pr_auc": xgb_pr_auc}, {"pr_auc": lr_pr_auc}, gap_threshold=gap_threshold
    )
    assert gap == gap_threshold
    assert champion_name == "LogisticRegression"


def test_select_champion_lr_wins_when_strictly_better():
    champion_name, gap = select_champion({"pr_auc": 0.70}, {"pr_auc": 0.85})
    assert champion_name == "LogisticRegression"
    assert gap < 0