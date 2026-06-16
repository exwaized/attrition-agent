"""
Unit tests for the pure decision-logic in multicollinearity_check.py:
find_high_correlation_pairs, select_correlated_feature_to_drop,
recommend_drops.

NOT covered here: the actual VIF computation (statsmodels), the L1
Lasso fit, and the correlation heatmap — these need real fitted
models on real data and stay covered by actually running the check
against features.csv, same reasoning as the model-fitting code in
train.py.
"""
import pandas as pd

from models.multicollinearity_check import (
    find_high_correlation_pairs,
    select_correlated_feature_to_drop,
    recommend_drops,
)


# ---------- find_high_correlation_pairs ----------

def test_find_high_correlation_pairs_above_threshold():
    corr = pd.DataFrame(
        [[1.0, 0.85, 0.1], [0.85, 1.0, 0.2], [0.1, 0.2, 1.0]],
        columns=["a", "b", "c"], index=["a", "b", "c"]
    )
    pairs = find_high_correlation_pairs(corr, ["a", "b", "c"], threshold=0.7)
    assert len(pairs) == 1
    assert pairs[0]["feature_1"] == "a"
    assert pairs[0]["feature_2"] == "b"
    assert pairs[0]["abs_r"] == 0.85


def test_find_high_correlation_pairs_none_above_threshold():
    corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], columns=["a", "b"], index=["a", "b"])
    pairs = find_high_correlation_pairs(corr, ["a", "b"], threshold=0.7)
    assert pairs == []


def test_find_high_correlation_pairs_negative_correlation_counts():
    # abs(r) > threshold, not r > threshold — a strong NEGATIVE
    # correlation is just as much a multicollinearity risk as a
    # strong positive one.
    corr = pd.DataFrame([[1.0, -0.9], [-0.9, 1.0]], columns=["a", "b"], index=["a", "b"])
    pairs = find_high_correlation_pairs(corr, ["a", "b"], threshold=0.7)
    assert len(pairs) == 1
    assert pairs[0]["r"] == -0.9
    assert pairs[0]["abs_r"] == 0.9


# ---------- select_correlated_feature_to_drop ----------

def test_select_correlated_feature_to_drop_picks_higher_vif():
    high_corr_df = pd.DataFrame([
        {"feature_1": "a", "feature_2": "b", "r": 0.9, "abs_r": 0.9},
    ])
    vif_df = pd.DataFrame([
        {"feature": "a", "VIF": 15.0},
        {"feature": "b", "VIF": 3.0},
    ])
    flagged = select_correlated_feature_to_drop(high_corr_df, vif_df, abs_r_threshold=0.85)
    assert flagged == {"a"}


def test_select_correlated_feature_to_drop_below_abs_r_threshold_ignored():
    high_corr_df = pd.DataFrame([
        {"feature_1": "a", "feature_2": "b", "r": 0.75, "abs_r": 0.75},
    ])
    vif_df = pd.DataFrame([
        {"feature": "a", "VIF": 15.0},
        {"feature": "b", "VIF": 3.0},
    ])
    # abs_r (0.75) is below this stage's 0.85 cutoff — a stricter
    # second gate before actually recommending a drop, even though
    # find_high_correlation_pairs already passed it at 0.7.
    flagged = select_correlated_feature_to_drop(high_corr_df, vif_df, abs_r_threshold=0.85)
    assert flagged == set()


def test_select_correlated_feature_to_drop_handles_empty_input():
    flagged = select_correlated_feature_to_drop(pd.DataFrame(), pd.DataFrame(), abs_r_threshold=0.85)
    assert flagged == set()


# ---------- recommend_drops ----------

def test_recommend_drops_requires_two_methods_to_agree():
    drops = recommend_drops(vif_flagged={"x"}, l1_flagged=set(), corr_flagged=set(), interaction_terms=set())
    assert drops == set()


def test_recommend_drops_vif_and_l1_agree():
    drops = recommend_drops(vif_flagged={"x", "y"}, l1_flagged={"x"}, corr_flagged=set(), interaction_terms=set())
    assert drops == {"x"}


def test_recommend_drops_vif_and_corr_agree():
    drops = recommend_drops(vif_flagged={"x"}, l1_flagged=set(), corr_flagged={"x", "z"}, interaction_terms=set())
    assert drops == {"x"}


def test_recommend_drops_protects_interaction_terms_even_with_agreement():
    # The exact rule that kept interact_high_performer_no_promo and
    # interact_underpaid_declining out of your 7 dropped features —
    # interaction terms are protected even when 2+ methods flag them.
    drops = recommend_drops(
        vif_flagged={"interact_high_performer_no_promo"},
        l1_flagged={"interact_high_performer_no_promo"},
        corr_flagged=set(),
        interaction_terms={"interact_high_performer_no_promo"},
    )
    assert drops == set()


def test_recommend_drops_matches_documented_collinearity_drops():
    # 5 of your 7 documented drops came from this exact VIF/L1/corr
    # logic (the other 2 — dept_HR, circle_Gujarat — are dropped as
    # one-hot reference categories elsewhere, not by this function).
    # If VIF and L1 both flag these 5, the rule should reproduce the
    # real decision you made.
    documented_collinearity_drops = {
        "recency_promotion", "recency_hike", "career_band_normalized",
        "comp_underpaid_flag", "onboard_days_to_project",
    }
    drops = recommend_drops(
        vif_flagged=documented_collinearity_drops,
        l1_flagged=documented_collinearity_drops,
        corr_flagged=set(),
        interaction_terms=set(),
    )
    assert drops == documented_collinearity_drops
