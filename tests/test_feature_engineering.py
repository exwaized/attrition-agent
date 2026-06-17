"""
Unit tests for data/synthetic/feature_engineering.py.

The bulk of this file is sequential feature assignment on one
dataframe (overload proxies, trend/volatility/career features, 5 of
the 7 interaction terms) — wrapped in main() but not otherwise
extracted, since re-deriving each assignment in a test would mostly
just restate the same line of code. Covered here: compute_peer_attrition
(already its own function) and the two interaction terms pulled out
as standalone functions, with boundary-value tests on the thresholds
that matter (4.0 perf rating, 18-month promo gap, 0.85 compa ratio).
"""
import pandas as pd

from data.synthetic.feature_engineering import (
    compute_interact_high_performer_no_promo,
    compute_interact_underpaid_declining,
    compute_peer_attrition,
)

# ---------- compute_peer_attrition ----------

def test_compute_peer_attrition_counts_same_dept_and_circle_peers():
    df = pd.DataFrame({
        "department":     ["Sales", "Sales", "Sales", "IT"],
        "circle":         ["North", "North", "North", "North"],
        "attrition_flag": [0,        1,       1,       1],
    })
    result = compute_peer_attrition(df)
    # Employee 0 (Sales/North): peers are rows 1,2 (both Sales/North, excluding self) -> 2 left
    assert result[0] == 2
    # Employee 3 (IT/North): no other IT/North peers -> 0
    assert result[3] == 0


def test_compute_peer_attrition_excludes_self():
    df = pd.DataFrame({
        "department":     ["Sales"],
        "circle":         ["North"],
        "attrition_flag": [1],  # this employee left, but shouldn't count themself
    })
    result = compute_peer_attrition(df)
    assert result[0] == 0


# ---------- compute_interact_high_performer_no_promo ----------

def test_high_performer_no_promo_fires_when_both_conditions_met():
    df = pd.DataFrame({
        "perf_rating_current":    [4.5],
        "months_since_promotion": [20],
    })
    assert compute_interact_high_performer_no_promo(df).tolist() == [1]


def test_high_performer_no_promo_boundary_perf_rating_exactly_4():
    # >= 4.0, so exactly 4.0 should count
    df = pd.DataFrame({
        "perf_rating_current":    [4.0],
        "months_since_promotion": [19],
    })
    assert compute_interact_high_performer_no_promo(df).tolist() == [1]


def test_high_performer_no_promo_boundary_promo_gap_exactly_18_does_not_fire():
    # > 18, so exactly 18 should NOT count
    df = pd.DataFrame({
        "perf_rating_current":    [4.5],
        "months_since_promotion": [18],
    })
    assert compute_interact_high_performer_no_promo(df).tolist() == [0]


def test_high_performer_no_promo_does_not_fire_for_low_performer():
    df = pd.DataFrame({
        "perf_rating_current":    [3.0],
        "months_since_promotion": [30],
    })
    assert compute_interact_high_performer_no_promo(df).tolist() == [0]


# ---------- compute_interact_underpaid_declining ----------

def test_underpaid_declining_fires_when_both_conditions_met():
    df = pd.DataFrame({
        "compa_ratio":           [0.80],
        "login_freq_slope_30d":  [-0.5],
    })
    assert compute_interact_underpaid_declining(df).tolist() == [1]


def test_underpaid_declining_boundary_compa_ratio_exactly_085_does_not_fire():
    # < 0.85, so exactly 0.85 should NOT count
    df = pd.DataFrame({
        "compa_ratio":           [0.85],
        "login_freq_slope_30d":  [-0.5],
    })
    assert compute_interact_underpaid_declining(df).tolist() == [0]


def test_underpaid_declining_does_not_fire_if_login_trend_flat_or_rising():
    df = pd.DataFrame({
        "compa_ratio":           [0.70],
        "login_freq_slope_30d":  [0.0],
    })
    assert compute_interact_underpaid_declining(df).tolist() == [0]
