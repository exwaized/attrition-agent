"""
Unit tests for the pure logic in ev_scoring.py: compute_ev,
assign_risk_tier, get_top_shap_drivers.

NOT covered here: model.predict_proba(), cph.predict_median(), and
explainer.shap_values() inside main() — those need a real fitted
model/cph/explainer loaded from pickle and stay covered by actually
running the pipeline end-to-end.
"""
import json

import pandas as pd
import pytest

from models.ev_scoring import compute_ev, assign_risk_tier, get_top_shap_drivers


# ---------- compute_ev ----------

def test_compute_ev_scalar():
    ev = compute_ev(0.5, replacement_cost=450000, intervention_cost=50000)
    assert ev == 0.5 * 450000 - 50000


def test_compute_ev_zero_probability_is_negative():
    # P(attrition)=0 should still cost the intervention itself —
    # negative EV correctly signals "don't spend the budget."
    ev = compute_ev(0.0, replacement_cost=450000, intervention_cost=50000)
    assert ev == -50000


def test_compute_ev_vectorized_on_series():
    p = pd.Series([0.0, 0.5, 1.0])
    ev = compute_ev(p, replacement_cost=100000, intervention_cost=10000)
    assert list(ev) == [-10000, 40000, 90000]


# ---------- assign_risk_tier ----------

# Your real config.yaml ev_framework thresholds — tests are grounded
# in the actual business numbers, not arbitrary ones.
HIGH = 200000
MEDIUM = 50000


@pytest.mark.parametrize("p,ev,expected", [
    (0.70, 200000, "CRITICAL"),  # exact CRITICAL boundary
    (0.69, 200000, "HIGH"),      # just under CRITICAL's p threshold
    (0.55, 100000, "HIGH"),      # exact HIGH boundary (half of 200000)
    (0.54, 100000, "MEDIUM"),    # just under HIGH's p threshold
    (0.40, 50000, "MEDIUM"),     # exact MEDIUM boundary
    (0.39, 50000, "LOW"),        # just under MEDIUM's p threshold
])
def test_assign_risk_tier_boundaries(p, ev, expected):
    row = {"p_attrition": p, "ev": ev}
    assert assign_risk_tier(row, HIGH, MEDIUM) == expected


def test_assign_risk_tier_high_p_low_ev_is_still_low():
    # The whole point of EV-gating: a near-certain leaver isn't worth
    # alerting on if the EV math says the intervention isn't worth it.
    row = {"p_attrition": 0.99, "ev": 100}
    assert assign_risk_tier(row, HIGH, MEDIUM) == "LOW"


def test_assign_risk_tier_works_with_pandas_series_row():
    # df.apply(..., axis=1) passes a pandas Series, not a dict — confirm
    # both access patterns work since row["p_attrition"] / row["ev"]
    # behave the same way on either.
    row = pd.Series({"p_attrition": 0.70, "ev": 200000})
    assert assign_risk_tier(row, HIGH, MEDIUM) == "CRITICAL"


# ---------- get_top_shap_drivers ----------

def test_get_top_shap_drivers_orders_by_absolute_value():
    shap_row = [0.5, -0.8, 0.1, -0.05]
    features = ["a", "b", "c", "d"]
    top = get_top_shap_drivers(shap_row, features, n=3)
    assert [d["feature"] for d in top] == ["b", "a", "c"]


def test_get_top_shap_drivers_preserves_sign():
    shap_row = [0.5, -0.8]
    top = get_top_shap_drivers(shap_row, ["a", "b"], n=2)
    by_feature = {d["feature"]: d["shap"] for d in top}
    assert by_feature["a"] == 0.5
    assert by_feature["b"] == -0.8  # not flattened to abs — sign carries meaning downstream


def test_get_top_shap_drivers_respects_n():
    shap_row = [0.1, 0.2, 0.3, 0.4, 0.5]
    top = get_top_shap_drivers(shap_row, ["a", "b", "c", "d", "e"], n=2)
    assert len(top) == 2


def test_get_top_shap_drivers_rounds_to_four_decimals():
    shap_row = [0.123456789]
    top = get_top_shap_drivers(shap_row, ["a"], n=1)
    assert top[0]["shap"] == 0.1235


def test_get_top_shap_drivers_output_is_json_serializable():
    # This list gets json.dumps()'d directly in ev_scoring.py's main()
    # — confirm that doesn't break.
    shap_row = [0.5, -0.3]
    top = get_top_shap_drivers(shap_row, ["a", "b"], n=2)
    json.dumps(top)
