"""
Unit tests for data/synthetic/generate.py.

NOT covered: main() — needs a real config.yaml and writes to disk,
same reasoning as everywhere else in this suite. Covered:
generate_employees, the pure function that does all the actual work.
Tests focus on the properties that matter for the rest of the
pipeline to function correctly downstream — schema validity, no
nulls, recency features never exceeding tenure, the exact attrition
rate being hit, and reproducibility given a fixed seed — rather than
exact values, since this is a random generator by design.
"""
import pandas as pd

from data.synthetic.generate import BANDS, CIRCLES, DEPARTMENTS, generate_employees


def test_generate_employees_returns_requested_row_count():
    df = generate_employees(n=200, attrition_rate=0.18, seed=1)
    assert len(df) == 200


def test_generate_employees_hits_exact_target_attrition_rate():
    df = generate_employees(n=1000, attrition_rate=0.2, seed=1)
    assert df["attrition_flag"].sum() == 200  # exactly 20% of 1000


def test_generate_employees_has_no_nulls():
    df = generate_employees(n=200, attrition_rate=0.18, seed=1)
    assert df.isnull().sum().sum() == 0


def test_generate_employees_band_values_are_in_allowed_set():
    df = generate_employees(n=500, attrition_rate=0.18, seed=1)
    assert set(df["band"].unique()).issubset(set(BANDS))


def test_generate_employees_department_and_circle_match_expected_lists():
    df = generate_employees(n=500, attrition_rate=0.18, seed=1)
    assert set(df["department"].unique()).issubset(set(DEPARTMENTS))
    assert set(df["circle"].unique()).issubset(set(CIRCLES))


def test_generate_employees_includes_reference_categories():
    # dept_HR and circle_Gujarat are the reference categories
    # feature_engineering.py drops — they need to actually exist
    df = generate_employees(n=500, attrition_rate=0.18, seed=1)
    assert "HR" in df["department"].values
    assert "Gujarat" in df["circle"].values


def test_generate_employees_recency_features_never_exceed_tenure():
    df = generate_employees(n=500, attrition_rate=0.18, seed=1)
    assert (df["months_since_promotion"] <= df["tenure_months"]).all()
    assert (df["months_since_hike"] <= df["tenure_months"]).all()
    assert (df["months_since_manager_changed"] <= df["tenure_months"]).all()


def test_generate_employees_is_reproducible_given_same_seed():
    df1 = generate_employees(n=100, attrition_rate=0.18, seed=42)
    df2 = generate_employees(n=100, attrition_rate=0.18, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_employees_different_seeds_produce_different_data():
    df1 = generate_employees(n=100, attrition_rate=0.18, seed=1)
    df2 = generate_employees(n=100, attrition_rate=0.18, seed=2)
    assert not df1["tenure_months"].equals(df2["tenure_months"])


def test_generate_employees_high_risk_factors_correlate_with_attrition():
    # Not just hitting the target rate — the signal needs to actually
    # be there for SHAP/Cox PH to find anything downstream
    df = generate_employees(n=2000, attrition_rate=0.18, seed=1)
    high_mgr_attrition = df["manager_attrition_rate_6m"] >= df["manager_attrition_rate_6m"].quantile(0.9)
    assert df[high_mgr_attrition]["attrition_flag"].mean() > df["attrition_flag"].mean()


def test_generate_employees_employee_ids_are_unique():
    df = generate_employees(n=500, attrition_rate=0.18, seed=1)
    assert df["employee_id"].is_unique
