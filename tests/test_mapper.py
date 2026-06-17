"""
Unit tests for data/raw/mapper.py.

mapper.py was already well-structured — 9 isolated step functions
plus an existing __main__ guard, same shape as validate.py and
eda.py. Covered here: the 5 highest-value pure functions
(clean_column_names, map_columns_fuzzy, clean_numeric, cap_outliers,
validate_final). NOT covered: load_raw_file (file I/O / encoding
detection is an integration concern), handle_nulls and
compute_date_derived_features (largely repeat the same
fillna/business-rule pattern already exercised elsewhere),
validate_peer_groups (thin groupby wrapper, low risk of regression).
"""
import numpy as np
import pandas as pd

from data.raw.mapper import (
    cap_outliers,
    clean_column_names,
    clean_numeric,
    coerce_types,
    map_columns_fuzzy,
    validate_final,
)

# ---------- clean_column_names ----------

def test_clean_column_names_strips_bom():
    df = pd.DataFrame({"\ufeffemployee_id": [1]})
    cleaned, _ = clean_column_names(df)
    assert "employee_id" in cleaned.columns


def test_clean_column_names_lowercases_and_strips_whitespace():
    df = pd.DataFrame({" Employee ID ": [1]})
    cleaned, _ = clean_column_names(df)
    assert "employee_id" in cleaned.columns


def test_clean_column_names_removes_special_characters():
    df = pd.DataFrame({"Tenure (Months)!": [1]})
    cleaned, _ = clean_column_names(df)
    assert "tenure_months" in cleaned.columns


def test_clean_column_names_collapses_repeated_underscores():
    df = pd.DataFrame({"emp--id": [1]})
    cleaned, _ = clean_column_names(df)
    assert "emp_id" in cleaned.columns


def test_clean_column_names_audit_only_logs_when_something_changed():
    df = pd.DataFrame({"employee_id": [1]})  # already clean
    _, audit = clean_column_names(df)
    assert audit == []


# ---------- map_columns_fuzzy ----------

def test_map_columns_fuzzy_exact_match():
    df = pd.DataFrame({"employee_id": [1], "band": [5]})
    mapped, _, mapping = map_columns_fuzzy(df)
    assert mapping["employee_id"] == "employee_id"
    assert "employee_id" in mapped.columns


def test_map_columns_fuzzy_known_alias():
    df = pd.DataFrame({"emp_id": [1]})
    mapped, _, mapping = map_columns_fuzzy(df)
    assert mapping["emp_id"] == "employee_id"
    assert "employee_id" in mapped.columns


def test_map_columns_fuzzy_resolves_close_typo():
    # Close enough to "department" to clear the 80% threshold
    df = pd.DataFrame({"departmnt": [1]})
    mapped, _, mapping = map_columns_fuzzy(df)
    assert mapping.get("departmnt") == "department"


def test_map_columns_fuzzy_drops_unrecognizable_column():
    df = pd.DataFrame({"employee_id": [1], "totally_unrelated_garbage_xyz": [1]})
    mapped, _, mapping = map_columns_fuzzy(df)
    assert "totally_unrelated_garbage_xyz" not in mapped.columns
    assert "totally_unrelated_garbage_xyz" not in mapping


# ---------- clean_numeric ----------

def test_clean_numeric_strips_currency_and_commas():
    assert clean_numeric("₹9,80,000") == 980000.0


def test_clean_numeric_strips_percentage_sign():
    assert clean_numeric("8.5%") == 8.5


def test_clean_numeric_handles_unicode_minus():
    assert clean_numeric("−5") == -5.0


def test_clean_numeric_unparseable_returns_nan():
    assert pd.isnull(clean_numeric("not_a_number"))


def test_clean_numeric_null_input_returns_nan():
    assert pd.isnull(clean_numeric(None))


def test_coerce_types_rescales_compa_ratio_stored_as_percentage():
    df = pd.DataFrame({"compa_ratio": [85, 90, 95]})  # stored as 85 not 0.85
    result, audit = coerce_types(df)
    assert result["compa_ratio"].max() < 2.0
    assert any("percentage" in a for a in audit)


def test_coerce_types_converts_attrition_flag_text_to_binary():
    df = pd.DataFrame({"attrition_flag": ["Resigned", "Active"]})
    result, _ = coerce_types(df)
    assert result["attrition_flag"].tolist() == [1, 0]


# ---------- cap_outliers ----------

def test_cap_outliers_clips_extreme_values():
    # One wild value way past the 99th percentile of the rest
    df = pd.DataFrame({"tenure_months": [10, 11, 12, 13, 14] * 20 + [5000]})
    result, audit = cap_outliers(df)
    assert result["tenure_months"].max() < 5000
    assert any("tenure_months" in a for a in audit)


def test_cap_outliers_leaves_clean_column_untouched():
    # All-identical values: p01 == p99 == the value itself, so nothing
    # falls outside [p01, p99] and no clipping happens. (A naive evenly-
    # spaced range like 10..29 actually DOES get clipped at the edges —
    # pandas' default linear-interpolation quantile estimates p01 above
    # the true min for small N, which is a real subtlety worth knowing,
    # not a bug in cap_outliers.)
    df = pd.DataFrame({"tenure_months": [15] * 50})
    result, audit = cap_outliers(df)
    assert result["tenure_months"].tolist() == [15] * 50
    assert audit == []


# ---------- validate_final ----------

def test_validate_final_flags_missing_required_column():
    df = pd.DataFrame({"employee_id": ["E1"]})  # missing band, tenure_months, etc.
    _, _, issues = validate_final(df)
    assert any("band" in i for i in issues)


def test_validate_final_flags_nulls_in_non_nullable_column():
    df = pd.DataFrame({
        "employee_id": ["E1", "E2"],
        "band":        [5, np.nan],  # band is required + non-nullable
        "tenure_months": [10, 20],
        "months_since_promotion": [1, 2],
        "months_since_hike": [1, 2],
        "compa_ratio": [0.9, 0.9],
        "perf_rating_current": [3, 3],
        "login_freq_30d": [10, 10],
        "attrition_flag": [0, 1],
    })
    _, _, issues = validate_final(df)
    assert any("band" in i and "nulls" in i for i in issues)


def test_validate_final_clean_data_has_no_issues():
    df = pd.DataFrame({
        "employee_id": ["E1", "E2"],
        "band":        [5, 6],
        "circle":      ["North", "South"],
        "department":  ["Sales", "IT"],
        "tenure_months": [10, 20],
        "months_since_promotion": [1, 2],
        "months_since_hike": [1, 2],
        "compa_ratio": [0.9, 0.9],
        "perf_rating_current": [3, 3],
        "login_freq_30d": [10, 10],
        "attrition_flag": [0, 1],
    })
    _, _, issues = validate_final(df)
    assert issues == []
