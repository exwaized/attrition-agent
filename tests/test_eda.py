"""
Unit tests for data/raw/eda.py.

eda.py needed no structural refactor — like validate.py, it's
already pure check functions plus an existing __main__ guard. Three
of the seven checks (check_data_quality, check_outliers,
check_pipeline_readiness) take a df/dicts and do zero I/O, so
they're tested directly. check_attrition_rate, check_distributions,
check_subgroups, and check_temporal_patterns all save matplotlib
plots as a side effect — not covered here, since exercising them
properly means actually running the pipeline against real data, same
reasoning as the model-fitting code elsewhere in this suite.
"""
import pandas as pd

from data.raw.eda import check_data_quality, check_outliers, check_pipeline_readiness

# ---------- check_data_quality ----------

def test_check_data_quality_reports_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = check_data_quality(df)
    assert result["shape"] == {"rows": 3, "cols": 2}


def test_check_data_quality_flags_missing_required_columns():
    df = pd.DataFrame({"employee_id": [1, 2]})  # missing band, department, etc.
    result = check_data_quality(df)
    assert "band" in result["missing_required"]
    assert "tenure_months" in result["missing_required"]


def test_check_data_quality_reports_null_counts():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    result = check_data_quality(df)
    assert result["nulls"]["a"]["count"] == 1
    assert "b" not in result["nulls"]


def test_check_data_quality_detects_duplicate_employee_ids():
    df = pd.DataFrame({"employee_id": [1, 2, 2, 3]})
    result = check_data_quality(df)
    assert result["duplicate_employee_ids"] == 1


def test_check_data_quality_flags_non_binary_attrition():
    df = pd.DataFrame({"attrition_flag": [0, 1, 2]})
    result = check_data_quality(df)
    assert 2 in result["attrition_flag_non_binary"]


# ---------- check_outliers ----------

def test_check_outliers_detects_extreme_high_values():
    # IQR-based: most values clustered, one wild outlier far past 3x IQR
    df = pd.DataFrame({"tenure_months": [10, 11, 12, 13, 14, 500]})
    result = check_outliers(df)
    assert "tenure_months" in result
    assert result["tenure_months"]["extreme_high"] >= 1


def test_check_outliers_clean_data_reports_nothing():
    df = pd.DataFrame({"tenure_months": [10, 11, 12, 13, 14, 15]})
    result = check_outliers(df)
    assert "tenure_months" not in result


# ---------- check_pipeline_readiness ----------

def test_check_pipeline_readiness_blocked_on_missing_columns():
    quality = {"missing_required": ["band", "department"]}
    result = check_pipeline_readiness(pd.DataFrame(), quality, {})
    assert result["can_proceed"] is False
    assert "[FAIL] BLOCKED" in result["verdict"]
    assert any("band" in b for b in result["blockers"])


def test_check_pipeline_readiness_blocked_on_duplicate_ids():
    quality = {"missing_required": [], "duplicate_employee_ids": 5}
    result = check_pipeline_readiness(pd.DataFrame(), quality, {})
    assert result["can_proceed"] is False
    assert any("5 duplicate" in b for b in result["blockers"])


def test_check_pipeline_readiness_blocked_on_high_null_rate():
    quality = {
        "missing_required": [],
        "nulls": {"tenure_months": {"pct": 35.0}}
    }
    result = check_pipeline_readiness(pd.DataFrame(), quality, {})
    assert result["can_proceed"] is False
    assert any("tenure_months" in b for b in result["blockers"])


def test_check_pipeline_readiness_warns_but_proceeds_on_attrition_deviation():
    quality = {"missing_required": [], "deviation": 0.12}
    result = check_pipeline_readiness(pd.DataFrame(), quality, {})
    assert result["can_proceed"] is True  # deviation is a warning, not a blocker
    assert any("deviates" in w for w in result["warnings"])


def test_check_pipeline_readiness_clean_data_is_ready():
    quality = {"missing_required": [], "duplicate_employee_ids": 0, "deviation": 0.01}
    result = check_pipeline_readiness(pd.DataFrame(), quality, {})
    assert result["can_proceed"] is True
    assert result["blockers"] == []
    assert "[OK] READY" in result["verdict"]
