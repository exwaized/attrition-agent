"""
Unit tests for validate.py's check functions.

validate.py already had good structure for testing — every check is
its own pure-ish function, and the file is already guarded by
`if __name__ == "__main__":`. The one fix that was needed: a
module-level `cfg = yaml.safe_load(...)` at the top that was never
actually used anywhere — removed, since it would've made importing
this module fail without a config.yaml in the working directory.

check_file_exists and check_feature_cols_exists hit the filesystem
with hardcoded relative paths, so they're tested via
monkeypatch.chdir into a temp directory. The other 11 checks take
df/feature_cols as explicit arguments and are tested directly with
small synthetic DataFrames.
"""
import pandas as pd

from data.raw.validate import (
    check_all_features_present,
    check_attrition_flag,
    check_class_balance,
    check_dummy_variables,
    check_employee_id_unique,
    check_feature_cols_exists,
    check_feature_ranges,
    check_feature_variance,
    check_file_exists,
    check_interaction_terms,
    check_no_nulls,
    check_row_count,
    check_train_test_split,
)

# ---------- filesystem checks ----------

def test_check_file_exists_true_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    (tmp_path / "data" / "synthetic" / "features.csv").write_text("a,b\n1,2\n")

    ok, msg = check_file_exists()
    assert ok is True
    assert "features.csv exists" in msg


def test_check_file_exists_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, msg = check_file_exists()
    assert ok is False
    assert "not found" in msg


def test_check_feature_cols_exists_reports_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    (tmp_path / "data" / "synthetic" / "feature_cols.txt").write_text("a\nb\nc\n")

    ok, msg = check_feature_cols_exists()
    assert ok is True
    assert "3 features" in msg


# ---------- row count / presence / nulls ----------

def test_check_row_count_below_minimum():
    df = pd.DataFrame({"x": range(100)})
    ok, msg = check_row_count(df)
    assert ok is False
    assert "100" in msg


def test_check_row_count_meets_minimum():
    df = pd.DataFrame({"x": range(500)})
    ok, _ = check_row_count(df)
    assert ok is True


def test_check_all_features_present_detects_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    ok, msg = check_all_features_present(df, ["a", "b", "c"])
    assert ok is False
    assert "c" in msg


def test_check_all_features_present_all_there():
    df = pd.DataFrame({"a": [1], "b": [2]})
    ok, _ = check_all_features_present(df, ["a", "b"])
    assert ok is True


def test_check_no_nulls_detects_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    ok, msg = check_no_nulls(df, ["a", "b"])
    assert ok is False
    assert "a" in msg


def test_check_no_nulls_clean():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    ok, _ = check_no_nulls(df, ["a", "b"])
    assert ok is True


# ---------- attrition_flag ----------

def test_check_attrition_flag_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    ok, msg = check_attrition_flag(df)
    assert ok is False
    assert "missing" in msg


def test_check_attrition_flag_non_binary():
    df = pd.DataFrame({"attrition_flag": [0, 1, 2]})
    ok, msg = check_attrition_flag(df)
    assert ok is False
    assert "Non-binary" in msg


def test_check_attrition_flag_nulls():
    df = pd.DataFrame({"attrition_flag": [0, 1, None]})
    ok, msg = check_attrition_flag(df)
    assert ok is False
    assert "nulls" in msg


def test_check_attrition_flag_valid():
    df = pd.DataFrame({"attrition_flag": [0, 0, 0, 1]})
    ok, msg = check_attrition_flag(df)
    assert ok is True
    assert "25.0%" in msg


# ---------- employee_id ----------

def test_check_employee_id_unique_detects_duplicates():
    df = pd.DataFrame({"employee_id": [1, 2, 2, 3]})
    ok, msg = check_employee_id_unique(df)
    assert ok is False
    assert "1 duplicate" in msg


def test_check_employee_id_unique_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    ok, _ = check_employee_id_unique(df)
    assert ok is False


def test_check_employee_id_unique_all_unique():
    df = pd.DataFrame({"employee_id": [1, 2, 3]})
    ok, _ = check_employee_id_unique(df)
    assert ok is True


# ---------- class balance ----------

def test_check_class_balance_too_low():
    df = pd.DataFrame({"attrition_flag": [0] * 96 + [1] * 4})  # 4%
    ok, msg = check_class_balance(df)
    assert ok is False
    assert "too low" in msg


def test_check_class_balance_too_high():
    df = pd.DataFrame({"attrition_flag": [0] * 35 + [1] * 65})  # 65%
    ok, msg = check_class_balance(df)
    assert ok is False
    assert "suspiciously high" in msg


def test_check_class_balance_lower_boundary_is_acceptable():
    # rate == 0.05 exactly should pass — the check is strictly "< 0.05"
    df = pd.DataFrame({"attrition_flag": [0] * 95 + [1] * 5})  # exactly 5%
    ok, _ = check_class_balance(df)
    assert ok is True


def test_check_class_balance_within_range():
    df = pd.DataFrame({"attrition_flag": [0] * 80 + [1] * 20})  # 20%
    ok, _ = check_class_balance(df)
    assert ok is True


# ---------- feature ranges ----------

def test_check_feature_ranges_flags_excessive_violations():
    df = pd.DataFrame({"band": [5] * 90 + [99] * 10})  # 10% out of [1,10]
    ok, msg = check_feature_ranges(df, [])
    assert ok is False
    assert "band" in msg


def test_check_feature_ranges_tolerates_small_violations():
    df = pd.DataFrame({"band": [5] * 98 + [99] * 2})  # 2% — under the 5% tolerance
    ok, _ = check_feature_ranges(df, [])
    assert ok is True


# ---------- feature variance ----------

def test_check_feature_variance_detects_zero_variance():
    df = pd.DataFrame({"constant_feature": [5] * 10, "varying_feature": range(10)})
    ok, msg = check_feature_variance(df, ["constant_feature", "varying_feature"])
    assert ok is False
    assert "constant_feature" in msg


def test_check_feature_variance_ignores_dummy_and_interaction_columns():
    # A genuinely zero-variance interact_ column would normally fail
    # this check — but dept_/circle_/interact_ prefixed columns are
    # excluded entirely, since a legitimately rare interaction term
    # being constant in a small sample shouldn't fail validation.
    df = pd.DataFrame({
        "interact_never_fires": [0] * 1000,  # std == 0 exactly
        "real_feature": range(1000),
    })
    ok, _ = check_feature_variance(df, ["interact_never_fires", "real_feature"])
    assert ok is True


# ---------- train/test split drift ----------

def test_check_train_test_split_detects_drift():
    # Both rates clear the 5% floor, but differ by more than 15%
    train_flags = [1] * 24 + [0] * 56   # 80 rows, 30% attrition
    test_flags = [1] * 2 + [0] * 18     # 20 rows, 10% attrition
    df = pd.DataFrame({"attrition_flag": train_flags + test_flags})
    ok, msg = check_train_test_split(df)
    assert ok is False
    assert "drift" in msg.lower()


def test_check_train_test_split_near_empty_class_flagged_first():
    # test set has 0% attrition — below the 5% floor. This should hit
    # the near-empty-class branch BEFORE the drift branch, even though
    # the rates also differ by a lot.
    df = pd.DataFrame({"attrition_flag": [1] * 16 + [0] * 64 + [0] * 20})
    ok, msg = check_train_test_split(df)
    assert ok is False
    assert "near-empty" in msg.lower()


def test_check_train_test_split_balanced_passes():
    df = pd.DataFrame({"attrition_flag": [0, 0, 0, 0, 1] * 20})  # 20% throughout
    ok, _ = check_train_test_split(df)
    assert ok is True


# ---------- interaction terms ----------

def test_check_interaction_terms_detects_low_firing():
    df = pd.DataFrame({"interact_rare": [0] * 999 + [1]})  # fires 0.1%
    ok, msg = check_interaction_terms(df)
    assert ok is False
    assert "interact_rare" in msg


def test_check_interaction_terms_adequate_firing():
    df = pd.DataFrame({"interact_common": [0] * 950 + [1] * 50})  # fires 5%
    ok, _ = check_interaction_terms(df)
    assert ok is True


# ---------- dummy variables ----------

def test_check_dummy_variables_flags_undropped_reference():
    # Every row sums to exactly 1 — no reference category was dropped
    df = pd.DataFrame({
        "dept_HR": [1, 0, 0],
        "dept_Sales": [0, 1, 0],
        "dept_IT": [0, 0, 1],
    })
    ok, msg = check_dummy_variables(df)
    assert ok is False
    assert "dept_" in msg


def test_check_dummy_variables_correctly_dropped_reference():
    # Reference category's rows have ALL dummies at 0, so the sum
    # isn't uniformly 1 — matches "Gujarat dropped as reference" / "HR
    # dropped as reference" in your actual feature set.
    df = pd.DataFrame({
        "dept_Sales": [0, 1, 0],
        "dept_IT": [0, 0, 1],
    })
    ok, _ = check_dummy_variables(df)
    assert ok is True