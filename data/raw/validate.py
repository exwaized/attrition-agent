import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================================================
# VALIDATION RULES
# ============================================================
# Each rule returns (passed: bool, message: str)
# All rules must pass for pipeline to proceed
# ============================================================

def check_file_exists() -> tuple:
    """Features.csv must exist before validation."""
    path = Path("data/synthetic/features.csv")
    if not path.exists():
        return False, "features.csv not found — run feature_engineering.py first"
    return True, f"features.csv exists ({path.stat().st_size / 1024:.1f} KB)"

def check_feature_cols_exists() -> tuple:
    """feature_cols.txt must exist."""
    path = Path("data/synthetic/feature_cols.txt")
    if not path.exists():
        return False, "feature_cols.txt not found"
    cols = path.read_text().strip().split("\n")
    return True, f"feature_cols.txt exists ({len(cols)} features)"

def check_row_count(df: pd.DataFrame) -> tuple:
    """Must have at least 500 rows for meaningful training."""
    n = len(df)
    if n < 500:
        return False, f"Only {n} rows — need at least 500 for training"
    return True, f"{n} rows — sufficient for training"

def check_all_features_present(df: pd.DataFrame,
                                feature_cols: list) -> tuple:
    """All features in feature_cols.txt must be in features.csv."""
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        return False, f"Missing features: {missing[:5]}{'...' if len(missing)>5 else ''}"
    return True, f"All {len(feature_cols)} features present"

def check_no_nulls(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Feature matrix must have zero nulls."""
    null_counts = df[feature_cols].isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        return False, f"Nulls in features: {null_cols.to_dict()}"
    return True, "Zero nulls in feature matrix"

def check_attrition_flag(df: pd.DataFrame) -> tuple:
    """attrition_flag must be binary (0/1) with no nulls."""
    if "attrition_flag" not in df.columns:
        return False, "attrition_flag column missing"
    unique_vals = df["attrition_flag"].dropna().unique()
    non_binary  = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
    if non_binary:
        return False, f"Non-binary attrition_flag: {non_binary}"
    null_count = df["attrition_flag"].isnull().sum()
    if null_count > 0:
        return False, f"attrition_flag has {null_count} nulls"
    rate = df["attrition_flag"].mean()
    return True, f"attrition_flag is binary, rate={rate:.1%}"

def check_employee_id_unique(df: pd.DataFrame) -> tuple:
    """employee_id must be unique — no duplicates."""
    if "employee_id" not in df.columns:
        return False, "employee_id column missing"
    dups = df["employee_id"].duplicated().sum()
    if dups > 0:
        return False, f"{dups} duplicate employee IDs found"
    return True, f"All {len(df)} employee IDs are unique"

def check_class_balance(df: pd.DataFrame) -> tuple:
    """
    Class imbalance must be within workable range.
    Less than 5% attrition = model will struggle badly.
    More than 60% attrition = suspicious, likely data error.
    """
    rate = df["attrition_flag"].mean()
    if rate < 0.05:
        return False, f"Attrition rate {rate:.1%} too low — model won't learn minority class"
    if rate > 0.60:
        return False, f"Attrition rate {rate:.1%} suspiciously high — check data"
    return True, f"Attrition rate {rate:.1%} is within workable range"

def check_feature_ranges(df: pd.DataFrame,
                          feature_cols: list) -> tuple:
    """
    Key features must be within expected ranges.
    Catches silent data corruption from mapper.
    """
    RANGE_CHECKS = {
        "comp_compa_ratio":     (0.2, 3.0),
        "perf_rating_current":  (1.0, 5.0),
        "tenure_months":        (0,   600),
        "career_tenure_at_band":(0,   600),
        "band":                 (1,   10),
    }

    violations = []
    for col, (low, high) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        below = (df[col] < low).sum()
        above = (df[col] > high).sum()
        if below + above > len(df) * 0.05:  # more than 5% out of range
            violations.append(
                f"{col}: {below+above} values outside [{low},{high}]"
            )

    if violations:
        return False, f"Range violations: {violations}"
    return True, "All key features within expected ranges"

def check_feature_variance(df: pd.DataFrame,
                            feature_cols: list) -> tuple:
    """
    Features with zero variance are useless and may cause issues.
    Binary features (interaction terms) can legitimately be near-zero.
    """
    numeric_features = [
        f for f in feature_cols
        if f in df.columns
        and not f.startswith("dept_")
        and not f.startswith("circle_")
        and not f.startswith("interact_")
    ]

    zero_var = []
    for col in numeric_features:
        if df[col].std() == 0:
            zero_var.append(col)

    if zero_var:
        return False, f"Zero variance features: {zero_var}"
    return True, f"All {len(numeric_features)} numeric features have variance"

def check_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Temporal split must produce balanced class distribution.
    Both train and test must have at least 10% attrition.
    """
    split_idx   = int(len(df) * 0.80)
    train_rate  = df.iloc[:split_idx]["attrition_flag"].mean()
    test_rate   = df.iloc[split_idx:]["attrition_flag"].mean()
    drift       = abs(train_rate - test_rate)

    if train_rate < 0.05 or test_rate < 0.05:
        return False, (
            f"Temporal split produces near-empty class: "
            f"train={train_rate:.1%}, test={test_rate:.1%}"
        )
    if drift > 0.15:
        return False, (
            f"Large attrition drift between train/test: "
            f"train={train_rate:.1%}, test={test_rate:.1%} "
            f"(drift={drift:.1%}) — temporal ordering may be wrong"
        )
    return True, (
        f"Train attrition={train_rate:.1%}, "
        f"Test attrition={test_rate:.1%}, "
        f"Drift={drift:.1%}"
    )

def check_interaction_terms(df: pd.DataFrame) -> tuple:
    """
    Interaction terms must fire for at least 1% of employees.
    Zero-firing interactions add noise without signal.
    """
    interact_cols = [c for c in df.columns if c.startswith("interact_")]
    low_firing    = []

    for col in interact_cols:
        if col not in df.columns:
            continue
        fire_rate = df[col].mean()
        if fire_rate < 0.002:
            low_firing.append(f"{col}: {fire_rate:.1%}")

    if low_firing:
        return False, f"Near-zero interaction terms: {low_firing}"
    return True, f"All {len(interact_cols)} interaction terms firing adequately"

def check_dummy_variables(df: pd.DataFrame) -> tuple:
    """
    Dummy variables must not sum to 1 for any row.
    Reference category must be dropped.
    """
    dept_cols   = [c for c in df.columns if c.startswith("dept_")]
    circle_cols = [c for c in df.columns if c.startswith("circle_")]

    issues = []

    if dept_cols:
        dept_sum = df[dept_cols].sum(axis=1)
        if (dept_sum == 1).all():
            issues.append(
                "dept_ dummies sum to 1 — reference category not dropped"
            )

    if circle_cols:
        circle_sum = df[circle_cols].sum(axis=1)
        if (circle_sum == 1).all():
            issues.append(
                "circle_ dummies sum to 1 — reference category not dropped"
            )

    if issues:
        return False, f"Dummy variable issues: {issues}"
    return True, "Dummy reference categories correctly dropped"

# ============================================================
# RUN ALL VALIDATIONS
# ============================================================
def run_validation() -> dict:
    """
    Runs all validation checks.
    Returns results dict with pass/fail per check.
    Prints summary and saves to logs/validation_report.json.
    """
    print(f"\n{'='*50}")
    print(f"FEATURE VALIDATION — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")

    results  = {}
    passed   = 0
    failed   = 0
    blockers = []

    # Load data if files exist
    df           = None
    feature_cols = []

    # Run file existence checks first
    for name, check_fn in [
        ("file_exists",         check_file_exists),
        ("feature_cols_exists", check_feature_cols_exists),
    ]:
        ok, msg = check_fn()
        results[name] = {"passed": ok, "message": msg}
        status = "[OK]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")
        if ok:
            passed += 1
        else:
            failed += 1
            blockers.append(msg)

    # If files exist, load and run remaining checks
    if results["file_exists"]["passed"] and \
       results["feature_cols_exists"]["passed"]:

        df = pd.read_csv("data/synthetic/features.csv")
        feature_cols = Path(
            "data/synthetic/feature_cols.txt"
        ).read_text().strip().split("\n")

        remaining_checks = [
            ("row_count",           lambda: check_row_count(df)),
            ("features_present",    lambda: check_all_features_present(df, feature_cols)),
            ("no_nulls",            lambda: check_no_nulls(df, feature_cols)),
            ("attrition_flag",      lambda: check_attrition_flag(df)),
            ("employee_id_unique",  lambda: check_employee_id_unique(df)),
            ("class_balance",       lambda: check_class_balance(df)),
            ("feature_ranges",      lambda: check_feature_ranges(df, feature_cols)),
            ("feature_variance",    lambda: check_feature_variance(df, feature_cols)),
            ("train_test_split",    lambda: check_train_test_split(df)),
            ("interaction_terms",   lambda: check_interaction_terms(df)),
            ("dummy_variables",     lambda: check_dummy_variables(df)),
        ]

        for name, check_fn in remaining_checks:
            try:
                ok, msg = check_fn()
            except Exception as e:
                ok  = False
                msg = f"Check crashed: {str(e)}"

            results[name] = {"passed": ok, "message": msg}
            status = "[OK]" if ok else "[FAIL]"
            print(f"{status} {name}: {msg}")

            if ok:
                passed += 1
            else:
                failed += 1
                blockers.append(f"{name}: {msg}")

    # Summary
    total   = passed + failed
    verdict = "[OK] PASSED — ready for train.py" if failed == 0 \
              else f"[FAIL] FAILED — {failed} checks failed"

    print(f"\n{'='*50}")
    print(f"RESULT: {verdict}")
    print(f"Passed: {passed}/{total}")

    if blockers:
        print("\nFailed checks:")
        for b in blockers:
            print(f"  [FAIL] {b}")

    # Save report
    report = {
        "timestamp":  datetime.now().isoformat(),
        "verdict":    verdict,
        "passed":     passed,
        "failed":     failed,
        "total":      total,
        "can_proceed":failed == 0,
        "checks":     results
    }

    Path("logs").mkdir(exist_ok=True)
    with open("logs/validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("Report saved: logs/validation_report.json")

    return report


if __name__ == "__main__":
    report = run_validation()
    # Exit with error code if validation failed
    # Allows run_pipeline.py to catch failure
    import sys
    sys.exit(0 if report["can_proceed"] else 1)