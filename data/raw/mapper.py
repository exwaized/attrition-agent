# ============================================================
# mapper.py — Robust Data Ingestion & Column Mapping Layer
# ============================================================
# PURPOSE: Handles real HRIS data in any format/structure
# HANDLES:
#   - Fuzzy column name matching (emp_id → employee_id)
#   - Uneven spacing, mixed case, special characters
#   - Multiple null representations (NA, N/A, -, ?, blank)
#   - Mixed date formats (DD/MM/YYYY, MM-DD-YY etc)
#   - Numeric columns stored as strings ("8.5%", "₹9,80,000")
#   - Outlier capping at 1st/99th percentile
#   - Minimum group size validation for peer attrition
#   - Full audit trail of every transformation applied
# FLOW: raw HR file → cleaned → employees.csv (pipeline input)
# ============================================================

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore")

# --- Step 1: Load config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ============================================================
# CANONICAL COLUMN SCHEMA
# ============================================================
# These are the exact column names the pipeline expects
# mapper.py's job = get ANY real HR file to match these
# ============================================================
CANONICAL_COLUMNS = {
    # Identity
    "employee_id":                  {"type": "str",   "required": True,  "nullable": False},
    "band":                         {"type": "int",   "required": True,  "nullable": False},
    "circle":                       {"type": "str",   "required": True,  "nullable": True},
    "department":                   {"type": "str",   "required": True,  "nullable": True},
    "gender":                       {"type": "str",   "required": False, "nullable": True},

    # Tenure
    "tenure_months":                {"type": "float", "required": True,  "nullable": False},
    "months_since_promotion":       {"type": "float", "required": True,  "nullable": True},
    "months_since_hike":            {"type": "float", "required": True,  "nullable": True},
    "months_since_manager_changed": {"type": "float", "required": False, "nullable": True},

    # Compensation
    "compa_ratio":                  {"type": "float", "required": True,  "nullable": True},
    "hike_vs_band_median":          {"type": "float", "required": False, "nullable": True},

    # Performance
    "perf_rating_current":          {"type": "float", "required": True,  "nullable": True},

    # Behavioral
    "login_freq_30d":               {"type": "float", "required": True,  "nullable": True},
    "login_freq_slope_30d":         {"type": "float", "required": False, "nullable": True},
    "training_completions_90d":     {"type": "float", "required": False, "nullable": True},
    "leave_days_30d":               {"type": "float", "required": False, "nullable": True},
    "performance_rating_delta":     {"type": "float", "required": False, "nullable": True},

    # Org context
    "peer_attrition_count_90d":     {"type": "float", "required": False, "nullable": True},
    "manager_attrition_rate_6m":    {"type": "float", "required": False, "nullable": True},
    "team_size_change_pct":         {"type": "float", "required": False, "nullable": True},

    # Label
    "attrition_flag":               {"type": "int",   "required": True,  "nullable": False},
}

# ============================================================
# FUZZY ALIAS MAP
# ============================================================
# Common real-world column name variations
# rapidfuzz handles anything not explicitly listed here
# ============================================================
KNOWN_ALIASES = {
    "employee_id":                  ["emp_id", "employee id", "empid", "eid",
                                     "staff_id", "staffid", "id", "emp_code",
                                     "employee_code", "hrid", "hr_id"],
    "band":                         ["grade", "level", "job_level", "pay_band",
                                     "band_level", "joblevel", "salary_band",
                                     "career_level", "gradelevel"],
    "circle":                       ["region", "location", "geo", "geography",
                                     "state", "zone", "business_unit", "bu",
                                     "circle_name"],
    "department":                   ["dept", "function", "team", "division",
                                     "business_function", "dept_name", "deptname"],
    "gender":                       ["sex", "gender_code", "m_f"],
    "tenure_months":                ["tenure", "months_at_company",
                                     "length_of_service", "service_months",
                                     "los", "experience_months", "total_tenure"],
    "months_since_promotion":       ["last_promotion_months", "promotion_gap",
                                     "months_since_last_promo", "promo_gap",
                                     "time_since_promotion"],
    "months_since_hike":            ["last_hike_months", "hike_gap",
                                     "months_since_last_hike", "salary_revision_gap"],
    "months_since_manager_changed": ["manager_change_months", "mgr_change_gap",
                                     "reporting_change_months"],
    "compa_ratio":                  ["comp_ratio", "compensation_ratio",
                                     "salary_ratio", "market_ratio", "pay_ratio"],
    "hike_vs_band_median":          ["hike_delta", "hike_vs_median",
                                     "increment_vs_median", "hike_comparison"],
    "perf_rating_current":          ["performance_rating", "perf_rating",
                                     "rating", "appraisal_rating", "perf_score",
                                     "annual_rating", "last_rating"],
    "login_freq_30d":               ["login_count", "logins_30d", "monthly_logins",
                                     "system_logins", "active_days"],
    "login_freq_slope_30d":         ["login_trend", "login_slope", "activity_trend"],
    "training_completions_90d":     ["trainings_completed", "training_count",
                                     "lms_completions", "courses_completed"],
    "leave_days_30d":               ["leave_days", "leaves_taken", "absent_days",
                                     "monthly_leaves"],
    "performance_rating_delta":     ["rating_change", "rating_delta",
                                     "perf_change", "rating_trend"],
    "peer_attrition_count_90d":     ["peer_exits", "team_attrition_count",
                                     "colleagues_left"],
    "manager_attrition_rate_6m":    ["mgr_attrition", "manager_exit_rate",
                                     "reporting_attrition"],
    "team_size_change_pct":         ["team_change", "headcount_change",
                                     "team_growth_pct"],
    "attrition_flag":               ["attrition", "left_flag", "resigned",
                                     "exit_flag", "churned", "left_company",
                                     "is_attrition", "turnover_flag"],
}

# Flatten alias map for quick lookup
ALIAS_TO_CANONICAL = {}
for canonical, aliases in KNOWN_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower().strip()] = canonical

# ============================================================
# STEP 1: Load raw file (any format)
# ============================================================
def load_raw_file(filepath: str) -> pd.DataFrame:
    """
    Loads CSV or Excel regardless of encoding, separator, or BOM.
    Tries multiple encodings before failing.
    Handles files with leading/trailing whitespace in headers.
    """
    path = Path(filepath)
    audit = []

    if path.suffix.lower() in [".xlsx", ".xls"]:
        # Excel — try multiple sheet strategies
        try:
            df = pd.read_excel(filepath, sheet_name=0)
            audit.append(f"Loaded Excel: {path.name}, sheet 0")
        except Exception:
            xl = pd.ExcelFile(filepath)
            df = pd.read_excel(filepath, sheet_name=xl.sheet_names[0])
            audit.append(f"Loaded Excel: {path.name}, sheet {xl.sheet_names[0]}")

    else:
        # CSV — try multiple encodings and separators
        encodings   = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        separators  = [",", ";", "\t", "|"]
        loaded      = False

        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding=enc,
                        sep=sep,
                        skipinitialspace=True,  # handle "col1, col2" with spaces
                        on_bad_lines="warn"
                    )
                    if df.shape[1] > 1:  # more than 1 column = correct separator
                        audit.append(f"Loaded CSV: encoding={enc}, sep='{sep}'")
                        loaded = True
                        break
                except Exception:
                    continue
            if loaded:
                break

        if not loaded:
            raise ValueError(f"Could not load {filepath} with any encoding/separator")

    audit.append(f"Raw shape: {df.shape}")
    return df, audit

# ============================================================
# STEP 2: Clean column names
# ============================================================
def clean_column_names(df: pd.DataFrame) -> tuple:
    """
    Normalizes column names:
    - Lowercase
    - Strip whitespace
    - Replace spaces/hyphens/dots with underscores
    - Remove special characters
    - Remove BOM characters
    """
    audit    = []
    original = df.columns.tolist()

    new_cols = []
    for col in original:
        cleaned = str(col)
        cleaned = cleaned.replace("\ufeff", "")     # BOM character
        cleaned = cleaned.lower().strip()           # lowercase + strip
        cleaned = re.sub(r"[\s\-\.]+", "_", cleaned) # spaces/hyphens/dots → _
        cleaned = re.sub(r"[^\w]", "", cleaned)     # remove special chars
        cleaned = re.sub(r"_+", "_", cleaned)       # multiple _ → single _
        cleaned = cleaned.strip("_")                # leading/trailing _
        new_cols.append(cleaned)

    df.columns = new_cols

    changed = [(o, n) for o, n in zip(original, new_cols) if o != n]
    if changed:
        audit.append(f"Cleaned {len(changed)} column names:")
        for orig, new in changed[:10]:  # show first 10
            audit.append(f"  '{orig}' → '{new}'")

    return df, audit

# ============================================================
# STEP 3: Fuzzy column mapping
# ============================================================
def map_columns_fuzzy(df: pd.DataFrame) -> tuple:
    """
    Maps real column names to canonical names using:
    1. Exact match (after cleaning)
    2. Known alias lookup
    3. rapidfuzz similarity (threshold 80)

    Returns df with renamed columns + mapping audit trail.
    """
    audit   = []
    mapping = {}  # real_col → canonical_col
    unmapped= []

    for col in df.columns:
        col_clean = col.lower().strip()

        # Method 1: Exact match to canonical
        if col_clean in CANONICAL_COLUMNS:
            mapping[col] = col_clean
            continue

        # Method 2: Known alias lookup
        if col_clean in ALIAS_TO_CANONICAL:
            canonical = ALIAS_TO_CANONICAL[col_clean]
            mapping[col] = canonical
            audit.append(f"Alias match: '{col}' → '{canonical}'")
            continue

        # Method 3: rapidfuzz similarity
        # Matches against all canonical names + all aliases
        all_targets = list(CANONICAL_COLUMNS.keys()) + list(ALIAS_TO_CANONICAL.keys())
        match, score, _ = process.extractOne(
            col_clean,
            all_targets,
            scorer=fuzz.token_sort_ratio  # handles word order differences
        )

        if score >= 80:
            # Resolve alias to canonical if needed
            canonical = ALIAS_TO_CANONICAL.get(match, match)
            if canonical in CANONICAL_COLUMNS:
                mapping[col] = canonical
                audit.append(f"Fuzzy match ({score}%): '{col}' → '{canonical}'")
                continue

        # No match found
        unmapped.append(col)
        audit.append(f"[WARN]  Unmapped column: '{col}' (best match: '{match}' at {score}%)")

    # Rename columns
    df = df.rename(columns=mapping)

    # Drop unmapped columns
    df = df.drop(columns=[c for c in unmapped if c in df.columns], errors="ignore")

    audit.append(f"\nMapped: {len(mapping)} columns")
    audit.append(f"Unmapped/dropped: {len(unmapped)} columns")

    return df, audit, mapping

# ============================================================
# STEP 4: Null handling
# ============================================================
def handle_nulls(df: pd.DataFrame) -> tuple:
    """
    Standardizes null representations:
    - "NA", "N/A", "n/a", "nan", "NULL", "null", "-", "?",
      ".", "none", "NONE", "#N/A", "na", "" → np.nan
    Then imputes based on column type and business logic.
    """
    audit = []

    # String representations of null
    NULL_STRINGS = [
        "na", "n/a", "nan", "null", "none", "-", "?",
        ".", "#n/a", "#na", "missing", "unknown", "",
        "not available", "not applicable", "nil"
    ]

    # Replace all null string variants with np.nan
    df = df.replace(NULL_STRINGS, np.nan, regex=False)

    # Also catch case-insensitive variants
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: np.nan
            if isinstance(x, str) and x.strip().lower() in NULL_STRINGS
            else x
        )

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]

    if len(null_cols) > 0:
        audit.append("Null counts before imputation:")
        for col, count in null_cols.items():
            pct = count / len(df) * 100
            audit.append(f"  {col}: {count} ({pct:.1f}%)")

    # Imputation strategy per column
    imputation_rules = {
        # Numeric — median imputation (robust to outliers)
        "tenure_months":                "median",
        "months_since_promotion":       "median",
        "months_since_hike":            "median",
        "months_since_manager_changed": "median",
        "compa_ratio":                  "median",
        "hike_vs_band_median":          0.0,       # assume at median if unknown
        "perf_rating_current":          "median",
        "login_freq_30d":               "median",
        "login_freq_slope_30d":         0.0,       # assume flat trend if unknown
        "training_completions_90d":     0,         # assume zero if unknown
        "leave_days_30d":               "median",
        "performance_rating_delta":     0.0,       # assume no change if unknown
        "peer_attrition_count_90d":     0,         # assume no peers left
        "manager_attrition_rate_6m":    "median",
        "team_size_change_pct":         0.0,       # assume stable team
        # Categorical — mode imputation
        "circle":                       "mode",
        "department":                   "mode",
        "gender":                       "mode",
        "band":                         "mode",
    }

    for col, strategy in imputation_rules.items():
        if col not in df.columns:
            continue
        null_mask = df[col].isnull()
        if null_mask.sum() == 0:
            continue

        if strategy == "median":
            fill_val = df[col].median()
        elif strategy == "mode":
            fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
        else:
            fill_val = strategy

        df[col] = df[col].fillna(fill_val)
        audit.append(f"Imputed {col}: {null_mask.sum()} nulls → {strategy} ({fill_val})")

    return df, audit

# ============================================================
# STEP 5: Type coercion
# ============================================================
def clean_numeric(val):
    """Strip currency symbols, commas, percentages from numeric strings."""
    if pd.isnull(val):
        return np.nan
    s = str(val).strip()
    s = re.sub(r"[₹$,\s]", "", s)  # remove currency + commas
    s = s.replace("%", "")          # remove percentage sign
    s = s.replace("−", "-")         # handle unicode minus
    try:
        return float(s)
    except ValueError:
        return np.nan

def coerce_types(df: pd.DataFrame) -> tuple:
    """
    Converts columns to correct types.
    Handles:
    - "8.5%" → 8.5 (percentage strings)
    - "₹9,80,000" → 980000 (currency strings)
    - "Yes/No" → 1/0 (binary strings)
    - Mixed date formats → months
    """
    audit = []

    # Numeric columns — clean and convert. attrition_flag is excluded
    # even though it's typed "int" in CANONICAL_COLUMNS — it gets its
    # own text-to-binary handling below, and running it through
    # clean_numeric first would wipe values like "Resigned"/"Active" to
    # NaN (float() can't parse them) before binary_maps ever sees them.
    numeric_cols = [
        col for col, meta in CANONICAL_COLUMNS.items()
        if meta["type"] in ["float", "int"] and col in df.columns
        and col != "attrition_flag"
    ]

    for col in numeric_cols:
        before_nulls = df[col].isnull().sum()
        df[col]      = df[col].apply(clean_numeric)
        after_nulls  = df[col].isnull().sum()

        if after_nulls > before_nulls:
            audit.append(
                f"[WARN]  {col}: {after_nulls - before_nulls} values "
                f"couldn't convert to numeric → set to NaN"
            )

    # Binary columns — Yes/No, True/False, Y/N → 1/0
    binary_maps = {
        "yes": 1, "no": 0, "y": 1, "n": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "resigned": 1, "active": 0,
        "left": 1, "stayed": 0,
    }

    if "attrition_flag" in df.columns:
        before = df["attrition_flag"].copy()
        df["attrition_flag"] = df["attrition_flag"].apply(
            lambda x: binary_maps.get(str(x).lower().strip(), x)
            if not pd.isnull(x) else np.nan
        )
        changed = (before != df["attrition_flag"]).sum()
        if changed > 0:
            audit.append(f"Converted attrition_flag text → binary: {changed} values")

    # Gender standardization
    if "gender" in df.columns:
        gender_map = {
            "male": "M", "female": "F", "m": "M", "f": "F",
            "1": "M", "0": "F", "man": "M", "woman": "F"
        }
        df["gender"] = df["gender"].apply(
            lambda x: gender_map.get(str(x).lower().strip(), x)
            if not pd.isnull(x) else x
        )

    # Compa ratio: sometimes stored as percentage (85 instead of 0.85)
    if "compa_ratio" in df.columns:
        median_cr = df["compa_ratio"].median()
        if median_cr > 5:  # stored as percentage not decimal
            df["compa_ratio"] = df["compa_ratio"] / 100
            audit.append("compa_ratio divided by 100 — was stored as percentage")

    return df, audit

# ============================================================
# STEP 6: Date columns → months
# ============================================================
def compute_date_derived_features(df: pd.DataFrame) -> tuple:
    """
    If raw data has date columns instead of pre-computed months,
    compute months_since_X from the dates.
    Handles multiple date formats automatically.
    """
    audit      = []
    today      = pd.Timestamp.today()

    date_column_map = {
        "hire_date":           "tenure_months",
        "joining_date":        "tenure_months",
        "date_of_joining":     "tenure_months",
        "last_promotion_date": "months_since_promotion",
        "promotion_date":      "months_since_promotion",
        "last_hike_date":      "months_since_hike",
        "salary_revision_date":"months_since_hike",
        "manager_change_date": "months_since_manager_changed",
    }

    for date_col, target_col in date_column_map.items():
        if date_col not in df.columns:
            continue
        if target_col in df.columns and df[target_col].notna().sum() > len(df) * 0.5:
            continue  # already have this column with good data — skip

        try:
            # infer_datetime_format handles most formats automatically
            dates      = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
            months     = ((today - dates).dt.days / 30.44).round(1)
            df[target_col] = months
            audit.append(f"Computed {target_col} from {date_col}")
        except Exception as e:
            audit.append(f"[WARN]  Could not parse {date_col}: {str(e)}")

    return df, audit

# ============================================================
# STEP 7: Outlier capping
# ============================================================
def cap_outliers(df: pd.DataFrame) -> tuple:
    """
    Caps extreme values at 1st/99th percentile.
    Prevents single data entry errors from breaking model.
    Only applied to continuous numeric features.
    """
    audit = []

    cap_cols = [
        "tenure_months", "compa_ratio", "login_freq_30d",
        "leave_days_30d", "perf_rating_current",
        "months_since_promotion", "months_since_hike",
        "peer_attrition_count_90d", "manager_attrition_rate_6m"
    ]

    for col in cap_cols:
        if col not in df.columns:
            continue

        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)

        outliers_low  = (df[col] < p01).sum()
        outliers_high = (df[col] > p99).sum()

        if outliers_low + outliers_high > 0:
            df[col] = df[col].clip(lower=p01, upper=p99)
            audit.append(
                f"Capped {col}: {outliers_low} low + "
                f"{outliers_high} high outliers → [{p01:.2f}, {p99:.2f}]"
            )

    return df, audit

# ============================================================
# STEP 8: Peer attrition small group validation
# ============================================================
def validate_peer_groups(df: pd.DataFrame,
                          min_group_size: int = 5) -> tuple:
    """
    Peer attrition count is only meaningful when group has
    sufficient members. Small groups (< 5) have noisy counts.
    Sets peer_attrition_count to 0 for small groups.
    """
    audit = []

    if "peer_attrition_count_90d" not in df.columns:
        return df, audit

    group_sizes = df.groupby(["department", "circle"]).size()

    small_groups = group_sizes[group_sizes < min_group_size]
    if len(small_groups) > 0:
        audit.append(f"Small peer groups (n<{min_group_size}) — zeroing peer attrition:")
        for (dept, circle), size in small_groups.items():
            mask = (df["department"] == dept) & (df["circle"] == circle)
            df.loc[mask, "peer_attrition_count_90d"] = 0
            audit.append(f"  {dept} / {circle}: n={size} → peer_attrition set to 0")

    return df, audit

# ============================================================
# STEP 9: Final validation
# ============================================================
def validate_final(df: pd.DataFrame) -> tuple:
    """
    Checks required columns are present and non-null.
    Reports any remaining quality issues.
    """
    audit  = []
    issues = []

    for col, meta in CANONICAL_COLUMNS.items():
        if meta["required"] and col not in df.columns:
            issues.append(f"CRITICAL: Required column missing: {col}")
        elif col in df.columns and not meta["nullable"]:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"WARNING: {col} has {null_count} nulls (should be 0)")

    # Check attrition flag is binary
    if "attrition_flag" in df.columns:
        unique_vals = df["attrition_flag"].dropna().unique()
        non_binary  = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
        if non_binary:
            issues.append(f"WARNING: attrition_flag has non-binary values: {non_binary}")

    # Check band range
    if "band" in df.columns:
        invalid_bands = df[~df["band"].isin([3, 4, 5, 6, 7])]["band"].unique()
        if len(invalid_bands) > 0:
            audit.append(f"[WARN]  Unexpected band values: {invalid_bands} — check mapping")

    if issues:
        audit.append("\nValidation issues:")
        for issue in issues:
            audit.append(f"  {issue}")
    else:
        audit.append("[OK] All validation checks passed")

    return df, audit, issues

# ============================================================
# MASTER PIPELINE — run all steps
# ============================================================
def run_mapper(input_filepath: str,
               output_filepath: str = None) -> pd.DataFrame:
    """
    Runs complete ingestion pipeline on any HR data file.
    Saves cleaned data to output_filepath (or employees.csv).
    Returns cleaned DataFrame.
    """
    full_audit = []
    full_audit.append(f"=== MAPPER RUN: {datetime.now().isoformat()} ===")
    full_audit.append(f"Input: {input_filepath}")

    # Step 1: Load
    print("Step 1: Loading file...")
    df, audit = load_raw_file(input_filepath)
    full_audit.extend(audit)

    # Step 2: Clean column names
    print("Step 2: Cleaning column names...")
    df, audit = clean_column_names(df)
    full_audit.extend(audit)

    # Step 3: Fuzzy column mapping
    print("Step 3: Mapping columns (fuzzy)...")
    df, audit, mapping = map_columns_fuzzy(df)
    full_audit.extend(audit)

    # Step 4: Date-derived features
    print("Step 4: Computing date-derived features...")
    df, audit = compute_date_derived_features(df)
    full_audit.extend(audit)

    # Step 5: Null handling
    print("Step 5: Handling nulls...")
    df, audit = handle_nulls(df)
    full_audit.extend(audit)

    # Step 6: Type coercion
    print("Step 6: Coercing types...")
    df, audit = coerce_types(df)
    full_audit.extend(audit)

    # Step 7: Outlier capping
    print("Step 7: Capping outliers...")
    df, audit = cap_outliers(df)
    full_audit.extend(audit)

    # Step 8: Peer group validation
    print("Step 8: Validating peer groups...")
    df, audit = validate_peer_groups(df)
    full_audit.extend(audit)

    # Step 9: Final validation
    print("Step 9: Final validation...")
    df, audit, issues = validate_final(df)
    full_audit.extend(audit)

    # Save cleaned data
    if output_filepath is None:
        output_filepath = cfg["data"]["output_path"]

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_filepath, index=False)

    # Save audit trail
    audit_path = Path("logs/mapper_audit.txt")
    audit_path.parent.mkdir(exist_ok=True)
    audit_path.write_text("\n".join(full_audit), encoding="utf-8")

    # Save column mapping
    mapping_path = Path("logs/column_mapping.json")
    mapping_path.write_text(json.dumps(mapping, indent=2))

    # Print summary
    print(f"\n{'='*50}")
    print("[OK] Mapper complete")
    print(f"Input rows:    {len(df)}")
    print(f"Output cols:   {len(df.columns)}")
    print(f"Output path:   {output_filepath}")
    print("Audit trail:   logs/mapper_audit.txt")
    print("Column map:    logs/column_mapping.json")

    if issues:
        print(f"\n[WARN]  {len(issues)} validation issues — check logs/mapper_audit.txt")
    else:
        print("Validation:    [OK] all checks passed")

    return df

# ============================================================
# Run directly with real data file
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python data/raw/mapper.py path/to/real_data.xlsx
        input_file = sys.argv[1]
    else:
        # Default test — uses synthetic data to verify mapper works
        input_file = cfg["data"]["output_path"]
        print(f"No input file specified — testing on synthetic data: {input_file}")

    df_clean = run_mapper(input_file)

    print("\nFirst 3 rows:")
    print(df_clean.head(3).to_string())
    print("\nColumn list:")
    print(df_clean.columns.tolist())