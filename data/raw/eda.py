# ============================================================
# eda.py — Exploratory Data Analysis for Real Jio HR Data
# ============================================================
# PURPOSE: Validates real data before feature engineering
# FLOW: employees_clean.csv → checks → eda_report.html
# RUN: python data/raw/eda.py [optional: path/to/data.csv]
# MANDATORY before first real data pipeline run
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ============================================================
# REQUIRED COLUMNS — pipeline breaks without these
# ============================================================
REQUIRED_COLS = [
    "employee_id", "band", "department", "circle",
    "tenure_months", "attrition_flag",
    "compa_ratio", "perf_rating_current",
    "months_since_promotion", "months_since_hike",
    "login_freq_30d"
]

NUMERIC_COLS = [
    "tenure_months", "compa_ratio", "perf_rating_current",
    "months_since_promotion", "months_since_hike",
    "login_freq_30d", "leave_days_30d",
    "training_completions_90d", "manager_attrition_rate_6m",
    "peer_attrition_count_90d", "team_size_change_pct",
    "login_freq_slope_30d", "performance_rating_delta"
]

# Expected value ranges — flag if outside these
EXPECTED_RANGES = {
    "compa_ratio":           (0.3,  2.0),
    "perf_rating_current":   (1.0,  5.0),
    "tenure_months":         (0,    360),
    "months_since_promotion":(0,    120),
    "months_since_hike":     (0,    60),
    "login_freq_30d":        (0,    31),
    "leave_days_30d":        (0,    31),
    "band":                  (1,    10),
}

# ============================================================
# HELPER — save plot
# ============================================================
def save_plot(fig, name: str, plots_dir: Path):
    path = plots_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

# ============================================================
# CHECK 1 — Basic Data Quality
# ============================================================
def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Checks shape, dtypes, nulls, duplicates.
    Returns dict of findings.
    """
    results = {}

    results["shape"]        = {"rows": len(df), "cols": len(df.columns)}
    results["columns"]      = df.columns.tolist()

    # Missing required columns
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    results["missing_required"] = missing_required

    # Null analysis
    null_counts = df.isnull().sum()
    null_pcts   = (null_counts / len(df) * 100).round(2)
    results["nulls"] = {
        col: {"count": int(null_counts[col]), "pct": float(null_pcts[col])}
        for col in df.columns if null_counts[col] > 0
    }

    # Duplicate employee IDs
    if "employee_id" in df.columns:
        dup_count = df["employee_id"].duplicated().sum()
        results["duplicate_employee_ids"] = int(dup_count)

    # Attrition flag validation
    if "attrition_flag" in df.columns:
        unique_vals = df["attrition_flag"].dropna().unique().tolist()
        non_binary  = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
        results["attrition_flag_values"]    = unique_vals
        results["attrition_flag_non_binary"]= non_binary

    return results

# ============================================================
# CHECK 2 — Attrition Rate Analysis
# ============================================================
def check_attrition_rate(df: pd.DataFrame, plots_dir: Path) -> dict:
    """
    Overall attrition rate + breakdown by key dimensions.
    Flags if rate differs significantly from config target.
    """
    results = {}

    if "attrition_flag" not in df.columns:
        return {"error": "attrition_flag missing"}

    overall_rate = df["attrition_flag"].mean()
    target_rate  = cfg["data"]["attrition_rate"]
    deviation    = abs(overall_rate - target_rate)

    results["overall_rate"]  = round(float(overall_rate), 4)
    results["target_rate"]   = target_rate
    results["deviation"]     = round(float(deviation), 4)
    results["flag"]          = "[WARN]  HIGH DEVIATION" if deviation > 0.08 else "[OK] OK"

    # Breakdown by dimension
    for dim in ["band", "department", "circle", "gender"]:
        if dim in df.columns:
            breakdown = df.groupby(dim)["attrition_flag"].agg(
                ["mean", "count"]
            ).round(3)
            breakdown.columns = ["attrition_rate", "count"]
            results[f"by_{dim}"] = breakdown.to_dict(orient="index")

    # Plot attrition by band and department
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "band" in df.columns:
        band_rates = df.groupby("band")["attrition_flag"].mean()
        axes[0].bar(band_rates.index.astype(str), band_rates.values,
                    color=["#e74c3c" if r > 0.25 else "#f1c40f"
                           if r > 0.15 else "#2ecc71" for r in band_rates.values])
        axes[0].axhline(y=overall_rate, color="black",
                        linestyle="--", label=f"Overall: {overall_rate:.1%}")
        axes[0].set_title("Attrition Rate by Band")
        axes[0].set_xlabel("Band")
        axes[0].set_ylabel("Attrition Rate")
        axes[0].legend()

    if "department" in df.columns:
        dept_rates = df.groupby("department")["attrition_flag"].mean().sort_values()
        axes[1].barh(dept_rates.index, dept_rates.values,
                     color=["#e74c3c" if r > 0.25 else "#f1c40f"
                            if r > 0.15 else "#2ecc71" for r in dept_rates.values])
        axes[1].axvline(x=overall_rate, color="black",
                        linestyle="--", label=f"Overall: {overall_rate:.1%}")
        axes[1].set_title("Attrition Rate by Department")
        axes[1].legend()

    plt.tight_layout()
    save_plot(fig, "attrition_by_dimension", plots_dir)

    return results

# ============================================================
# CHECK 3 — Feature Distribution Analysis
# ============================================================
def check_distributions(df: pd.DataFrame, plots_dir: Path) -> dict:
    """
    Distribution stats for all numeric features.
    Flags values outside expected ranges.
    """
    results = {}

    available_numeric = [c for c in NUMERIC_COLS if c in df.columns]

    for col in available_numeric:
        stats = df[col].describe().round(3).to_dict()
        out_of_range = {}

        if col in EXPECTED_RANGES:
            low, high = EXPECTED_RANGES[col]
            below = (df[col] < low).sum()
            above = (df[col] > high).sum()
            if below + above > 0:
                out_of_range = {
                    "below_min": int(below),
                    "above_max": int(above),
                    "expected": f"[{low}, {high}]"
                }

        results[col] = {
            "stats":        stats,
            "out_of_range": out_of_range,
            "flag":         "[WARN]  CHECK" if out_of_range else "[OK] OK"
        }

    # Plot distributions of key features
    key_features = [c for c in [
        "tenure_months", "compa_ratio", "perf_rating_current",
        "months_since_promotion", "login_freq_30d"
    ] if c in df.columns]

    if key_features:
        fig, axes = plt.subplots(1, len(key_features),
                                 figsize=(4 * len(key_features), 4))
        if len(key_features) == 1:
            axes = [axes]

        for ax, col in zip(axes, key_features):
            ax.hist(df[col].dropna(), bins=30,
                    color="#3498db", alpha=0.7, edgecolor="white")
            ax.set_title(col.replace("_", " ").title(), fontsize=9)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

        plt.tight_layout()
        save_plot(fig, "feature_distributions", plots_dir)

    return results

# ============================================================
# CHECK 4 — Attrition by Subgroup
# ============================================================
def check_subgroups(df: pd.DataFrame, plots_dir: Path) -> dict:
    """
    Identifies high-attrition subgroups.
    Flags segments where attrition > 2× overall rate.
    """
    results   = {}
    overall   = df["attrition_flag"].mean()
    threshold = overall * 2  # flag if 2× average

    high_risk_segments = []

    # Band × Department cross-tab
    if "band" in df.columns and "department" in df.columns:
        cross = df.groupby(["band", "department"]).agg(
            attrition_rate=("attrition_flag", "mean"),
            count=("employee_id", "count")
        ).round(3).reset_index()

        # Flag segments with high attrition and sufficient sample
        flagged = cross[
            (cross["attrition_rate"] > threshold) &
            (cross["count"] >= 10)
        ]
        for _, row in flagged.iterrows():
            high_risk_segments.append({
                "segment":        f"Band {row['band']} - {row['department']}",
                "attrition_rate": float(row["attrition_rate"]),
                "count":          int(row["count"]),
                "vs_overall":     f"{row['attrition_rate']/overall:.1f}×"
            })

    # High performer attrition — most important business metric
    if "perf_rating_current" in df.columns:
        high_perf      = df[df["perf_rating_current"] >= 4.0]
        hp_rate        = high_perf["attrition_flag"].mean()
        results["high_performer_attrition"] = {
            "rate":    round(float(hp_rate), 4),
            "count":   len(high_perf),
            "vs_overall": f"{hp_rate/overall:.1f}×",
            "flag":    "[ALERT] CRITICAL" if hp_rate > overall * 1.5 else "[OK] OK"
        }

    results["high_risk_segments"] = high_risk_segments
    results["segments_flagged"]   = len(high_risk_segments)

    # Heatmap: band × department attrition
    if "band" in df.columns and "department" in df.columns:
        pivot = df.pivot_table(
            values="attrition_flag",
            index="band",
            columns="department",
            aggfunc="mean"
        ).round(3)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            pivot, annot=True, fmt=".2f",
            cmap="RdYlGn_r", center=overall,
            linewidths=0.5, ax=ax
        )
        ax.set_title(
            f"Attrition Rate Heatmap: Band × Department\n"
            f"(Overall: {overall:.1%})",
            fontsize=12
        )
        plt.tight_layout()
        save_plot(fig, "attrition_heatmap", plots_dir)

    return results

# ============================================================
# CHECK 5 — Temporal Patterns
# ============================================================
def check_temporal_patterns(df: pd.DataFrame, plots_dir: Path) -> dict:
    """
    Checks if attrition patterns match expected temporal signals.
    Post-appraisal window, tenure cohorts, promotion gaps.
    """
    results = {}

    # Post-appraisal window: months 1-2 after hike should show higher attrition
    if "months_since_hike" in df.columns:
        df["hike_bucket"] = pd.cut(
            df["months_since_hike"],
            bins=[0, 2, 6, 12, 24, 999],
            labels=["0-2m", "2-6m", "6-12m", "12-24m", "24m+"]
        )
        hike_attrition = df.groupby("hike_bucket", observed=True)[
            "attrition_flag"
        ].mean().round(3)
        results["attrition_by_hike_gap"] = hike_attrition.to_dict()
        results["post_appraisal_signal"] = (
            "[OK] Confirmed" if hike_attrition.get("0-2m", 0) >
            hike_attrition.get("6-12m", 0) else "[WARN]  Not detected"
        )

    # Tenure cohorts: new joiners vs mid-tenure vs long-tenure
    if "tenure_months" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 12, 36, 72, 120, 999],
            labels=["0-1yr", "1-3yr", "3-6yr", "6-10yr", "10yr+"]
        )
        tenure_attrition = df.groupby("tenure_bucket", observed=True)[
            "attrition_flag"
        ].mean().round(3)
        results["attrition_by_tenure"] = tenure_attrition.to_dict()

        # New joiner attrition
        new_joiner_rate = df[df["tenure_months"] <= 12]["attrition_flag"].mean()
        results["new_joiner_attrition"] = {
            "rate":  round(float(new_joiner_rate), 4),
            "count": int((df["tenure_months"] <= 12).sum()),
            "flag":  "[ALERT] HIGH" if new_joiner_rate > 0.30 else "[OK] OK"
        }

    # Plot tenure attrition curve
    if "tenure_months" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        tenure_attrition.plot(kind="bar", ax=axes[0],
                              color="#e74c3c", alpha=0.8)
        axes[0].set_title("Attrition Rate by Tenure Cohort")
        axes[0].set_xlabel("Tenure")
        axes[0].set_ylabel("Attrition Rate")
        axes[0].tick_params(axis="x", rotation=0)

        if "months_since_hike" in df.columns:
            hike_attrition.plot(kind="bar", ax=axes[1],
                                color="#e67e22", alpha=0.8)
            axes[1].set_title("Attrition Rate by Months Since Last Hike")
            axes[1].set_xlabel("Time Since Hike")
            axes[1].set_ylabel("Attrition Rate")
            axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        save_plot(fig, "temporal_patterns", plots_dir)

    return results

# ============================================================
# CHECK 6 — Outlier Detection
# ============================================================
def check_outliers(df: pd.DataFrame) -> dict:
    """
    Identifies extreme values that will break model.
    Uses IQR method — more robust than std for skewed data.
    """
    results = {}

    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 3 * iqr  # 3× IQR = extreme outliers only
        upper = q3 + 3 * iqr

        extreme_low  = (df[col] < lower).sum()
        extreme_high = (df[col] > upper).sum()

        if extreme_low + extreme_high > 0:
            results[col] = {
                "extreme_low":  int(extreme_low),
                "extreme_high": int(extreme_high),
                "lower_fence":  round(float(lower), 3),
                "upper_fence":  round(float(upper), 3),
                "max_value":    round(float(df[col].max()), 3),
                "flag":         "[WARN]  CHECK"
            }

    return results

# ============================================================
# CHECK 7 — Pipeline Readiness
# ============================================================
def check_pipeline_readiness(df: pd.DataFrame,
                              quality: dict,
                              distributions: dict) -> dict:
    """
    Final go/no-go check before running feature engineering.
    Green = run pipeline. Red = fix data first.
    """
    blockers  = []
    warnings_ = []

    # Blockers — pipeline will fail
    if quality.get("missing_required"):
        blockers.append(
            f"Missing required columns: {quality['missing_required']}"
        )

    if quality.get("duplicate_employee_ids", 0) > 0:
        blockers.append(
            f"{quality['duplicate_employee_ids']} duplicate employee IDs"
        )

    if quality.get("attrition_flag_non_binary"):
        blockers.append(
            f"Non-binary attrition_flag values: "
            f"{quality['attrition_flag_non_binary']}"
        )

    # High null rate in required columns
    for col in REQUIRED_COLS:
        null_info = quality.get("nulls", {}).get(col, {})
        if null_info.get("pct", 0) > 20:
            blockers.append(
                f"{col} has {null_info['pct']}% nulls — too high"
            )

    # Warnings — pipeline runs but results may be unreliable
    attrition_deviation = quality.get("deviation", 0)
    if attrition_deviation > 0.08:
        warnings_.append(
            f"Attrition rate deviates {attrition_deviation:.1%} "
            f"from config target — update scale_pos_weight"
        )

    for col, info in distributions.items():
        if info.get("out_of_range"):
            oor = info["out_of_range"]
            total = oor.get("below_min", 0) + oor.get("above_max", 0)
            if total > 10:
                warnings_.append(
                    f"{col}: {total} values outside expected range "
                    f"{oor['expected']}"
                )

    verdict = "[FAIL] BLOCKED" if blockers else "[OK] READY"

    return {
        "verdict":   verdict,
        "blockers":  blockers,
        "warnings":  warnings_,
        "can_proceed": len(blockers) == 0
    }

# ============================================================
# GENERATE HTML REPORT
# ============================================================
def generate_html_report(results: dict, output_path: Path):
    """
    Generates a clean HTML report from all EDA results.
    Saves to logs/eda_report.html — open in browser.
    """
    def format_dict(d, indent=0):
        if not isinstance(d, dict):
            return str(d)
        rows = ""
        for k, v in d.items():
            if isinstance(v, dict):
                rows += f"<tr><td><b>{k}</b></td><td>{format_dict(v)}</td></tr>"
            else:
                rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
        return f"<table border='1' cellpadding='4'>{rows}</table>"

    verdict      = results.get("readiness", {}).get("verdict", "UNKNOWN")
    verdict_color = "#2ecc71" if "READY" in verdict else "#e74c3c"

    html = f"""
    <html>
    <head>
        <title>EDA Report — Jio Attrition Intelligence</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            h1   {{ color: #2c3e50; }}
            h2   {{ color: #34495e; border-bottom: 2px solid #eee; }}
            .verdict {{ background: {verdict_color}; color: white;
                        padding: 15px; border-radius: 8px;
                        font-size: 1.4em; margin: 20px 0; }}
            table {{ border-collapse: collapse; margin: 10px 0; }}
            td, th {{ padding: 6px 12px; border: 1px solid #ddd; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            img {{ max-width: 100%; margin: 10px 0; }}
            .blocker {{ color: #e74c3c; font-weight: bold; }}
            .warning {{ color: #e67e22; }}
            .ok      {{ color: #2ecc71; }}
        </style>
    </head>
    <body>
        <h1>EDA Report — Jio Attrition Intelligence System</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="verdict">{verdict}</div>

        <h2>Pipeline Readiness</h2>
        <p><b>Blockers:</b></p>
        <ul>
            {"".join(f"<li class='blocker'>{b}</li>"
                     for b in results["readiness"]["blockers"])
             or "<li class='ok'>None</li>"}
        </ul>
        <p><b>Warnings:</b></p>
        <ul>
            {"".join(f"<li class='warning'>{w}</li>"
                     for w in results["readiness"]["warnings"])
             or "<li class='ok'>None</li>"}
        </ul>

        <h2>Data Quality</h2>
        <p>Shape: {results['quality']['shape']['rows']} rows ×
                  {results['quality']['shape']['cols']} columns</p>
        <p>Duplicate IDs: {results['quality'].get('duplicate_employee_ids', 0)}</p>
        <p>Null columns: {len(results['quality'].get('nulls', {}))}</p>

        <h2>Attrition Rate</h2>
        <p>Overall: <b>{results['attrition']['overall_rate']:.1%}</b>
           (target: {results['attrition']['target_rate']:.1%})
           {results['attrition']['flag']}</p>
        <img src='../models/attrition_by_dimension.png'>

        <h2>High Risk Segments</h2>
        <p>Segments with attrition > 2× average:
           <b>{results['subgroups']['segments_flagged']}</b></p>
        {"".join(f"<p>• {s['segment']}: {s['attrition_rate']:.1%} ({s['vs_overall']})</p>"
                 for s in results['subgroups']['high_risk_segments'])}
        <img src='../models/attrition_heatmap.png'>

        <h2>Feature Distributions</h2>
        <img src='../models/feature_distributions.png'>

        <h2>Temporal Patterns</h2>
        <p>Post-appraisal signal:
           {results['temporal'].get('post_appraisal_signal', 'N/A')}</p>
        <p>New joiner attrition:
           {results['temporal'].get('new_joiner_attrition', {}).get('rate', 'N/A')}</p>
        <img src='../models/temporal_patterns.png'>

        <h2>Outliers</h2>
        {"".join(f"<p>[WARN]  {col}: {info['extreme_low']} low + {info['extreme_high']} high extremes</p>"
                 for col, info in results['outliers'].items())
         or "<p class='ok'>[OK] No extreme outliers detected</p>"}

    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")

# ============================================================
# MASTER EDA PIPELINE
# ============================================================
def run_eda(input_filepath: str = None) -> dict:
    """
    Runs complete EDA on HR data file.
    Generates plots + HTML report.
    Returns results dict for programmatic use.
    """
    if input_filepath is None:
        input_filepath = cfg["data"]["output_path"]
        print(f"No input specified — running on: {input_filepath}")

    print(f"Loading data: {input_filepath}")
    df = pd.read_csv(input_filepath)
    print(f"Shape: {df.shape}")

    # Create output directories
    plots_dir  = Path("models")
    logs_dir   = Path("logs")
    plots_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    results = {}

    print("\nCheck 1: Data quality...")
    results["quality"] = check_data_quality(df)

    print("Check 2: Attrition rate analysis...")
    results["attrition"] = check_attrition_rate(df, plots_dir)

    print("Check 3: Feature distributions...")
    results["distributions"] = check_distributions(df, plots_dir)

    print("Check 4: Subgroup analysis...")
    results["subgroups"] = check_subgroups(df, plots_dir)

    print("Check 5: Temporal patterns...")
    results["temporal"] = check_temporal_patterns(df, plots_dir)

    print("Check 6: Outlier detection...")
    results["outliers"] = check_outliers(df)

    print("Check 7: Pipeline readiness...")
    results["readiness"] = check_pipeline_readiness(
        df, results["quality"], results["distributions"]
    )

    # Save JSON results
    json_path = logs_dir / "eda_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate HTML report
    report_path = logs_dir / "eda_report.html"
    generate_html_report(results, report_path)

    # Print summary
    print(f"\n{'='*50}")
    print(f"EDA COMPLETE")
    print(f"{'='*50}")
    print(f"Verdict:     {results['readiness']['verdict']}")
    print(f"Rows:        {results['quality']['shape']['rows']}")
    print(f"Attrition:   {results['attrition']['overall_rate']:.1%} "
          f"({results['attrition']['flag']})")
    print(f"Blockers:    {len(results['readiness']['blockers'])}")
    print(f"Warnings:    {len(results['readiness']['warnings'])}")
    print(f"High-risk segments: {results['subgroups']['segments_flagged']}")

    if results["readiness"]["blockers"]:
        print(f"\n[FAIL] BLOCKERS — fix before running pipeline:")
        for b in results["readiness"]["blockers"]:
            print(f"  • {b}")
    else:
        print(f"\n[OK] Data is clean — run feature_engineering.py next")

    if results["readiness"]["warnings"]:
        print(f"\n[WARN]  WARNINGS:")
        for w in results["readiness"]["warnings"]:
            print(f"  • {w}")

    print(f"\nReports saved:")
    print(f"  HTML: {report_path}")
    print(f"  JSON: {json_path}")

    return results


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_eda(input_file)