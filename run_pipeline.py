# ============================================================
# run_pipeline.py — ETL + ML Pipeline Orchestrator
# ============================================================
# PURPOSE: Single command to run entire pipeline end to end
# FLOW: raw data → mapper → eda → features → validate →
#       multicollinearity → train → score → rag → ready
# RUN: python run_pipeline.py
#      python run_pipeline.py --from step3  (resume from step)
#      python run_pipeline.py --data path/to/real_data.xlsx
# ============================================================

import subprocess
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime

# --- Load config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ============================================================
# PIPELINE STEPS
# ============================================================
# Each step has:
#   name:    display name
#   script:  python file to run
#   args:    additional arguments
#   required: if False — skip failure won't block pipeline
# ============================================================
PIPELINE_STEPS = [
    {
        "id":       "step1",
        "name":     "Data Ingestion (Mapper)",
        "script":   "data/raw/mapper.py",
        "required": True,
        "description": "Fuzzy column mapping, null handling, type coercion"
    },
    {
        "id":       "step2",
        "name":     "EDA Validation",
        "script":   "data/raw/eda.py",
        "required": True,
        "description": "Distribution checks, attrition rate, subgroup analysis"
    },
    {
        "id":       "step3",
        "name":     "Feature Engineering",
        "script":   "data/synthetic/feature_engineering.py",
        "required": True,
        "description": "48 features across 6 families + interaction terms"
    },
    {
        "id":       "step4",
        "name":     "Multicollinearity Check",
        "script":   "models/multicollinearity_check.py",
        "required": False,  # warning only — doesn't block
        "description": "VIF + L1 + correlation analysis"
    },
    {
        "id":       "step5",
        "name":     "Feature Validation",
        "script":   "data/raw/validate.py",
        "required": True,
        "description": "Schema + business rule validation before training"
    },
    {
        "id":       "step6",
        "name":     "Model Training",
        "script":   "models/train.py",
        "required": True,
        "description": "XGBoost + LR champion-challenger + Cox PH"
    },
    {
        "id":       "step7",
        "name":     "EV Scoring",
        "script":   "models/ev_scoring.py",
        "required": True,
        "description": "Score all employees with P(attrition) + EV + SHAP"
    },
    {
        "id":       "step8",
        "name":     "RAG Build",
        "script":   "rag/build_rag.py",
        "required": False,  # skip if policies unchanged
        "description": "Embed HR policy docs into ChromaDB"
    },
]

# ============================================================
# RUN ONE STEP
# ============================================================
def run_step(step: dict, extra_args: list = None) -> dict:
    """
    Runs a single pipeline step as subprocess.
    Captures stdout/stderr.
    Returns result dict with pass/fail + output.
    """
    script  = step["script"]
    cmd     = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    start = datetime.now()

    try:
        result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=600
)
        
        end      = datetime.now()
        duration = (end - start).total_seconds()

        passed = result.returncode == 0

        return {
            "step":     step["id"],
            "name":     step["name"],
            "passed":   passed,
            "duration": round(duration, 1),
            "stdout":   result.stdout[-2000:],  # last 2000 chars
            "stderr":   result.stderr[-500:] if result.stderr else "",
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "step":     step["id"],
            "name":     step["name"],
            "passed":   False,
            "duration": 600,
            "stdout":   "",
            "stderr":   "TIMEOUT after 600 seconds",
            "returncode": -1
        }
    except Exception as e:
        return {
            "step":     step["id"],
            "name":     step["name"],
            "passed":   False,
            "duration": 0,
            "stdout":   "",
            "stderr":   str(e),
            "returncode": -1
        }

# ============================================================
# CHECKPOINT SYSTEM
# ============================================================
# Saves progress after each step
# Allows resuming from any step without rerunning earlier ones
# ============================================================
CHECKPOINT_PATH = Path("logs/pipeline_checkpoint.json")

def save_checkpoint(completed_steps: list, run_id: str):
    Path("logs").mkdir(exist_ok=True)
    checkpoint = {
        "run_id":          run_id,
        "completed_steps": completed_steps,
        "saved_at":        datetime.now().isoformat()
    }
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {}

# ============================================================
# MAIN PIPELINE RUNNER
# ============================================================
def run_pipeline(start_from: str = None,
                 data_path:  str = None,
                 rebuild_rag:bool = False) -> dict:
    """
    Runs full pipeline with checkpoints.

    start_from: step ID to resume from (e.g. "step3")
    data_path:  path to real HR data file
    rebuild_rag: force RAG rebuild even if policies unchanged
    """
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_start = datetime.now()

    print(f"\n{'='*60}")
    print(f"ATTRITION INTELLIGENCE PIPELINE")
    print(f"Run ID: {run_id}")
    print(f"Started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    if data_path:
        print(f"Data: {data_path}")
    if start_from:
        print(f"Resuming from: {start_from}")
    print(f"{'='*60}\n")

    results        = []
    completed      = []
    pipeline_failed = False
    failed_step    = None

    for step in PIPELINE_STEPS:
        step_id = step["id"]

        # Skip steps before start_from
        if start_from:
            step_ids = [s["id"] for s in PIPELINE_STEPS]
            start_idx = step_ids.index(start_from) \
                if start_from in step_ids else 0
            current_idx = step_ids.index(step_id)
            if current_idx < start_idx:
                print(f"⏭️  Skipping {step['name']} (before resume point)")
                continue

        # Skip RAG if not forced
        if step_id == "step8" and not rebuild_rag:
            rag_path = Path(cfg["paths"]["chroma_db"])
            if rag_path.exists():
                print(f"⏭️  Skipping RAG build (ChromaDB exists, use --rebuild-rag to force)")
                continue

        print(f"{'─'*60}")
        print(f"▶ {step['name']}")
        print(f"  {step['description']}")

        # Build extra args for mapper/eda if data_path provided
        extra_args = []
        if data_path and step_id in ["step1", "step2"]:
            extra_args = [data_path]

        # Run step
        result = run_step(step, extra_args)
        results.append(result)

        status   = "[OK] PASSED" if result["passed"] else "❌ FAILED"
        duration = result["duration"]
        print(f"  {status} ({duration}s)")

        # Print last few lines of output
        if result["stdout"]:
            last_lines = result["stdout"].strip().split("\n")[-5:]
            for line in last_lines:
                if line.strip():
                    print(f"  │ {line}")

        if result["passed"]:
            completed.append(step_id)
            save_checkpoint(completed, run_id)
        else:
            print(f"\n  Error output:")
            if result["stderr"]:
                for line in result["stderr"].strip().split("\n")[-10:]:
                    print(f"  │ {line}")

            if step["required"]:
                pipeline_failed = True
                failed_step     = step["name"]
                print(f"\n[FAIL] Required step failed — stopping pipeline")
                print(f"   Fix the error above and resume with:")
                print(f"   python run_pipeline.py --from {step_id}")
                break
            else:
                print(f"  [WARN]  Optional step failed — continuing pipeline")

    # Pipeline summary
    run_end      = datetime.now()
    total_seconds = (run_end - run_start).total_seconds()

    print(f"\n{'='*60}")
    if pipeline_failed:
        print(f"[FAIL] PIPELINE FAILED at: {failed_step}")
    else:
        print(f"[OK] PIPELINE COMPLETE")

    print(f"Duration:  {total_seconds:.0f}s ({total_seconds/60:.1f} min)")
    print(f"Steps completed: {len(completed)}/{len(PIPELINE_STEPS)}")

    if not pipeline_failed:
        print(f"\nNext steps:")
        print(f"  Start API:       uvicorn api.main:app --reload --port 8000")
        print(f"  Start Dashboard: streamlit run dashboard.py")

    # Save full run report
    report = {
        "run_id":           run_id,
        "started_at":       run_start.isoformat(),
        "completed_at":     run_end.isoformat(),
        "duration_seconds": round(total_seconds, 1),
        "pipeline_passed":  not pipeline_failed,
        "steps_completed":  len(completed),
        "steps_total":      len(PIPELINE_STEPS),
        "failed_step":      failed_step,
        "step_results":     results
    }

    Path("logs").mkdir(exist_ok=True)
    report_path = Path(f"logs/pipeline_run_{run_id}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Run report: {report_path}")
    print(f"{'='*60}\n")

    return report


# ============================================================
# CLI INTERFACE
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attrition Intelligence Pipeline Orchestrator"
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        default=None,
        help="Resume from step ID (e.g. step3)"
    )
    parser.add_argument(
        "--data",
        dest="data_path",
        default=None,
        help="Path to real HR data file (e.g. data/raw/jio_hr.xlsx)"
    )
    parser.add_argument(
        "--rebuild-rag",
        dest="rebuild_rag",
        action="store_true",
        default=False,
        help="Force RAG rebuild even if ChromaDB exists"
    )
    parser.add_argument(
        "--list-steps",
        dest="list_steps",
        action="store_true",
        default=False,
        help="List all pipeline steps and exit"
    )

    args = parser.parse_args()

    if args.list_steps:
        print("\nPipeline steps:")
        for step in PIPELINE_STEPS:
            req = "required" if step["required"] else "optional"
            print(f"  {step['id']}: {step['name']} ({req})")
            print(f"         {step['description']}")
        sys.exit(0)

    report = run_pipeline(
        start_from  = args.start_from,
        data_path   = args.data_path,
        rebuild_rag = args.rebuild_rag
    )

    sys.exit(0 if report["pipeline_passed"] else 1)