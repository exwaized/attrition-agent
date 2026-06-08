# ============================================================
# attrition_agent.py — LangGraph 4-Node Attrition Agent
# ============================================================
# PURPOSE: Orchestrates full pipeline from score to action
# FLOW: scored_employees → risk_scorer → shap_explainer →
#       intervention_generator → router → Slack/digest/log
# CONNECTED TO: ev_scoring.py (input) → api/main.py (called by)
# ============================================================
# ⚠️  OPUS 4.8 NEEDED: LangGraph StateGraph wiring + prompt
#     engineering for structured JSON LLM output is complex.
#     Use Opus if node chaining breaks or LLM output malformed.
# ============================================================

import json
import pickle
import sqlite3
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Optional
from groq import Groq

from langgraph.graph import StateGraph, END

# Import RAG retriever
import sys
sys.path.append(".")
from rag.retriever import retrieve_for_employee

# --- Step 1: Load all dependencies ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Load Groq client
# Groq for Colab (fast, free tier sufficient for 1800 employees in batches)
# Switch to Ollama for real Jio data (privacy)
import os
from dotenv import load_dotenv
load_dotenv()  # loads .env file from project root
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

feature_cols = Path("data/synthetic/feature_cols.txt").read_text().strip().split("\n")

# --- Step 2: Define LangGraph State ---
# State = shared memory across all nodes in the graph
# Each node reads from and writes to this state dict
# TypedDict enforces schema — prevents silent key errors
class AttritionState(TypedDict):
    employee_id:          str
    employee_data:        dict          # raw employee record
    p_attrition:          float         # from ev_scoring
    median_survival_months: float       # from Cox PH
    ev:                   float         # expected value
    risk_tier:            str           # CRITICAL/HIGH/MEDIUM/LOW
    shap_drivers:         list          # top 3 SHAP features
    policy_context:       str           # RAG retrieved chunks
    llm_recommendation:   str           # Ollama/Groq output
    routed_to:            str           # where alert was sent
    error:                Optional[str] # catches node failures

# ============================================================
# NODE 1: risk_scorer
# ============================================================
# Reads pre-computed scores from ev_scoring.py
# Could re-score in real-time for single employee lookups
# Batch scoring done offline for efficiency
# ============================================================
def risk_scorer(state: AttritionState) -> AttritionState:
    """
    Loads pre-computed risk scores for this employee.
    In production: could call model.predict_proba() in real-time.
    For batch: reads from scored_employees.csv (pre-computed).
    """
    try:
        scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
        emp_row   = scored_df[scored_df["employee_id"] == state["employee_id"]]

        if emp_row.empty:
            state["error"] = f"Employee {state['employee_id']} not found in scored data"
            return state

        emp = emp_row.iloc[0].to_dict()

        state["p_attrition"]           = float(emp["p_attrition"])
        state["median_survival_months"] = float(emp["median_survival_months"])
        state["ev"]                    = float(emp["ev"])
        state["risk_tier"]             = emp["risk_tier"]
        state["employee_data"]         = emp

    except Exception as e:
        state["error"] = f"risk_scorer error: {str(e)}"

    return state

# ============================================================
# NODE 2: shap_explainer
# ============================================================
# Parses pre-computed SHAP values for this employee
# Formats top 3 drivers for LLM context
# SHAP values already computed in ev_scoring.py
# ============================================================
def shap_explainer(state: AttritionState) -> AttritionState:
    """
    Extracts and formats top SHAP drivers for LLM prompt.
    Translates feature names into HR-readable descriptions.
    These become the 'why' in the intervention recommendation.
    """
    if state.get("error"):
        return state

    try:
        raw_drivers = json.loads(state["employee_data"]["top_shap_drivers"])

        # Translate feature names to HR-readable descriptions
        # LLM understands "high performer stuck" better than "interact_high_performer_no_promo"
        FEATURE_LABELS = {
            "interact_high_performer_no_promo":     "High performer with no promotion in 18+ months",
            "interact_underpaid_declining":          "Underpaid vs market AND activity declining",
            "interact_new_mgr_peer_left":            "New manager + peers recently left (contagion risk)",
            "interact_post_appraisal_dissatisfied":  "Received below-median hike in recent appraisal",
            "comp_compa_ratio":                      "Salary below market benchmark",
            "comp_underpaid_flag":                   "Significantly underpaid vs peers",
            "org_peer_attrition":                    "Multiple peers left team recently",
            "org_manager_attrition":                 "High manager turnover in team",
            "org_team_shrink":                       "Team headcount reduced >10%",
            "trend_login":                           "Login activity declining sharply",
            "trend_performance":                     "Performance rating declining",
            "career_tenure_at_band":                 "Stuck at same band for extended period",
            "recency_promotion":                     "Long gap since last promotion",
        }

        formatted_drivers = []
        for d in raw_drivers:
            label     = FEATURE_LABELS.get(d["feature"], d["feature"].replace("_", " "))
            direction = "RISK FACTOR" if d["shap"] > 0 else "PROTECTIVE"
            formatted_drivers.append({
                "feature":   d["feature"],
                "label":     label,
                "shap":      d["shap"],
                "direction": direction
            })

        state["shap_drivers"] = formatted_drivers

    except Exception as e:
        state["error"] = f"shap_explainer error: {str(e)}"

    return state

# ============================================================
# NODE 3: intervention_generator
# ============================================================
# RAG retrieves relevant policy context
# Groq/Ollama generates specific HR recommendation
# Output is structured JSON: narrative + actions + timeline
# ============================================================
# ⚠️ OPUS 4.8 NEEDED: Prompt engineering for consistent
#    structured JSON output with policy grounding is high-reasoning.
#    If Groq output is malformed JSON, switch to Opus via API.
# ============================================================
def intervention_generator(state: AttritionState) -> AttritionState:
    """
    Generates plain-English HR recommendation using:
    1. Employee context (band, tenure, circle, performance)
    2. SHAP drivers (why they're at risk)
    3. RAG policy context (what policy says to do)
    Output: specific, actionable, policy-grounded recommendation.
    """
    if state.get("error"):
        return state

    try:
        # Step 3a: RAG retrieval
        policy_context = retrieve_for_employee(state["employee_data"])
        state["policy_context"] = policy_context

        # Step 3b: Build LLM prompt
        emp   = state["employee_data"]
        drivers_text = "\n".join([
            f"- {d['label']} (SHAP impact: {d['shap']:+.3f})"
            for d in state["shap_drivers"]
        ])

        prompt = f"""You are an expert HR retention analyst at Jio Platforms.

EMPLOYEE PROFILE:
- Employee ID: {emp['employee_id']}
- Band: {emp['band']} | Department: {emp['department']} | Circle: {emp['circle']}
- Tenure: {emp['tenure_months']} months
- Performance Rating: {emp['perf_rating_current']}/5.0
- Compa-Ratio: {emp['compa_ratio']} (1.0 = market rate)
- Months since promotion: {emp['months_since_promotion']}
- Risk Profile: {emp['risk_profile']}

ATTRITION RISK SCORES:
- P(attrition): {state['p_attrition']:.3f} ({state['risk_tier']} risk)
- Predicted months remaining: {state['median_survival_months']:.1f}
- Expected Value of intervention: Rs {state['ev']:,.0f}

TOP RISK DRIVERS (from ML model):
{drivers_text}

RELEVANT HR POLICY:
{policy_context}

Generate a retention recommendation as a JSON object with exactly these fields:
{{
  "narrative": "2-3 sentence plain English explanation of why this employee is at risk",
  "immediate_actions": ["action 1", "action 2", "action 3"],
  "timeline": "specific timeline for each action",
  "policy_references": ["relevant policy section 1", "relevant policy section 2"],
  "intervention_window": "how many weeks before risk becomes irreversible"
}}

Be specific. Reference exact policy clauses. Give concrete rupee amounts where relevant.
Return ONLY the JSON object, no other text."""

        # Step 3c: Call Groq
        response = groq_client.chat.completions.create(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["llm"]["temperature"],
            max_tokens=600
        )

        raw_output = response.choices[0].message.content.strip()

        # Step 3d: Parse JSON output
        # Strip markdown fences if Groq adds them
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()

        recommendation = json.loads(raw_output)
        state["llm_recommendation"] = json.dumps(recommendation, indent=2)

    except json.JSONDecodeError as e:
        # If JSON parsing fails — store raw text, don't crash agent
        state["llm_recommendation"] = json.dumps({
            "narrative": raw_output,
            "immediate_actions": ["Review manually — LLM output parse error"],
            "timeline": "ASAP",
            "policy_references": [],
            "intervention_window": "Unknown"
        })
    except Exception as e:
        state["error"] = f"intervention_generator error: {str(e)}"

    return state

# ============================================================
# NODE 4: router
# ============================================================
# Conditional edge — routes based on risk_tier
# CRITICAL → Slack alert immediately
# HIGH     → weekly digest queue
# MEDIUM   → log only, manager awareness
# LOW      → audit log, no action
# ============================================================
def router(state: AttritionState) -> AttritionState:
    """
    Routes output based on risk tier.
    All decisions logged to SQLite audit trail — full auditability.
    Slack fires for CRITICAL only — prevents alert fatigue.
    """
    if state.get("error"):
        log_to_audit(state, "ERROR")
        state["routed_to"] = "ERROR_LOG"
        return state

    tier = state["risk_tier"]

    if tier == "CRITICAL":
        send_slack_alert(state)
        state["routed_to"] = "SLACK_ALERT"
    elif tier == "HIGH":
        add_to_weekly_digest(state)
        state["routed_to"] = "WEEKLY_DIGEST"
    elif tier == "MEDIUM":
        state["routed_to"] = "MANAGER_AWARENESS"
    else:
        state["routed_to"] = "AUDIT_LOG_ONLY"

    log_to_audit(state, tier)
    return state

# ============================================================
# HELPER: Slack alert
# ============================================================
def send_slack_alert(state: AttritionState):
    """
    Sends formatted Slack message for CRITICAL risk employees.
    If webhook not configured, prints to console (dev mode).
    """
    webhook_url = cfg["slack"]["webhook_url"]
    emp         = state["employee_data"]

    try:
        rec = json.loads(state["llm_recommendation"])
    except:
        rec = {"narrative": state["llm_recommendation"], "immediate_actions": []}

    message = {
        "text": f"🚨 *CRITICAL ATTRITION ALERT — {emp['employee_id']}*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{emp['employee_id']}* | Band {emp['band']} | "
                        f"{emp['department']} | {emp['circle']}\n"
                        f"P(attrition): *{state['p_attrition']:.2f}* | "
                        f"EV: *Rs {state['ev']:,.0f}* | "
                        f"Survival: *{state['median_survival_months']:.1f} months*\n\n"
                        f"*Why at risk:*\n" +
                        "\n".join([f"• {d['label']}" for d in state["shap_drivers"]]) +
                        f"\n\n*Recommendation:*\n{rec.get('narrative', 'See full report')}\n\n"
                        f"*Immediate actions:*\n" +
                        "\n".join([f"{i+1}. {a}" for i, a in enumerate(rec.get('immediate_actions', []))])
                    )
                }
            }
        ]
    }

    if webhook_url:
        import urllib.request
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(message).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)
        print(f"✅ Slack alert sent for {emp['employee_id']}")
    else:
        # Dev mode — print to console
        print(f"\n{'='*50}")
        print(f"[SLACK ALERT — DEV MODE] {emp['employee_id']}")
        print(f"P(attrition): {state['p_attrition']:.3f} | EV: Rs {state['ev']:,.0f}")
        print(f"Narrative: {rec.get('narrative', 'N/A')}")
        print(f"Actions: {rec.get('immediate_actions', [])}")
        print(f"{'='*50}\n")

# ============================================================
# HELPER: Weekly digest queue
# ============================================================
def add_to_weekly_digest(state: AttritionState):
    """Appends HIGH risk employees to weekly digest JSON file."""
    digest_path = Path("logs/weekly_digest.json")

    try:
        existing = json.loads(digest_path.read_text()) if digest_path.exists() else []
    except:
        existing = []

    existing.append({
        "employee_id":   state["employee_id"],
        "risk_tier":     state["risk_tier"],
        "p_attrition":   state["p_attrition"],
        "ev":            state["ev"],
        "added_at":      datetime.now().isoformat(),
        "recommendation": state["llm_recommendation"]
    })

    digest_path.write_text(json.dumps(existing, indent=2))

# ============================================================
# HELPER: SQLite audit log
# ============================================================
def log_to_audit(state: AttritionState, tier: str):
    """
    Logs every agent run to SQLite — full auditability.
    Every score, recommendation, routing decision timestamped.
    Required for HR compliance — who was alerted, when, why.
    """
    db_path = cfg["paths"]["audit_db"]
    Path("logs").mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id     TEXT,
            p_attrition     REAL,
            ev              REAL,
            risk_tier       TEXT,
            routed_to       TEXT,
            recommendation  TEXT,
            error           TEXT,
            created_at      TEXT
        )
    """)
    conn.execute("""
        INSERT INTO audit_log
        (employee_id, p_attrition, ev, risk_tier, routed_to, recommendation, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        state["employee_id"],
        state.get("p_attrition", 0),
        state.get("ev", 0),
        tier,
        state.get("routed_to", ""),
        state.get("llm_recommendation", ""),
        state.get("error", ""),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

# ============================================================
# Step 3: Build LangGraph StateGraph
# ============================================================
# Nodes connected in sequence with conditional edge at router
# State flows through each node — each adds its output to state
# ============================================================
def build_agent():
    """
    Builds and compiles the LangGraph StateGraph.
    Returns compiled graph ready for .invoke() calls.
    """
    graph = StateGraph(AttritionState)

    # Add all 4 nodes
    graph.add_node("risk_scorer",           risk_scorer)
    graph.add_node("shap_explainer",        shap_explainer)
    graph.add_node("intervention_generator",intervention_generator)
    graph.add_node("router",                router)

    # Wire nodes sequentially
    # State passes through each node in order
    graph.set_entry_point("risk_scorer")
    graph.add_edge("risk_scorer",            "shap_explainer")
    graph.add_edge("shap_explainer",         "intervention_generator")
    graph.add_edge("intervention_generator", "router")
    graph.add_edge("router",                 END)

    return graph.compile()

# Compile once at module level — reused across all API calls
agent = build_agent()

# ============================================================
# Step 4: Run agent on one employee
# ============================================================
def run_agent(employee_id: str) -> dict:
    """
    Runs full 4-node pipeline for one employee.
    Returns final state with all scores + recommendation.
    """
    initial_state = AttritionState(
        employee_id=employee_id,
        employee_data={},
        p_attrition=0.0,
        median_survival_months=0.0,
        ev=0.0,
        risk_tier="LOW",
        shap_drivers=[],
        policy_context="",
        llm_recommendation="",
        routed_to="",
        error=None
    )

    final_state = agent.invoke(initial_state)
    return final_state

# ============================================================
# Step 5: Batch run — process all employees
# ============================================================
def run_batch(limit: int = None, tiers: list = None) -> list:
    """
    Runs agent on multiple employees.
    limit: max employees to process (None = all)
    tiers: filter by risk tier e.g. ["CRITICAL", "HIGH"]
    Returns list of final states.
    """
    scored_df = pd.read_csv("data/synthetic/scored_employees.csv")

    if tiers:
        scored_df = scored_df[scored_df["risk_tier"].isin(tiers)]

    if limit:
        scored_df = scored_df.head(limit)

    print(f"Running agent on {len(scored_df)} employees...")
    results = []

    for i, row in scored_df.iterrows():
        emp_id = row["employee_id"]
        try:
            result = run_agent(emp_id)
            results.append(result)
            tier = result.get("risk_tier", "?")
            routed = result.get("routed_to", "?")
            print(f"  [{i+1}/{len(scored_df)}] {emp_id} | {tier} | → {routed}")
        except Exception as e:
            print(f"  [{i+1}] {emp_id} | ERROR: {str(e)}")

    return results


# --- Quick test when run directly ---
if __name__ == "__main__":
    print("Testing agent on top 3 CRITICAL risk employees...")
    scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
    critical  = scored_df[scored_df["risk_tier"] == "CRITICAL"].head(3)

    for _, row in critical.iterrows():
        print(f"\nRunning agent for {row['employee_id']}...")
        result = run_agent(row["employee_id"])

        print(f"Risk tier:       {result['risk_tier']}")
        print(f"P(attrition):    {result['p_attrition']:.3f}")
        print(f"Survival months: {result['median_survival_months']:.1f}")
        print(f"EV:              Rs {result['ev']:,.0f}")
        print(f"Routed to:       {result['routed_to']}")

        if result.get("llm_recommendation"):
            rec = json.loads(result["llm_recommendation"])
            print(f"Narrative: {rec.get('narrative', 'N/A')}")
            print(f"Actions:   {rec.get('immediate_actions', [])}")

        if result.get("error"):
            print(f"ERROR: {result['error']}")
