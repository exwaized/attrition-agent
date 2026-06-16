import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
import yaml
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph

load_dotenv(override=True)

sys.path.append(".")
from rag.retriever import retrieve_for_employee  # noqa: E402, I001 — needs sys.path.append above to resolve

# --- Step 1: Load all dependencies ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# --- Step 2: Define LangGraph State ---
# State = shared memory across all nodes
# Each node reads from and writes to this state dict
class AttritionState(TypedDict):
    employee_id:            str
    employee_data:          dict
    p_attrition:            float
    median_survival_months: float
    ev:                     float
    risk_tier:              str
    shap_drivers:           list
    policy_context:         str
    llm_recommendation:     str
    routed_to:              str
    error:                  Optional[str]


# Feature-name -> HR-readable label mapping, used by shap_explainer.
# Pulled out to module level (was rebuilt on every single call before) —
# also makes the mapping itself independently inspectable/testable.
FEATURE_LABELS = {
    "interact_high_performer_no_promo":    "High performer with no promotion in 18+ months",
    "interact_underpaid_declining":         "Underpaid vs market AND activity declining",
    "interact_new_mgr_peer_left":           "New manager + peers recently left (contagion risk)",
    "interact_post_appraisal_dissatisfied": "Received below-median hike in recent appraisal",
    "comp_compa_ratio":                     "Salary below market benchmark",
    "comp_underpaid_flag":                  "Significantly underpaid vs peers",
    "org_peer_attrition":                   "Multiple peers left team recently",
    "org_manager_attrition":                "High manager turnover in team",
    "org_team_shrink":                      "Team headcount reduced >10%",
    "trend_login":                          "Login activity declining sharply",
    "trend_performance":                    "Performance rating declining",
    "career_tenure_at_band":                "Stuck at same band for extended period",
    "recency_promotion":                    "Long gap since last promotion",
    "recency_hike":                         "Long gap since last salary hike",
    "recency_manager_change":               "Recent manager change — destabilisation risk",
    "volatility_leave":                     "Unusual leave pattern vs tenure",
    "volatility_performance":               "Erratic performance ratings",
}


# ============================================================
# Pure, testable functions
# ============================================================

def parse_llm_json_output(raw_output: str) -> dict:
    """
    Groq sometimes wraps its JSON response in markdown fences despite
    being told not to. Strips ```json / ``` fences if present, then
    parses. Raises json.JSONDecodeError on genuinely malformed output —
    intervention_generator's except block is what catches that and
    falls back to a manual-review placeholder.
    """
    if "```json" in raw_output:
        raw_output = raw_output.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_output:
        raw_output = raw_output.split("```")[1].split("```")[0].strip()
    return json.loads(raw_output)


def build_slack_message(state: "AttritionState", rec: dict) -> dict:
    """
    Builds the Slack Block Kit payload for a CRITICAL alert. Pure
    given state + the parsed LLM recommendation dict — the only side
    effect (the actual HTTP POST) stays in send_slack_alert.
    """
    emp = state["employee_data"]
    return {
        "text": f"CRITICAL ATTRITION ALERT — {emp['employee_id']}",
        "blocks": [{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{emp['employee_id']}* | Band {emp['band']} | "
                    f"{emp['department']} | {emp['circle']}\n"
                    f"P(attrition): *{state['p_attrition']:.2f}* | "
                    f"EV: *Rs {state['ev']:,.0f}*\n\n"
                    f"*Why at risk:*\n" +
                    "\n".join([f"• {d['label']}" for d in state["shap_drivers"]]) +
                    f"\n\n*Recommendation:*\n{rec.get('narrative', 'See full report')}\n\n"
                    f"*Immediate actions:*\n" +
                    "\n".join([f"{i+1}. {a}" for i, a in
                               enumerate(rec.get('immediate_actions', []))])
                )
            }
        }]
    }


# ============================================================
# NODE 1: risk_scorer
# ============================================================
# Loads pre-computed scores from ev_scoring.py
# Reads scored_employees.csv — batch scored offline
# ============================================================
def risk_scorer(state: AttritionState) -> AttritionState:
    try:
        scored_df = pd.read_csv("data/synthetic/scored_employees.csv")
        emp_row   = scored_df[scored_df["employee_id"] == state["employee_id"]]

        if emp_row.empty:
            state["error"] = f"Employee {state['employee_id']} not found"
            return state

        emp = emp_row.iloc[0].to_dict()
        state["p_attrition"]            = float(emp["p_attrition"])
        state["median_survival_months"] = float(emp["median_survival_months"])
        state["ev"]                     = float(emp["ev"])
        state["risk_tier"]              = emp["risk_tier"]
        state["employee_data"]          = emp

    except Exception as e:
        state["error"] = f"risk_scorer error: {str(e)}"

    return state


# ============================================================
# NODE 2: shap_explainer
# ============================================================
# Parses pre-computed SHAP values for this employee
# Translates feature names into HR-readable descriptions
# These become the 'why' in the LLM recommendation
# ============================================================
def shap_explainer(state: AttritionState) -> AttritionState:
    if state.get("error"):
        return state

    try:
        raw_drivers = json.loads(state["employee_data"]["top_shap_drivers"])

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
# Groq generates specific HR recommendation
# Updated prompt fixes survival months misinterpretation
# ============================================================
def intervention_generator(state: AttritionState) -> AttritionState:
    if state.get("error"):
        return state

    raw_output = ""
    try:
        # Step 3a: RAG retrieval
        policy_context         = retrieve_for_employee(state["employee_data"])
        state["policy_context"] = policy_context

        # Step 3b: Build LLM prompt
        emp          = state["employee_data"]
        drivers_text = "\n".join([
            f"- {d['label']} (SHAP impact: {d['shap']:+.3f}) — {d['direction']}"
            for d in state["shap_drivers"]
        ])

        prompt = f"""You are an expert HR retention analyst at Jio Platforms.
Your job is to generate a precise, actionable retention recommendation
based on ML model outputs. Read the following carefully before generating.

IMPORTANT — HOW TO INTERPRET MODEL OUTPUTS:
- P(attrition): probability employee leaves within 90 days based on
  current behavioural signals. >0.70 = urgent. This is your PRIMARY
  urgency signal.
- Cox PH median survival months: population-level estimate of how long
  employees with SIMILAR HISTORICAL PROFILES typically stay. This is NOT
  a countdown timer. A high survival months with high P(attrition) means
  the employee is deviating from their historical pattern — that deviation
  IS the risk signal.
- EV (Expected Value): rupee value of intervening. Higher = more budget
  justified.
- SHAP drivers: specific features driving THIS employee's risk score.
  Positive SHAP = pushing toward leaving. These are your intervention targets.

EMPLOYEE PROFILE:
- Employee ID: {emp['employee_id']}
- Band: {emp['band']} | Department: {emp['department']} | Circle: {emp['circle']}
- Tenure: {emp['tenure_months']} months
- Performance Rating: {emp['perf_rating_current']}/5.0
- Compa-Ratio: {emp['compa_ratio']} (1.0 = market rate, <0.85 = significantly underpaid)
- Months since promotion: {emp['months_since_promotion']}
- Risk Profile: {emp['risk_profile']}

ATTRITION RISK SCORES:
- P(attrition): {state['p_attrition']:.3f} — {'URGENT action needed' if state['p_attrition'] > 0.70 else 'Monitor closely'}
- Cox PH survival estimate: {state['median_survival_months']:.1f} months
  (population median for similar profiles — use only to contextualise urgency,
   NOT as remaining time. Never say X months remaining before exit.)
- Expected Value of intervention: Rs {state['ev']:,.0f}
- Risk tier: {state['risk_tier']}

TOP RISK DRIVERS (from ML model — these are your intervention targets):
{drivers_text}

RELEVANT HR POLICY:
{policy_context}

INSTRUCTIONS FOR YOUR RESPONSE:
1. Lead with the specific behavioural reason this employee is at risk
   based on SHAP drivers — not generic statements
2. Reference their actual numbers (compa-ratio, rating, tenure, months
   since promotion) in your narrative
3. Recommend specific policy-grounded actions with rupee amounts where relevant
4. If P(attrition) is high but survival months is also high — acknowledge
   this means current behaviour is deviating from their historical pattern
5. Calibrate urgency to P(attrition):
   P>0.85 = immediate this week
   P=0.65-0.85 = action within 2 weeks
   P<0.65 = action this month
6. NEVER say "X months remaining before exit" — survival months is not a countdown

Generate a retention recommendation as a JSON object with exactly these fields:
{{
  "narrative": "2-3 sentences specific to THIS employee's actual numbers and drivers",
  "immediate_actions": ["specific action 1 with rupee amount if relevant",
                        "specific action 2 with timeline",
                        "specific action 3"],
  "timeline": "specific timeline for each action based on urgency tier",
  "policy_references": ["exact policy section name 1", "exact policy section name 2"],
  "intervention_window": "specific number of weeks based on P(attrition) level"
}}

Be specific. Use the employee's actual numbers. Reference exact policy clauses.
Return ONLY the JSON object, no other text, no markdown fences."""

        # Step 3c: Call Groq
        response = groq_client.chat.completions.create(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["llm"]["temperature"],
            max_tokens=600
        )

        raw_output = response.choices[0].message.content.strip()

        # Step 3d: Parse JSON — strip markdown fences if present
        recommendation             = parse_llm_json_output(raw_output)
        state["llm_recommendation"] = json.dumps(recommendation, indent=2)

    except json.JSONDecodeError:
        # Store raw text if JSON parsing fails — don't crash agent
        state["llm_recommendation"] = json.dumps({
            "narrative":        raw_output,
            "immediate_actions":["Review manually — LLM output parse error"],
            "timeline":         "ASAP",
            "policy_references":[],
            "intervention_window": "Unknown"
        })
    except Exception as e:
        state["error"] = f"intervention_generator error: {str(e)}"

    return state


# ============================================================
# NODE 4: router
# ============================================================
# Routes based on risk_tier
# CRITICAL → Slack, HIGH → digest, MEDIUM → log, LOW → log
# All decisions logged to SQLite audit trail
# ============================================================
def router(state: AttritionState) -> AttritionState:
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
def send_slack_alert(state: AttritionState, webhook_url: Optional[str] = None):
    if webhook_url is None:
        webhook_url = cfg["slack"]["webhook_url"]

    emp = state["employee_data"]

    try:
        rec = json.loads(state["llm_recommendation"])
    except Exception:
        rec = {"narrative": state["llm_recommendation"], "immediate_actions": []}

    message = build_slack_message(state, rec)

    if webhook_url:
        import urllib.request
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(message).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)
        print(f"Slack alert sent for {emp['employee_id']}")
    else:
        print(f"\n{'='*50}")
        print(f"[SLACK ALERT — DEV MODE] {emp['employee_id']}")
        print(f"P(attrition): {state['p_attrition']:.3f} | EV: Rs {state['ev']:,.0f}")
        print(f"Narrative: {rec.get('narrative', 'N/A')}")
        print(f"Actions: {rec.get('immediate_actions', [])}")
        print(f"{'='*50}\n")


# ============================================================
# HELPER: Weekly digest
# ============================================================
def add_to_weekly_digest(state: AttritionState, digest_path: Optional[Path] = None):
    if digest_path is None:
        digest_path = Path("logs/weekly_digest.json")

    try:
        existing = json.loads(digest_path.read_text()) if digest_path.exists() else []
    except Exception:
        existing = []

    existing.append({
        "employee_id":    state["employee_id"],
        "risk_tier":      state["risk_tier"],
        "p_attrition":    state["p_attrition"],
        "ev":             state["ev"],
        "added_at":       datetime.now().isoformat(),
        "recommendation": state["llm_recommendation"]
    })
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(json.dumps(existing, indent=2))


# ============================================================
# HELPER: SQLite audit log
# ============================================================
def log_to_audit(state: AttritionState, tier: str, db_path: Optional[str] = None):
    if db_path is None:
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
        (employee_id, p_attrition, ev, risk_tier, routed_to,
         recommendation, error, created_at)
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
# Build LangGraph StateGraph
# ============================================================
def build_agent():
    graph = StateGraph(AttritionState)

    graph.add_node("risk_scorer",            risk_scorer)
    graph.add_node("shap_explainer",         shap_explainer)
    graph.add_node("intervention_generator", intervention_generator)
    graph.add_node("router",                 router)

    graph.set_entry_point("risk_scorer")
    graph.add_edge("risk_scorer",            "shap_explainer")
    graph.add_edge("shap_explainer",         "intervention_generator")
    graph.add_edge("intervention_generator", "router")
    graph.add_edge("router",                 END)

    return graph.compile()


agent = build_agent()


# ============================================================
# Run agent on one employee
# ============================================================
def run_agent(employee_id: str) -> dict:
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
    return agent.invoke(initial_state)


# ============================================================
# Batch run
# ============================================================
def run_batch(limit: int = None, tiers: list = None) -> list:
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
            print(f"  [{i+1}] {emp_id} | {result.get('risk_tier')} | → {result.get('routed_to')}")
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