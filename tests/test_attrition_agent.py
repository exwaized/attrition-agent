"""
Unit tests for agents/attrition_agent.py.

NOT covered here: risk_scorer (needs scored_employees.csv),
intervention_generator (needs a real/mocked Groq call + RAG retriever),
build_agent/run_agent/run_batch (full LangGraph execution, needs the
whole pipeline's real data). Those are exercised by actually running
the agent end-to-end, not by isolated unit tests — same reasoning as
the model-fitting code in train.py.

Covered: shap_explainer (pure given a well-formed state),
parse_llm_json_output and build_slack_message (extracted pure
helpers), router's dispatch logic (mocking out the three
side-effecting helpers it calls), and log_to_audit/add_to_weekly_digest
(real file/DB I/O against tmp_path, not your actual logs/ folder).

Requires GROQ_API_KEY to be set to *something* before this module can
even be imported (module-level Groq client construction) — conftest.py
should set a placeholder if it isn't already present in the environment.
"""
import json
import sqlite3

import pytest

from agents.attrition_agent import (
    FEATURE_LABELS,
    add_to_weekly_digest,
    build_slack_message,
    log_to_audit,
    parse_llm_json_output,
    router,
    shap_explainer,
)

# ---------- shap_explainer ----------

def test_shap_explainer_maps_known_feature_to_label():
    state = {
        "employee_data": {
            "top_shap_drivers": json.dumps([
                {"feature": "comp_compa_ratio", "shap": 0.45}
            ])
        }
    }
    result = shap_explainer(state)
    driver = result["shap_drivers"][0]
    assert driver["label"] == FEATURE_LABELS["comp_compa_ratio"]
    assert driver["direction"] == "RISK FACTOR"


def test_shap_explainer_unknown_feature_falls_back_to_underscore_replace():
    state = {
        "employee_data": {
            "top_shap_drivers": json.dumps([
                {"feature": "some_new_feature_not_in_map", "shap": -0.2}
            ])
        }
    }
    result = shap_explainer(state)
    driver = result["shap_drivers"][0]
    assert driver["label"] == "some new feature not in map"
    assert driver["direction"] == "PROTECTIVE"


def test_shap_explainer_skips_when_error_already_set():
    # Nodes short-circuit once an earlier node has set an error — this
    # is what lets a single bad employee fail without crashing the
    # whole graph run.
    state = {"error": "risk_scorer error: something broke"}
    result = shap_explainer(state)
    assert "shap_drivers" not in result
    assert result["error"] == "risk_scorer error: something broke"


def test_shap_explainer_catches_malformed_json_as_error():
    state = {"employee_data": {"top_shap_drivers": "not valid json"}}
    result = shap_explainer(state)
    assert "shap_explainer error" in result["error"]


# ---------- parse_llm_json_output ----------

def test_parse_llm_json_output_handles_json_fence():
    raw = '```json\n{"narrative": "test"}\n```'
    assert parse_llm_json_output(raw) == {"narrative": "test"}


def test_parse_llm_json_output_handles_plain_fence():
    raw = '```\n{"narrative": "test"}\n```'
    assert parse_llm_json_output(raw) == {"narrative": "test"}


def test_parse_llm_json_output_handles_no_fence():
    raw = '{"narrative": "test"}'
    assert parse_llm_json_output(raw) == {"narrative": "test"}


def test_parse_llm_json_output_raises_on_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json_output("not json at all")


# ---------- build_slack_message ----------

def test_build_slack_message_includes_employee_and_drivers():
    state = {
        "employee_data": {"employee_id": "E123", "band": "B3", "department": "Tech", "circle": "Mumbai"},
        "p_attrition": 0.82,
        "ev": 275000,
        "shap_drivers": [{"label": "Underpaid vs market"}],
    }
    rec = {"narrative": "At risk due to comp gap.", "immediate_actions": ["Give a raise"]}

    message = build_slack_message(state, rec)

    assert "E123" in message["text"]
    block_text = message["blocks"][0]["text"]["text"]
    assert "Underpaid vs market" in block_text
    assert "At risk due to comp gap." in block_text
    assert "1. Give a raise" in block_text


def test_build_slack_message_handles_missing_narrative_gracefully():
    state = {
        "employee_data": {"employee_id": "E1", "band": "B1", "department": "D", "circle": "C"},
        "p_attrition": 0.9,
        "ev": 100000,
        "shap_drivers": [],
    }
    message = build_slack_message(state, {})  # empty rec — no narrative, no actions
    block_text = message["blocks"][0]["text"]["text"]
    assert "See full report" in block_text


# ---------- router ----------

def test_router_critical_calls_slack_and_sets_routed_to(monkeypatch):
    calls = []
    monkeypatch.setattr("agents.attrition_agent.send_slack_alert", lambda state: calls.append("slack"))
    monkeypatch.setattr("agents.attrition_agent.log_to_audit", lambda state, tier: calls.append(f"audit:{tier}"))

    state = {"risk_tier": "CRITICAL"}
    result = router(state)

    assert result["routed_to"] == "SLACK_ALERT"
    assert "slack" in calls
    assert "audit:CRITICAL" in calls


def test_router_high_calls_weekly_digest(monkeypatch):
    calls = []
    monkeypatch.setattr("agents.attrition_agent.add_to_weekly_digest", lambda state: calls.append("digest"))
    monkeypatch.setattr("agents.attrition_agent.log_to_audit", lambda state, tier: calls.append(f"audit:{tier}"))

    state = {"risk_tier": "HIGH"}
    result = router(state)

    assert result["routed_to"] == "WEEKLY_DIGEST"
    assert "digest" in calls


def test_router_medium_no_side_effects_beyond_audit(monkeypatch):
    calls = []
    monkeypatch.setattr("agents.attrition_agent.send_slack_alert", lambda state: calls.append("slack"))
    monkeypatch.setattr("agents.attrition_agent.add_to_weekly_digest", lambda state: calls.append("digest"))
    monkeypatch.setattr("agents.attrition_agent.log_to_audit", lambda state, tier: calls.append(f"audit:{tier}"))

    state = {"risk_tier": "MEDIUM"}
    result = router(state)

    assert result["routed_to"] == "MANAGER_AWARENESS"
    assert calls == ["audit:MEDIUM"]  # neither slack nor digest fired


def test_router_low_routes_to_audit_log_only(monkeypatch):
    monkeypatch.setattr("agents.attrition_agent.log_to_audit", lambda state, tier: None)
    state = {"risk_tier": "LOW"}
    result = router(state)
    assert result["routed_to"] == "AUDIT_LOG_ONLY"


def test_router_error_state_skips_tier_logic_entirely(monkeypatch):
    calls = []
    monkeypatch.setattr("agents.attrition_agent.send_slack_alert", lambda state: calls.append("slack"))
    monkeypatch.setattr("agents.attrition_agent.log_to_audit", lambda state, tier: calls.append(f"audit:{tier}"))

    state = {"error": "risk_scorer error: not found", "risk_tier": "CRITICAL"}
    result = router(state)

    # Error short-circuits BEFORE the tier is even checked — a CRITICAL
    # tier never reaches send_slack_alert if an earlier node failed.
    assert result["routed_to"] == "ERROR_LOG"
    assert "slack" not in calls
    assert "audit:ERROR" in calls


# ---------- log_to_audit ----------

def test_log_to_audit_writes_row(tmp_path):
    db_path = str(tmp_path / "audit.db")
    state = {
        "employee_id": "E999",
        "p_attrition": 0.91,
        "ev": 300000,
        "routed_to": "SLACK_ALERT",
        "llm_recommendation": "{}",
        "error": None,
    }
    log_to_audit(state, "CRITICAL", db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT employee_id, risk_tier FROM audit_log").fetchone()
    conn.close()

    assert row == ("E999", "CRITICAL")


def test_log_to_audit_creates_table_if_missing(tmp_path):
    db_path = str(tmp_path / "fresh.db")
    state = {"employee_id": "E1", "p_attrition": 0.1, "ev": 0, "routed_to": "", "llm_recommendation": "", "error": None}
    # Should not raise even though fresh.db has no audit_log table yet
    log_to_audit(state, "LOW", db_path=db_path)
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    conn.close()
    assert count == 1


# ---------- add_to_weekly_digest ----------

def test_add_to_weekly_digest_creates_file_when_missing(tmp_path):
    digest_path = tmp_path / "digest.json"
    state = {
        "employee_id": "E1", "risk_tier": "HIGH", "p_attrition": 0.6,
        "ev": 80000, "llm_recommendation": "{}",
    }
    add_to_weekly_digest(state, digest_path=digest_path)

    saved = json.loads(digest_path.read_text())
    assert len(saved) == 1
    assert saved[0]["employee_id"] == "E1"


def test_add_to_weekly_digest_appends_to_existing(tmp_path):
    digest_path = tmp_path / "digest.json"
    digest_path.write_text(json.dumps([{"employee_id": "OLD"}]))

    state = {
        "employee_id": "NEW", "risk_tier": "HIGH", "p_attrition": 0.6,
        "ev": 80000, "llm_recommendation": "{}",
    }
    add_to_weekly_digest(state, digest_path=digest_path)

    saved = json.loads(digest_path.read_text())
    assert len(saved) == 2
    assert saved[0]["employee_id"] == "OLD"
    assert saved[1]["employee_id"] == "NEW"


def test_add_to_weekly_digest_recovers_from_corrupted_file(tmp_path):
    digest_path = tmp_path / "digest.json"
    digest_path.write_text("not valid json{{{")

    state = {
        "employee_id": "E1", "risk_tier": "HIGH", "p_attrition": 0.6,
        "ev": 80000, "llm_recommendation": "{}",
    }
    add_to_weekly_digest(state, digest_path=digest_path)  # should not raise

    saved = json.loads(digest_path.read_text())
    assert len(saved) == 1  # corrupted content discarded, started fresh
