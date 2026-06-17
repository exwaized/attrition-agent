"""
Unit tests for api/main.py.

NOT covered here: /explain, /report/weekly, /report/department,
/score/batch, /audit/recent — these need real scored_employees.csv,
a real audit.db, and a real run_agent() call (which itself needs the
full LangGraph pipeline). Covered: resolve_tier_filter and
compute_budget_summary/simulate_budget (the extracted EV/ROI math,
pure given a dataframe + cost constants), plus a couple of TestClient
checks on /health since that endpoint only touches the filesystem.
"""
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, cfg, compute_budget_summary, resolve_tier_filter, simulate_budget

client = TestClient(app)


# ---------- resolve_tier_filter ----------

def test_resolve_tier_filter_critical():
    assert resolve_tier_filter("CRITICAL") == ["CRITICAL"]


def test_resolve_tier_filter_high_includes_critical():
    assert resolve_tier_filter("HIGH") == ["CRITICAL", "HIGH"]


def test_resolve_tier_filter_all_includes_medium():
    assert resolve_tier_filter("ALL") == ["CRITICAL", "HIGH", "MEDIUM"]


def test_resolve_tier_filter_case_insensitive():
    assert resolve_tier_filter("high") == ["CRITICAL", "HIGH"]


def test_resolve_tier_filter_unknown_falls_back_to_critical():
    # A typo'd query param shouldn't silently widen the scope to everyone
    assert resolve_tier_filter("typo") == ["CRITICAL"]


# ---------- compute_budget_summary ----------

def test_compute_budget_summary_basic_math():
    urgent = pd.DataFrame({"ev": [100000, 200000], "p_attrition": [0.6, 0.8]})
    result = compute_budget_summary(urgent, replacement_cost=450000)
    assert result["total_ev_urgent"] == 300000
    assert result["expected_replacements"] == 1.4
    assert result["replacement_cost_avoided"] == round(1.4 * 450000)


def test_compute_budget_summary_empty_df_does_not_divide_by_zero():
    urgent = pd.DataFrame({"ev": [], "p_attrition": []})
    result = compute_budget_summary(urgent, replacement_cost=450000)
    assert result["net_roi"] == 0.0  # max(total_ev, 1) guards the divide


# ---------- simulate_budget ----------

def test_simulate_budget_basic_math():
    subset = pd.DataFrame({"p_attrition": [0.8, 0.6]})
    result = simulate_budget(subset, intervention_cost=50000,
                              replacement_cost=450000, correction_type="CRITICAL")
    assert result["employees_affected"] == 2
    assert result["total_intervention_cost"] == 100000
    expected_retentions = round((0.8 + 0.6) * 0.75, 1)
    assert result["expected_retentions"] == expected_retentions


def test_simulate_budget_empty_subset_does_not_divide_by_zero():
    subset = pd.DataFrame({"p_attrition": []})
    result = simulate_budget(subset, intervention_cost=50000,
                              replacement_cost=450000, correction_type="ALL")
    assert result["employees_affected"] == 0
    assert result["net_roi"] == 0.0


# ---------- /health (TestClient) ----------

def test_health_endpoint_returns_degraded_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["champion_model"] is False


def test_health_endpoint_returns_healthy_when_all_files_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "champion_model.pkl").write_text("stub")
    (tmp_path / "models" / "cox_model.pkl").write_text("stub")
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    (tmp_path / "data" / "synthetic" / "scored_employees.csv").write_text("stub")
    (tmp_path / "chroma_db_dir").mkdir()
    monkeypatch.setitem(cfg["paths"], "chroma_db", str(tmp_path / "chroma_db_dir"))

    response = client.get("/health")
    body = response.json()
    assert body["status"] == "healthy"
    assert all(body["checks"].values())
