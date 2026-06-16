"""
Tests for config.yaml — built directly against the real file, not a
guess. Each test encodes an actual design decision from the build so a
future edit (or IDE autocomplete) can't silently revert it:
  - aucpr (not the default ROC-AUC) because the attrition label is imbalanced
  - decision threshold deliberately tuned below 0.5 to bias toward recall
  - llm.provider must be one of the two clients attrition_agent.py actually supports
  - ev_framework's rupee thresholds have to stay internally consistent
    or the EV math in ev_scoring.py stops making sense
"""

REQUIRED_TOP_LEVEL_SECTIONS = {"project", "data", "model", "llm", "ev_framework", "slack", "paths"}


def test_config_has_required_sections(config):
    missing = REQUIRED_TOP_LEVEL_SECTIONS - config.keys()
    assert not missing, f"config.yaml is missing top-level sections: {missing}"


# ---------- data ----------

def test_data_section(config):
    data = config["data"]
    assert data["n_employees"] > 0
    assert 0 < data["attrition_rate"] < 1, (
        f"attrition_rate should be a fraction, got {data['attrition_rate']}"
    )
    assert isinstance(data["random_seed"], int)
    assert data["output_path"]  # non-empty string


# ---------- model ----------

def test_xgboost_uses_pr_auc_not_roc_auc(config):
    eval_metric = config["model"]["xgboost"].get("eval_metric")
    assert eval_metric == "aucpr", (
        f"eval_metric is {eval_metric!r} — must stay 'aucpr', ROC-AUC is "
        "the wrong metric for this class imbalance"
    )


def test_xgboost_hyperparams_are_sane(config):
    xgb = config["model"]["xgboost"]
    assert xgb["n_estimators"] > 0
    assert xgb["max_depth"] > 0
    assert 0 < xgb["learning_rate"] < 1


def test_cox_ph_penalizer_is_non_negative(config):
    assert config["model"]["cox_ph"]["penalizer"] >= 0


def test_decision_threshold_favors_recall(config):
    threshold = config["model"]["threshold"]
    assert 0 < threshold < 1
    # Deliberately tuned below 0.5 to catch more true attrition cases —
    # if this drifts back to 0.5, the recall tradeoff you made is gone.
    assert threshold < 0.5, f"threshold is {threshold} — was tuned below 0.5 for recall"


# ---------- llm ----------

def test_llm_provider_is_supported(config):
    provider = config["llm"]["provider"]
    assert provider in {"groq", "ollama"}, (
        f"llm.provider is {provider!r} — attrition_agent.py's client-init "
        "switch only handles 'groq' and 'ollama'"
    )


def test_llm_temperature_is_low_for_structured_output(config):
    temp = config["llm"]["temperature"]
    assert 0 <= temp <= 1
    assert temp <= 0.5, "temperature this high risks inconsistent structured JSON output"


# ---------- ev_framework ----------

def test_ev_framework_values_present_and_non_negative(config):
    ev = config["ev_framework"]
    for key in ("replacement_cost", "intervention_cost", "high_risk_threshold", "medium_risk_threshold"):
        assert key in ev, f"ev_framework missing '{key}'"
        assert ev[key] >= 0


def test_ev_framework_thresholds_are_internally_consistent(config):
    ev = config["ev_framework"]
    assert ev["medium_risk_threshold"] <= ev["high_risk_threshold"], (
        "a 'medium' alert threshold above the 'high' threshold would never fire"
    )
    assert ev["intervention_cost"] < ev["replacement_cost"], (
        "intervention must be cheaper than replacement, or ev_scoring.py "
        "never recommends intervening"
    )


# ---------- paths ----------

def test_paths_section_present(config):
    paths = config["paths"]
    for key in ("models", "logs", "audit_db", "chroma_db"):
        assert key in paths and paths[key]


def test_audit_db_lives_inside_logs_dir(config):
    paths = config["paths"]
    assert paths["audit_db"].startswith(paths["logs"]), (
        f"audit_db ({paths['audit_db']}) should live under logs ({paths['logs']})"
    )
