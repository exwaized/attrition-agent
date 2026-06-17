"""
Unit tests for rag/retriever.py.

NOT covered here: retrieve_policy_context and retrieve_for_employee —
both need a real ChromaDB collection built by build_rag.py plus a
loaded sentence-transformers model, neither of which a unit test
should depend on. Covered: select_shap_queries and
format_policy_context, the two pure pieces extracted from
retrieve_policy_context.
"""
from rag.retriever import FEATURE_TO_QUERY, format_policy_context, select_shap_queries

# ---------- select_shap_queries ----------

def test_select_shap_queries_skips_negative_shap():
    # Negative SHAP = protective factor, not a risk to retrieve policy for
    drivers = [{"feature": "comp_compa_ratio", "shap": -0.2}]
    assert select_shap_queries(drivers) == []


def test_select_shap_queries_maps_known_feature():
    drivers = [{"feature": "comp_compa_ratio", "shap": 0.3}]
    result = select_shap_queries(drivers)
    assert result == [("comp_compa_ratio", FEATURE_TO_QUERY["comp_compa_ratio"])]


def test_select_shap_queries_falls_back_for_unmapped_feature():
    drivers = [{"feature": "some_unmapped_feature", "shap": 0.1}]
    result = select_shap_queries(drivers)
    assert result == [("some_unmapped_feature", "some unmapped feature")]


def test_select_shap_queries_preserves_order_and_handles_mixed_signs():
    drivers = [
        {"feature": "comp_compa_ratio", "shap": 0.3},
        {"feature": "trend_login", "shap": -0.5},
        {"feature": "org_peer_attrition", "shap": 0.2},
    ]
    result = select_shap_queries(drivers)
    assert [f for f, _ in result] == ["comp_compa_ratio", "org_peer_attrition"]


# ---------- format_policy_context ----------

def test_format_policy_context_empty_list():
    assert format_policy_context([]) == "No specific policy found for these risk drivers."


def test_format_policy_context_single_chunk():
    chunks = [{"source": "MAP_Policy.pdf", "content": "Salary correction details.", "feature": "comp_compa_ratio"}]
    result = format_policy_context(chunks)
    assert "[Policy: MAP_Policy.pdf | Relevant to: comp_compa_ratio]" in result
    assert "Salary correction details." in result


def test_format_policy_context_joins_multiple_chunks_with_separator():
    chunks = [
        {"source": "A.pdf", "content": "First.", "feature": "f1"},
        {"source": "B.pdf", "content": "Second.", "feature": "f2"},
    ]
    result = format_policy_context(chunks)
    assert "\n\n---\n\n" in result
    assert "First." in result and "Second." in result
