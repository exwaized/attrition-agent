# ============================================================
# retriever.py — RAG Retrieval Interface for Agent
# ============================================================
# PURPOSE: Given employee risk drivers, retrieves relevant
#          policy chunks from ChromaDB for LLM context
# FLOW: shap_drivers → query → chroma → policy chunks
# CONNECTED TO: build_rag.py (input) → attrition_agent.py (output)
# ============================================================

import chromadb
from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path

# --- Step 1: Load config and connect to ChromaDB ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Reuse same persistent client and collection built in build_rag.py
# If collection doesn't exist, agent will catch the error cleanly
chroma_path = cfg["paths"]["chroma_db"]
client      = chromadb.PersistentClient(path=chroma_path)
collection  = client.get_collection("hr_policies")

# Same embedding model as build_rag.py
# CRITICAL: must be identical model — different model = different vector space = bad retrieval
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 2: Feature → query mapping ---
# Maps SHAP feature names to human-readable search queries
# This is the bridge between ML output and NLP retrieval
# Each feature name maps to the policy concept it relates to
FEATURE_TO_QUERY = {
    "interact_high_performer_no_promo": "high performer no promotion 18 months escalation",
    "interact_underpaid_declining":     "salary below market compa ratio MAP correction underpaid",
    "interact_new_mgr_peer_left":       "peer attrition contagion team members leaving new manager",
    "interact_post_appraisal_dissatisfied": "post appraisal resignation retention hike below median",
    "comp_compa_ratio":                 "salary market alignment MAP correction compa ratio",
    "comp_underpaid_flag":              "underpaid salary correction market alignment",
    "org_peer_attrition":               "peer attrition team contagion stabilisation",
    "org_team_shrink":                  "team size reduction health check morale",
    "trend_login":                      "login activity declining engagement review",
    "volatility_leave":                 "excessive leave unplanned attendance",
    "career_tenure_at_band":            "promotion criteria band tenure career progression",
    "recency_promotion":                "promotion overdue career stagnation",
}

def retrieve_policy_context(top_shap_drivers: list, n_results: int = 2) -> str:
    """
    Given top SHAP drivers from XGBoost explanation,
    retrieves relevant policy chunks from ChromaDB.

    Flow:
    1. Map each SHAP feature name to a search query
    2. Embed the query using sentence-transformers
    3. Query ChromaDB for top matching policy chunks
    4. Deduplicate and return as formatted context string

    Returns: formatted string of policy chunks for LLM prompt
    """
    retrieved_chunks = []
    seen_ids = set()  # deduplicate across multiple queries

    for driver in top_shap_drivers:
        feature_name = driver["feature"]
        shap_value   = driver["shap"]

        # Only retrieve for features pushing toward leaving (positive SHAP)
        # Negative SHAP = protective factor — no policy action needed
        if shap_value <= 0:
            continue

        # Map feature to query — fall back to feature name if not mapped
        query = FEATURE_TO_QUERY.get(feature_name, feature_name.replace("_", " "))

        # Embed query and search ChromaDB
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # Collect unique chunks with source attribution
        for doc, meta, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0]
        ):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                retrieved_chunks.append({
                    "source":  meta["source"],
                    "content": doc,
                    "feature": feature_name
                })

    # Format as context string for LLM prompt
    if not retrieved_chunks:
        return "No specific policy found for these risk drivers."

    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Policy: {chunk['source']} | Relevant to: {chunk['feature']}]\n{chunk['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_for_employee(employee_row: dict) -> str:
    """
    Convenience wrapper — takes scored employee dict,
    parses SHAP drivers JSON, returns policy context.
    Used directly by LangGraph agent node.
    """
    import json
    try:
        shap_drivers = json.loads(employee_row["top_shap_drivers"])
        return retrieve_policy_context(shap_drivers)
    except Exception as e:
        return f"Policy retrieval error: {str(e)}"


# --- Step 3: Test when run directly ---
if __name__ == "__main__":
    test_drivers = [
        {"feature": "interact_high_performer_no_promo", "shap": 0.31},
        {"feature": "comp_compa_ratio",                 "shap": 0.24},
        {"feature": "org_peer_attrition",               "shap": 0.18},
    ]

    print("Testing retriever with sample SHAP drivers...")
    context = retrieve_policy_context(test_drivers)
    print("\nRetrieved policy context:")
    print(context)
