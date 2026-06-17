import json

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

# --- Step 1: Load config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Lazy-loaded singletons. The ORIGINAL version connected to ChromaDB and
# loaded the sentence-transformers model at import time — meaning just
# `import rag.retriever` would fail with no usable error message on any
# machine where build_rag.py hasn't been run yet (fresh clone, CI runner,
# a new contributor's laptop), since get_collection() raises if the
# "hr_policies" collection doesn't exist. Loading lazily means the module
# always imports cleanly; the actual ChromaDB/model dependency only
# becomes a hard requirement when retrieve_policy_context is genuinely
# called, with an error that's traceable to that specific call instead
# of a mysterious import-time crash.
_client = None
_collection = None
_embedder = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"])
        # Reuses the same persistent client/collection built in build_rag.py
        _collection = _client.get_collection("hr_policies")
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Same embedding model as build_rag.py — CRITICAL that this stays
        # identical. A different model means a different vector space,
        # which means retrieval silently returns garbage, not an error.
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# --- Step 2: Feature -> query mapping ---
# Maps SHAP feature names to human-readable search queries — the bridge
# between ML output and NLP retrieval. Each feature name maps to the
# policy concept it relates to.
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


# ============================================================
# Pure, testable functions
# ============================================================

def select_shap_queries(top_shap_drivers: list) -> list:
    """
    Filters SHAP drivers down to the ones worth retrieving policy for,
    and maps each to a search query. Only positive-SHAP drivers
    (pushing toward leaving) get a query — negative SHAP is a
    protective factor, no policy action needed for it. Unmapped
    features fall back to their name with underscores replaced.
    Returns a list of (feature_name, query) tuples.
    """
    queries = []
    for driver in top_shap_drivers:
        feature_name = driver["feature"]
        shap_value   = driver["shap"]

        if shap_value <= 0:
            continue

        query = FEATURE_TO_QUERY.get(feature_name, feature_name.replace("_", " "))
        queries.append((feature_name, query))
    return queries


def format_policy_context(retrieved_chunks: list) -> str:
    """
    Formats retrieved policy chunks into the context string injected
    into the LLM prompt. Pure given the chunk list — the actual
    ChromaDB query stays in retrieve_policy_context.
    """
    if not retrieved_chunks:
        return "No specific policy found for these risk drivers."

    context_parts = [
        f"[Policy: {chunk['source']} | Relevant to: {chunk['feature']}]\n{chunk['content']}"
        for chunk in retrieved_chunks
    ]
    return "\n\n---\n\n".join(context_parts)


# ============================================================
# Retrieval (needs the real ChromaDB collection + embedder)
# ============================================================

def retrieve_policy_context(top_shap_drivers: list, n_results: int = 2) -> str:
    """
    Given top SHAP drivers from XGBoost explanation, retrieves
    relevant policy chunks from ChromaDB.

    Flow:
    1. select_shap_queries — map each SHAP feature to a search query
    2. Embed each query using sentence-transformers
    3. Query ChromaDB for top matching policy chunks
    4. format_policy_context — dedupe and format as context string
    """
    collection = _get_collection()
    embedder   = _get_embedder()

    retrieved_chunks = []
    seen_ids = set()  # deduplicate across multiple queries

    for feature_name, query in select_shap_queries(top_shap_drivers):
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

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

    return format_policy_context(retrieved_chunks)


def retrieve_for_employee(employee_row: dict) -> str:
    """
    Convenience wrapper — takes a scored employee dict, parses the
    SHAP drivers JSON, returns policy context. Used directly by the
    LangGraph agent node.
    """
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