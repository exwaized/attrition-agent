"""
Tests for the 48-feature contract (data/synthetic/feature_cols.txt).

Why this exists: train.py, ev_scoring.py, and attrition_agent.py all
treat feature_cols.txt as the single source of truth for column
alignment. If it drifts from what feature_engineering.py actually
outputs — or if multicollinearity_check.py's drop logic regresses —
you get a silent KeyError or a re-introduced VIF problem deep inside
ev_scoring.py instead of a clear failure here.
"""

EXPECTED_FEATURE_COUNT = 48

# The 7 features multicollinearity_check.py should have dropped
# (VIF / Pearson r=1.0 collinearity, or redundant/reference-dummy) —
# per Feature_Engineering_Reference.docx. If any of these reappear in
# feature_cols.txt, the drop step has regressed.
SHOULD_BE_DROPPED = {
    "recency_promotion",
    "recency_hike",
    "career_band_normalized",
    "comp_underpaid_flag",
    "onboard_days_to_project",
    "dept_HR",
    "circle_Gujarat",
}


def test_feature_count(feature_cols):
    assert len(feature_cols) == EXPECTED_FEATURE_COUNT, (
        f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_cols)}. "
        "Did feature_engineering.py or multicollinearity_check.py change?"
    )


def test_no_collinear_features_leaked_back_in(feature_cols):
    leaked = SHOULD_BE_DROPPED & set(feature_cols)
    assert not leaked, f"Dropped collinear features reappeared in feature_cols.txt: {leaked}"


def test_no_duplicate_features(feature_cols):
    assert len(feature_cols) == len(set(feature_cols)), "Duplicate feature names found"


def test_no_empty_feature_names(feature_cols):
    assert all(col for col in feature_cols), "Found an empty/whitespace-only feature name"
