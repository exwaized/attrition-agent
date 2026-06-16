"""
Shared pytest fixtures for the attrition-agent test suite.

ASSUMPTIONS — verify these against your actual repo and adjust if wrong:
  - config.yaml lives at the repo root.
  - data/synthetic/feature_cols.txt has one feature name per line.
  - You're running pytest from the repo root (e.g. `pytest` or
    `pytest tests/`), not from inside tests/.
"""
import os
import sys
from pathlib import Path

import pytest
import yaml

# Step 0 — agents/attrition_agent.py constructs a Groq client at import
# time, which needs SOME value for GROQ_API_KEY even if you're not
# actually calling the LLM in a given test. setdefault() only fills
# this in if it's genuinely unset — your real .env / exported key (if
# present) always wins.
os.environ.setdefault("GROQ_API_KEY", "test-placeholder-key")

# Step 1 — make the repo root importable. This lets test files do
# `from models.train import ...` or `from api.main import app` no
# matter what directory pytest was actually invoked from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repo root — handy for any test that needs
    to find a file (config.yaml, feature_cols.txt, scored_employees.csv)."""
    return ROOT


@pytest.fixture(scope="session")
def config(repo_root) -> dict:
    """Loads config.yaml once per test session instead of once per test
    — config.yaml shouldn't change mid-run, so re-reading it every time
    just slows things down."""
    config_path = repo_root / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def feature_cols(repo_root) -> list[str]:
    """Loads the canonical 48-feature list that train.py, ev_scoring.py,
    and attrition_agent.py all rely on for column alignment."""
    cols_path = repo_root / "data" / "synthetic" / "feature_cols.txt"
    with open(cols_path, "r") as f:
        return [line.strip() for line in f if line.strip()]