"""
Smoke tests — confirm every pipeline module at least imports cleanly.

These won't catch logic bugs (that's what test_config.py / test_features.py
and the deeper tests we'll add once I see train.py / ev_scoring.py /
validate.py / api/main.py are for), but they catch the single most
embarrassing failure mode: a module that breaks on a fresh clone
(missing dependency, leftover debug `print()`, a relative import that
only worked because of how you happened to run it before) — right
before you'd otherwise discover it live in an interview demo.

Note: data/, data/raw/, data/synthetic/, models/, agents/, rag/, and
api/ don't need __init__.py files for this to work — Python 3.3+
treats them as implicit namespace packages as long as the repo root
is on sys.path (conftest.py handles that). If you DO get a
ModuleNotFoundError here, the fix is almost always either (a) the
folder name doesn't match what's listed below, or (b) the module has
a `python data/raw/mapper.py`-style relative import that breaks when
imported as a package instead of run as a script.
"""
import importlib

import pytest

MODULES = [
    "data.synthetic.generate",
    "data.raw.mapper",
    "data.raw.eda",
    "data.raw.validate",
    "data.synthetic.feature_engineering",
    "models.multicollinearity_check",
    "models.train",
    "models.ev_scoring",
    "agents.attrition_agent",
    "rag.retriever",
    "rag.build_rag",
    "api.main",
    "run_pipeline",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)