"""
Unit tests for run_pipeline.py.

NOT covered: run_pipeline() itself end-to-end — it really does call
every other pipeline script as a real subprocess, which is what
running the actual pipeline is for, not a unit test. Covered:
save_checkpoint/load_checkpoint (real file I/O, but small and fast,
worth round-tripping), run_step's pass/fail/timeout mapping (with
subprocess.run mocked out — we're testing run_step's own logic, not
re-testing subprocess itself), and the start_from validation added
in this round (an unrecognized --from value used to silently fall
back to running the whole pipeline from scratch instead of erroring).
"""
import subprocess

import pytest

import run_pipeline as rp

# ---------- save_checkpoint / load_checkpoint ----------

def test_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    rp.save_checkpoint(["step1", "step2"], "run_123")
    result = rp.load_checkpoint()
    assert result["run_id"] == "run_123"
    assert result["completed_steps"] == ["step1", "step2"]


def test_load_checkpoint_returns_empty_dict_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "CHECKPOINT_PATH", tmp_path / "does_not_exist.json")
    assert rp.load_checkpoint() == {}


def test_save_checkpoint_creates_logs_dir_if_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rp, "CHECKPOINT_PATH", tmp_path / "logs" / "checkpoint.json")
    rp.save_checkpoint([], "run_456")
    assert (tmp_path / "logs" / "checkpoint.json").exists()


# ---------- run_step (subprocess.run mocked) ----------

class _FakeCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_step_marks_passed_on_zero_returncode(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(0, stdout="all good")
    )
    step = {"id": "step1", "name": "Mapper", "script": "fake.py", "required": True}
    result = rp.run_step(step)
    assert result["passed"] is True
    assert result["returncode"] == 0


def test_run_step_marks_failed_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(1, stderr="boom")
    )
    step = {"id": "step1", "name": "Mapper", "script": "fake.py", "required": True}
    result = rp.run_step(step)
    assert result["passed"] is False
    assert result["stderr"] == "boom"


def test_run_step_truncates_long_stdout_to_last_2000_chars(monkeypatch):
    long_output = "x" * 5000
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(0, stdout=long_output)
    )
    step = {"id": "step1", "name": "Mapper", "script": "fake.py", "required": True}
    result = rp.run_step(step)
    assert len(result["stdout"]) == 2000


def test_run_step_handles_timeout(monkeypatch):
    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="fake.py", timeout=600)
    monkeypatch.setattr(subprocess, "run", raise_timeout)
    step = {"id": "step1", "name": "Mapper", "script": "fake.py", "required": True}
    result = rp.run_step(step)
    assert result["passed"] is False
    assert "TIMEOUT" in result["stderr"]


def test_run_step_passes_extra_args_into_command(monkeypatch):
    captured_cmd = {}
    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return _FakeCompletedProcess(0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    step = {"id": "step1", "name": "Mapper", "script": "fake.py", "required": True}
    rp.run_step(step, extra_args=["data/raw/real.xlsx"])
    assert "data/raw/real.xlsx" in captured_cmd["cmd"]


# ---------- start_from validation ----------

def test_run_pipeline_rejects_unknown_start_from(monkeypatch, tmp_path):
    # Don't actually run any real steps — just check validation fires
    # before any subprocess gets launched
    monkeypatch.setattr(rp, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    with pytest.raises(ValueError, match="Unknown step"):
        rp.run_pipeline(start_from="step99")
