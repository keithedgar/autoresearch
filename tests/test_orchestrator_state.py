from pathlib import Path

import orchestrator


def test_state_round_trip(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(orchestrator, "STATE_FILE", str(state_file))

    state = orchestrator._empty_state()
    state["tag"] = "apr"
    state["total_experiments"] = 7

    orchestrator.save_state(state)
    loaded = orchestrator.load_state()

    assert loaded["tag"] == "apr"
    assert loaded["total_experiments"] == 7


def test_load_state_handles_invalid_json(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text("{invalid")
    monkeypatch.setattr(orchestrator, "STATE_FILE", str(state_file))

    loaded = orchestrator.load_state()

    assert loaded["total_experiments"] == 0
    assert loaded["runs"] == []
