import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from harness import Evaluator, Rubric, State, feedback, run_loop, validate_model, read_video, MCPBackend


def texture():
    rng = np.random.default_rng(4)
    a = np.clip(120 + rng.normal(0, 8, (96, 128)), 0, 255).astype(np.uint8)
    cv2.rectangle(a, (20, 20), (55, 60), 180, 2)
    return a


def grade(frames, **kwargs):
    return Evaluator(Rubric(**kwargs)).evaluate(iter(frames), np.ones(frames[0].shape, bool))


def test_smooth_white_blob_is_rejected():
    a = texture()
    a[20:70, 30:95] = 255
    assert not grade([a] * 3)["specular"]["passed"]


def test_small_glint_is_allowed():
    a = texture()
    a[40:42, 40:42] = 255
    assert grade([a] * 3)["specular"]["passed"]


def test_static_grain_passes_without_temporal_noise():
    r = grade([texture()] * 4)
    assert r["passed"]
    assert r["luminance"]["temporal_std"] == 0


def test_smooth_wall_fails_texture():
    assert not grade([np.full((96, 128), 120, np.uint8)] * 3)["luminance"]["passed"]


def test_flicker_fails_even_when_it_returns_to_first_frame():
    a = texture()
    r = grade([a, a + 15, a])
    assert not r["temporal"]["passed"]


def test_movement_that_returns_is_detected():
    a = texture()
    assert not grade([a, np.roll(a, 5, axis=1), a])["temporal"]["passed"]


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="mask"):
        Evaluator().evaluate(iter([texture()] * 2), np.zeros((96, 128), bool))


def test_single_frame_rejected():
    with pytest.raises(ValueError, match="two"):
        grade([texture()])


def test_feedback_updates_both_controls_and_bounds_strength():
    s = State(strength=0.02, seed=4)
    feedback(s, {"specular": {"passed": False}, "luminance": {"passed": False}, "temporal": {"passed": True}})
    assert s.strength == 0
    assert s.seed == 5
    assert "physical drywall occlusion" in s.modifiers


def test_drift_feedback_preserves_camera_instruction():
    s = State()
    feedback(s, {"specular": {"passed": True}, "luminance": {"passed": True}, "temporal": {"passed": False}})
    assert "No geometry deformation" in s.motion_prompt


def test_invalid_strength_rejected():
    with pytest.raises(ValueError):
        State(strength=0.36)


def test_schema_rejects_unsupported_parameter():
    with pytest.raises(ValueError, match="strength"):
        validate_model({"id": "gpt_image_2_5", "parameters": []}, {"strength": 0.35})


def test_schema_rejects_resolution_without_silent_fallback():
    model = {"id": "seedance_2_5", "parameters": [{"name": "resolution", "options": ["1080p"], "type": "string"}]}
    with pytest.raises(ValueError, match="1080p_pro"):
        validate_model(model, {"resolution": "1080p_pro"})


class FakeBackend:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def preflight(self, state):
        if self.fail:
            raise ValueError("unsupported strength")

    async def iteration(self, state, directory):
        self.calls.append((state.strength, state.seed, state.modifiers))
        return directory / "video.mp4"


def failed_grade(_):
    return {"passed": False, "specular": {"passed": False}, "luminance": {"passed": False}, "temporal": {"passed": False}}


def test_loop_stops_at_five_and_alerts(tmp_path):
    backend = FakeBackend()
    code = asyncio.run(run_loop(backend, State(), tmp_path, failed_grade))
    assert code == 2
    assert len(backend.calls) == 5
    assert backend.calls[-1][:2] == (0.15, 46)
    assert json.loads((tmp_path / "alert.json").read_text())["reason"] == "validation_exhausted"
    assert len((tmp_path / "iterations.jsonl").read_text().splitlines()) == 5


def test_loop_stops_on_pass(tmp_path):
    b = FakeBackend()
    assert asyncio.run(run_loop(b, State(), tmp_path, lambda _: {"passed": True})) == 0
    assert len(b.calls) == 1


def test_preflight_failure_never_generates_and_alerts(tmp_path):
    b = FakeBackend(fail=True)
    with pytest.raises(ValueError):
        asyncio.run(run_loop(b, State(), tmp_path, failed_grade))
    assert not b.calls
    assert (tmp_path / "alert.json").exists()


def test_no_unbounded_loop(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(run_loop(FakeBackend(), State(), tmp_path, failed_grade, max_loops=6))


def test_real_video_decode_includes_last_frame(tmp_path):
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR))
    writer.release()
    frames = list(read_video(path))
    assert len(frames) == 4
    assert grade(frames)["passed"]


class Session:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def call_tool(self, name, arguments):
        self.calls += 1
        from types import SimpleNamespace
        return SimpleNamespace(isError=False, structuredContent=self.result, content=[])


def test_adjusted_generation_is_rejected_without_retry(tmp_path):
    session = Session({"adjustments": {"resolution": {"used": "720p"}}, "results": [{"id": "existing-job"}]})
    b = MCPBackend(session, {}, tmp_path)
    with pytest.raises(RuntimeError, match="adjusted"):
        asyncio.run(b.generate("video", {"model": "seedance_2_5"}))
    assert session.calls == 1
    assert "existing-job" in (tmp_path / "jobs.jsonl").read_text()


def test_terminal_failure_does_not_poll_forever(tmp_path):
    s = Session({"jobs": [{"job_id": "x", "status": "failed"}]})
    b = MCPBackend(s, {}, tmp_path)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(b.wait("x"))
    assert s.calls == 1


def test_cli_never_overwrites_an_existing_run(tmp_path, monkeypatch):
    import harness
    existing = tmp_path / "run"
    existing.mkdir()
    alert = existing / "alert.json"
    alert.write_text('{"reason":"original"}')
    config = tmp_path / "config.json"
    config.write_text('{}')
    monkeypatch.setattr("sys.argv", ["harness.py", "preflight", "--config", str(config), "--run-dir", str(existing)])
    assert harness.main() == 1
    assert json.loads(alert.read_text()) == {"reason": "original"}


def test_random_temporal_grain_cannot_pass():
    a = texture()
    frames = [a, np.roll(a, 20, axis=0), a]
    assert not grade(frames)["temporal"]["passed"]


def test_budget_stops_before_submission(tmp_path):
    class Backend(MCPBackend):
        async def call(self, name, arguments):
            assert arguments["params"]["get_cost"] is True
            return {"cost": {"credits_exact": 1}}

    b = Backend(None, {"max_credits": 0}, tmp_path)
    b.catalog = {"soul_cinematic": {"id": "soul_cinematic", "aspect_ratios": ["16:9"],
                                     "parameters": [{"name": "seed", "type": "number"}]}}
    with pytest.raises(RuntimeError, match="budget"):
        asyncio.run(b.iteration(State(), tmp_path))
    assert not (tmp_path / "jobs.jsonl").exists()


def test_wait_timeout_retains_job_identity(tmp_path):
    s = Session({})
    b = MCPBackend(s, {"job_timeout_seconds": 0}, tmp_path)
    with pytest.raises(TimeoutError, match="retained-job"):
        asyncio.run(b.wait("retained-job"))
    assert s.calls == 0
