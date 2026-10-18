import asyncio
import json

import cv2
import numpy as np
import pytest

from asset_loop import (PilotConfig, composite, plan_run, run_pilot, screen_candidate,
                        FalQueue, PilotError, write_image, read_image, fingerprint)


def fixtures(tmp_path):
    source = np.full((128, 192, 3), 120, np.uint8)
    cv2.rectangle(source, (50, 20), (130, 110), (190, 190, 190), 2)
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[40:90, 70:110] = 255
    write_image(tmp_path / "source.png", source)
    write_image(tmp_path / "mask.png", mask)
    return PilotConfig(source=str(tmp_path / "source.png"), mask=str(tmp_path / "mask.png"),
                       prompt="Matte painted cabinet, fine realistic surface variation",
                       crop=[32, 0, 128, 128], max_usd=1.0)


def test_plan_keeps_seed_and_only_varies_strength(tmp_path):
    config = fixtures(tmp_path)
    plan = plan_run(config)
    assert [p["strength"] for p in plan["candidates"]] == [0.15, 0.2, 0.25, 0.3, 0.35]
    assert {p["seed"] for p in plan["candidates"]} == {42}
    assert plan["estimated_total_usd"] == pytest.approx(0.175)
    assert "image_url" not in json.dumps(plan)


@pytest.mark.parametrize("strengths", [[0.4], [0.1] * 6, [], [float("nan")]])
def test_invalid_sweep_rejected(strengths):
    with pytest.raises(ValueError):
        PilotConfig(source="s", mask="m", prompt="p", strengths=strengths)


def test_empty_mask_rejected_before_any_network(tmp_path):
    c = fixtures(tmp_path)
    write_image(tmp_path / "mask.png", np.zeros((128, 192), np.uint8))
    with pytest.raises(ValueError, match="mask"):
        plan_run(c)


def test_mask_must_be_inside_crop(tmp_path):
    c = fixtures(tmp_path)
    c.crop = [0, 0, 32, 32]
    with pytest.raises(ValueError, match="crop"):
        plan_run(c)


def test_budget_checked_before_submission(tmp_path):
    c = fixtures(tmp_path)
    c.max_usd = 0.01
    with pytest.raises(ValueError, match="budget"):
        plan_run(c)


def test_composite_preserves_every_unmasked_pixel():
    source = np.full((40, 60, 3), 100, np.uint8)
    mask = np.zeros((40, 60), np.uint8)
    mask[15:25, 25:35] = 255
    edit = np.full((40, 40, 3), 220, np.uint8)
    result = composite(source, edit, mask, [10, 0, 40, 40], feather_px=3)
    assert np.array_equal(result[mask == 0], source[mask == 0])
    assert (result[20, 30] == 220).all()


def test_composite_rejects_silent_resize():
    with pytest.raises(ValueError, match="dimensions"):
        composite(np.zeros((40, 60, 3), np.uint8), np.zeros((20, 20, 3), np.uint8),
                  np.ones((40, 60), np.uint8) * 255, [10, 0, 40, 40])


def test_shifted_structure_fails_screening(tmp_path):
    c = fixtures(tmp_path)
    a = read_image(c.source)
    b = np.roll(a, 10, axis=1)
    mask = read_image(c.mask, gray=True)
    report = screen_candidate(a, b, mask)
    assert not report["structure"]["passed"]


class FakeFal:
    def __init__(self, fail_submit=False):
        self.submissions = []
        self.waits = []
        self.fail_submit = fail_submit

    async def submit(self, payload):
        self.submissions.append(payload)
        if self.fail_submit:
            raise TimeoutError("lost response")
        return {"request_id": str(len(self.submissions)),
                "status_url": "https://queue.fal.run/status", "response_url": "https://queue.fal.run/result"}

    async def result(self, receipt):
        self.waits.append(receipt["request_id"])
        return {"seed": 42, "images": [{"url": "https://fal.media/test.png"}], "has_nsfw_concepts": [False]}

    async def image(self, url):
        return np.full((128, 128, 3), 125, np.uint8)


def test_all_five_saved_with_fixed_seed_and_no_auto_approval(tmp_path):
    c = fixtures(tmp_path)
    b = FakeFal()
    run = tmp_path / "run"
    result = asyncio.run(run_pilot(c, run, b))
    assert result["status"] == "awaiting_review"
    assert len(b.submissions) == 5
    assert len({p["seed"] for p in b.submissions}) == 1
    assert (run / "contact-sheet.png").exists()
    assert len(list(run.glob("candidate-*/composite.png"))) == 5
    assert not (run / "approved.png").exists()


def test_unknown_submission_is_never_replayed(tmp_path):
    c = fixtures(tmp_path)
    run = tmp_path / "run"
    b = FakeFal(fail_submit=True)
    with pytest.raises(TimeoutError):
        asyncio.run(run_pilot(c, run, b))
    assert len(b.submissions) == 1
    with pytest.raises(PilotError, match="unknown"):
        asyncio.run(run_pilot(c, run, b, resume=True))
    assert len(b.submissions) == 1
    assert (run / "alert.json").exists()


def test_cached_resume_has_zero_new_submissions(tmp_path):
    c = fixtures(tmp_path)
    run = tmp_path / "run"
    b = FakeFal()
    asyncio.run(run_pilot(c, run, b))
    second = FakeFal()
    asyncio.run(run_pilot(c, run, second, resume=True))
    assert not second.submissions and not second.waits


def test_resume_rejects_changed_settings(tmp_path):
    c = fixtures(tmp_path)
    run = tmp_path / "run"
    asyncio.run(run_pilot(c, run, FakeFal()))
    c.seed += 1
    with pytest.raises(PilotError, match="changed"):
        asyncio.run(run_pilot(c, run, FakeFal(), resume=True))


def test_incorrect_returned_seed_stops_run(tmp_path):
    class WrongSeed(FakeFal):
        async def result(self, receipt):
            return {"seed": 999, "images": [{"url": "https://fal.media/test.png"}]}
    with pytest.raises(PilotError, match="seed"):
        asyncio.run(run_pilot(fixtures(tmp_path), tmp_path / "run", WrongSeed()))


def test_queue_errors_are_terminal_and_do_not_resubmit():
    import httpx
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"status": "COMPLETED", "error": "generation failed"})
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            backend = FalQueue("test-secret", client=client, poll_interval=0)
            with pytest.raises(PilotError, match="failed"):
                await backend.result({"status_url": "https://queue.fal.run/status", "response_url": "https://queue.fal.run/result"})
    asyncio.run(execute())
    assert len(calls) == 1


def test_queue_never_sends_credential_to_another_host():
    import httpx
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("network should not run"))) as client:
            b = FalQueue("test-secret", client=client)
            with pytest.raises(PilotError, match="host"):
                await b.result({"status_url": "https://example.com/status", "response_url": "https://queue.fal.run/result"})
    asyncio.run(execute())


def test_download_has_no_authorization_header():
    import httpx
    ok, encoded = cv2.imencode('.png', np.zeros((32, 32, 3), np.uint8))
    assert ok
    def handle(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, content=encoded.tobytes())
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            b = FalQueue("test-secret", client=client)
            assert (await b.image("https://fal.media/test.png")).shape == (32, 32, 3)
    asyncio.run(execute())


def test_offline_replay_rebuilds_composites_without_backend(tmp_path):
    c = fixtures(tmp_path)
    run = tmp_path / "run"
    asyncio.run(run_pilot(c, run, FakeFal()))
    first = read_image(run / "candidate-01" / "composite.png")
    (run / "candidate-01" / "composite.png").unlink()
    asyncio.run(run_pilot(c, run, None, resume=True))
    assert np.array_equal(first, read_image(run / "candidate-01" / "composite.png"))


def test_offline_replay_of_incomplete_run_has_actionable_error(tmp_path):
    from asset_loop import prepare
    c = fixtures(tmp_path)
    run = tmp_path / "run"
    prepare(c, run)
    with pytest.raises(PilotError, match="cached"):
        asyncio.run(run_pilot(c, run, None, resume=True))
    assert not (run / "candidate-01" / "intent.json").exists()


def test_resume_recovers_existing_receipt_without_resubmitting(tmp_path):
    class Interrupted(FakeFal):
        async def result(self, receipt):
            raise TimeoutError("interrupted while waiting")
    c = fixtures(tmp_path)
    run = tmp_path / "run"
    first = Interrupted()
    with pytest.raises(TimeoutError):
        asyncio.run(run_pilot(c, run, first))
    second = FakeFal()
    asyncio.run(run_pilot(c, run, second, resume=True))
    assert len(first.submissions) == 1
    assert len(second.submissions) == 4
    assert len(second.waits) == 5


def test_queue_post_is_sent_once_on_transport_loss():
    import httpx
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response")
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            backend = FalQueue("test-secret", client=client)
            with pytest.raises(httpx.ReadTimeout):
                await backend.submit({"prompt": "test"})
    asyncio.run(execute())
    assert len(calls) == 1


def test_keychain_preferred_without_printing_secret(monkeypatch, capsys):
    import asset_loop
    import subprocess
    monkeypatch.setattr(asset_loop.sys, 'platform', 'darwin')
    calls = []
    def lookup(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, 'fixture-key\n', '')
    monkeypatch.setattr(subprocess, 'run', lookup)
    monkeypatch.setenv('FAL_KEY', 'stale-fixture')
    assert asset_loop.resolve_fal_key() == 'fixture-key'
    args, kwargs = calls[0]
    assert args[:7] == ['/usr/bin/security', 'find-generic-password', '-s', 'soft-cactus.fal', '-a', 'FAL_KEY', '-w']
    assert kwargs['capture_output'] is True
    assert not kwargs.get('shell', False)
    assert 'fixture-key' not in capsys.readouterr().out


def test_keychain_denial_does_not_silently_fallback(monkeypatch):
    import asset_loop
    import subprocess
    monkeypatch.setattr(asset_loop.sys, 'platform', 'darwin')
    monkeypatch.setenv('FAL_KEY', 'stale-fixture')
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: subprocess.CompletedProcess(args, 36, '', 'denied'))
    with pytest.raises(PilotError, match='Keychain'):
        asset_loop.resolve_fal_key()


def test_missing_keychain_item_allows_environment_fallback(monkeypatch):
    import asset_loop
    import subprocess
    monkeypatch.setattr(asset_loop.sys, 'platform', 'darwin')
    monkeypatch.setenv('FAL_KEY', 'environment-fixture')
    monkeypatch.setattr(asset_loop, 'load_dotenv', lambda *a, **kw: False)
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: subprocess.CompletedProcess(args, 44, '', 'missing'))
    assert asset_loop.resolve_fal_key() == 'environment-fixture'


def test_keychain_timeout_is_actionable_and_does_not_expose_output(monkeypatch):
    import asset_loop
    import subprocess
    monkeypatch.setattr(asset_loop.sys, 'platform', 'darwin')
    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 20, output='sensitive-fixture')
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(PilotError, match='timed out') as error:
        asset_loop.resolve_fal_key()
    assert 'sensitive-fixture' not in str(error.value)


def test_non_mac_environment_does_not_invoke_keychain(monkeypatch):
    import asset_loop
    import subprocess
    monkeypatch.setattr(asset_loop.sys, 'platform', 'linux')
    monkeypatch.setenv('FAL_KEY', 'linux-fixture')
    monkeypatch.setattr(asset_loop, 'load_dotenv', lambda *a, **kw: False)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('Must not call security on Linux'))
    assert asset_loop.resolve_fal_key() == 'linux-fixture'


def test_http_error_exposes_reason_but_redacts_credentials():
    import httpx
    from asset_loop import FalHTTPError
    def handle(request):
        return httpx.Response(403, json={'detail':'Model forbidden for fixture-secret-value'})
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            b = FalQueue('fixture-secret-value', client)
            with pytest.raises(FalHTTPError) as err:
                await b.submit({'prompt':'test'})
            assert err.value.status_code == 403
            assert 'Model forbidden' in str(err.value)
            assert 'fixture-secret-value' not in str(err.value)
    asyncio.run(execute())


def test_definite_rejection_can_resume_after_user_fixes_access(tmp_path):
    from asset_loop import FalHTTPError
    class Rejected(FakeFal):
        async def submit(self,payload):
            raise FalHTTPError(403, 'forbidden')
    c = fixtures(tmp_path)
    run = tmp_path/'run'
    with pytest.raises(FalHTTPError):
        asyncio.run(run_pilot(c,run,Rejected()))
    assert (run/'candidate-01/rejection.json').exists()
    b=FakeFal()
    asyncio.run(run_pilot(c,run,b,resume=True))
    assert len(b.submissions)==5


def test_lost_response_after_rejection_is_not_replayed(tmp_path):
    from asset_loop import FalHTTPError
    class Rejected(FakeFal):
        async def submit(self,payload):
            raise FalHTTPError(403, 'forbidden')
    c=fixtures(tmp_path)
    run=tmp_path/'run'
    with pytest.raises(FalHTTPError):
        asyncio.run(run_pilot(c,run,Rejected()))
    lost=FakeFal(fail_submit=True)
    with pytest.raises(TimeoutError):
        asyncio.run(run_pilot(c,run,lost,resume=True))
    with pytest.raises(PilotError,match='unknown'):
        asyncio.run(run_pilot(c,run,lost,resume=True))
    assert len(lost.submissions)==1
