"""Five-candidate, fixed-seed fal material pilot. No creative auto-approval."""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv

from harness import Evaluator, save_json, append_json

MODEL = "fal-ai/flux-lora/inpainting"
RATE_USD_PER_MP = 0.035


class PilotError(RuntimeError):
    """Safe, actionable application error; contains no transport response bodies."""


class FalHTTPError(PilotError):
    def __init__(self, status_code, reason):
        self.status_code = status_code
        super().__init__(f"fal returned HTTP {status_code}: {reason}")


@dataclass
class PilotConfig:
    source: str
    mask: str
    prompt: str
    crop: list[int] | None = None  # x, y, width, height in source pixels
    strengths: list[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30, 0.35])
    seed: int = 42
    steps: int = 28
    guidance: float = 3.5
    max_usd: float = 1.0
    feather_px: int = 4
    timeout_seconds: float = 1200
    label: str = "material-pilot"

    def __post_init__(self):
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if not 1 <= len(self.strengths) <= 5:
            raise ValueError("Choose 1..5 strengths")
        if any(not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 < s <= 0.35 for s in self.strengths):
            raise ValueError("strengths must be finite and in (0, 0.35]")
        if len(set(self.strengths)) != len(self.strengths):
            raise ValueError("Duplicate strengths add cost without changing the comparison")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if type(self.steps) is not int or not 1 <= self.steps <= 50:
            raise ValueError("steps must be 1..50")
        if not math.isfinite(self.guidance) or not 0 <= self.guidance <= 35:
            raise ValueError("guidance must be 0..35")
        if not math.isfinite(self.max_usd) or self.max_usd < 0:
            raise ValueError("max_usd must be finite and nonnegative")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(self.feather_px) is not int or not 0 <= self.feather_px <= 64:
            raise ValueError("feather_px must be an integer in 0..64")


def read_image(path, gray=False):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def fingerprint(value):
    if isinstance(value, np.ndarray):
        data = str((value.shape, str(value.dtype))).encode() + value.tobytes()
    else:
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def inputs(config):
    source = read_image(config.source)
    mask = read_image(config.mask, gray=True)
    if source.shape[:2] != mask.shape or np.count_nonzero(mask) < 64:
        raise ValueError("mask must match source dimensions and select at least 64 pixels")
    if not np.isin(mask, [0, 255]).all():
        raise ValueError("Use a binary mask: white selects edits, black preserves the source")
    h, w = mask.shape
    crop = config.crop or [0, 0, w, h]
    if len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError("crop must contain four integers")
    x, y, cw, ch = crop
    if x < 0 or y < 0 or cw < 16 or ch < 16 or x + cw > w or y + ch > h:
        raise ValueError("crop is outside source dimensions")
    if cw % 16 or ch % 16:
        raise ValueError("crop width and height must be multiples of 16")
    if np.count_nonzero(mask[y:y+ch, x:x+cw]) != np.count_nonzero(mask):
        raise ValueError("crop must contain the entire mask")
    return source, mask, crop


def plan_run(config):
    config.__post_init__()
    source, mask, crop = inputs(config)
    cost = round(math.ceil(crop[2] * crop[3] / 1_000_000) * RATE_USD_PER_MP, 6)
    total = round(cost * len(config.strengths), 6)
    if total > config.max_usd:
        raise ValueError(f"Estimated cost ${total:.3f} exceeds budget ${config.max_usd:.3f}")
    candidates = [{"model": MODEL, "prompt": config.prompt, "seed": config.seed, "strength": strength,
                   "num_inference_steps": config.steps, "guidance_scale": config.guidance,
                   "image_size": {"width": crop[2], "height": crop[3]}, "num_images": 1,
                   "output_format": "png", "acceleration": "none", "enable_safety_checker": True}
                  for strength in config.strengths]
    return {"label": config.label, "model": MODEL, "source_sha256": fingerprint(source),
            "mask_sha256": fingerprint(mask), "source_dimensions": [source.shape[1], source.shape[0]],
            "crop": crop, "feather_px": config.feather_px, "max_usd": config.max_usd,
            "timeout_seconds": config.timeout_seconds, "candidates": candidates,
            "estimated_candidate_usd": cost, "estimated_total_usd": total,
            "pricing": {"usd_per_rounded_megapixel": RATE_USD_PER_MP,
                        "source": "https://fal.ai/models/fal-ai/flux-lora/inpainting", "checked_date": "2026-10-18"},
            "reproducibility": "Fixed seed requests; provider version is not pinned. Cached images give exact replay."}


def data_uri(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def composite(source, edit, mask, crop, feather_px=0):
    x, y, w, h = crop
    if edit.shape != (h, w, 3):
        raise ValueError("Provider changed image dimensions; resizing is not permitted")
    if mask.shape != source.shape[:2]:
        raise ValueError("mask dimensions do not match source")
    selection = (mask[y:y+h, x:x+w] > 0).astype(np.uint8)
    if feather_px:
        # Feather inward. No blending can modify pixels outside the original mask.
        padded = np.pad(selection, 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        alpha = np.minimum(distance / feather_px, 1.0) * selection
    else:
        alpha = selection.astype(np.float32)
    original = source[y:y+h, x:x+w].astype(np.float32)
    blended = np.rint(original * (1-alpha[..., None]) + edit.astype(np.float32) * alpha[..., None]).clip(0, 255).astype(np.uint8)
    result = source.copy()
    result[y:y+h, x:x+w] = blended
    if not np.array_equal(result[mask == 0], source[mask == 0]):
        raise PilotError("Unmasked source pixels changed")
    return result


def screen_candidate(source, candidate, mask):
    """Heuristic screening, never a creative approval or a physical roughness estimate."""
    a = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape or mask.shape != a.shape:
        raise ValueError("Screening dimensions do not match")
    a_edges = cv2.Canny(cv2.GaussianBlur(a, (5, 5), 1), 40, 100) > 0
    b_edges = cv2.Canny(cv2.GaussianBlur(b, (5, 5), 1), 40, 100) > 0
    if a_edges.any() and b_edges.any():
        da = cv2.distanceTransform((~a_edges).astype(np.uint8), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((~b_edges).astype(np.uint8), cv2.DIST_L2, 3)
        edge_error = max(float(np.percentile(da[b_edges], 90)), float(np.percentile(db[a_edges], 90)))
    else:
        edge_error = None
    selected = mask > 0
    f = b.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), 1)
    grain = float((f-low)[selected].std())
    flat = Evaluator().specular(b, selected)
    return {"structure": {"edge_distance_p90_px": edge_error, "observable": edge_error is not None,
                           "passed": edge_error is not None and edge_error <= 2.0},
            "material": {"highpass_std": grain, "flat_white_fraction": flat,
                         "screen_passed": grain >= 1.2 and flat <= 0.003},
            "outside_mask_unchanged": bool(np.array_equal(source[~selected], candidate[~selected])),
            "approval": "human_review_required"}


def contact_sheet(source, rows, path):
    images = [("SOURCE / unchanged", source)] + [(f"strength {strength:.2f} / screening only", img) for strength, img in rows]
    cell_w, cell_h = 480, 340
    sheet = np.full((2 * cell_h, 3 * cell_w, 3), 245, np.uint8)
    for index, (label, img) in enumerate(images):
        ox, oy = (index % 3)*cell_w, (index // 3)*cell_h
        scale = min((cell_w-16)/img.shape[1], (cell_h-42)/img.shape[0])
        thumb = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        sheet[oy+34:oy+34+thumb.shape[0], ox+8:ox+8+thumb.shape[1]] = thumb
        cv2.putText(sheet, label, (ox+8, oy+23), cv2.FONT_HERSHEY_SIMPLEX, .48, (20,20,20), 1, cv2.LINE_AA)
    write_image(path, sheet)


def prepare(config, directory):
    plan = plan_run(config)
    source, mask, crop = inputs(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "plan.json", plan)
    save_json(directory / "config.json", asdict(config))
    write_image(directory / "source.png", source)
    write_image(directory / "mask.png", mask)
    x, y, w, h = crop
    write_image(directory / "source-crop.png", source[y:y+h, x:x+w])
    write_image(directory / "mask-crop.png", mask[y:y+h, x:x+w])
    preview = source.copy()
    preview[mask > 0] = np.rint(source[mask > 0]*.5 + np.array([30, 30, 255])*.5).astype(np.uint8)
    cv2.rectangle(preview, (x,y), (x+w-1,y+h-1), (255,180,0), 3)
    write_image(directory / "mask-preview.png", preview)
    save_json(directory / "state.json", {"status": "prepared", "plan_sha256": fingerprint(plan)})
    return plan


class FalQueue:
    def __init__(self, key, client, timeout=1200, poll_interval=3):
        if not key:
            raise PilotError("Missing FAL_KEY; populate outputs/.env locally")
        self.key, self.client, self.timeout, self.poll_interval = key, client, timeout, poll_interval

    async def queue_request(self, method, url, **kwargs):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "queue.fal.run" or parsed.port not in (None,443) or parsed.username:
            raise PilotError("Unexpected queue host; credential was not sent")
        response = await self.client.request(method, url, headers={"Authorization": "Key " + self.key}, follow_redirects=False, **kwargs)
        if not 200 <= response.status_code < 300:
            reason = "Request rejected; inspect fal dashboard"
            try:
                body = response.json()
                detail = body.get("detail", body.get("message")) if isinstance(body, dict) else None
                if isinstance(detail, str):
                    reason = detail
            except ValueError:
                pass  # The HTTP error remains explicit even if its optional JSON detail is absent.
            for token in [self.key] + [p for p in self.key.split(":") if len(p) >= 8]:
                reason = reason.replace(token, "[redacted]")
            reason = re.sub(r"(?:https?://|data:)[^\s]+", "[link omitted]", reason)[:400]
            raise FalHTTPError(response.status_code, reason)
        return response.json()

    async def submit(self, payload):
        data = await self.queue_request("POST", "https://queue.fal.run/" + MODEL, json=payload)
        if not all(isinstance(data.get(k), str) and data[k] for k in ("request_id", "status_url", "response_url")):
            raise PilotError("Unknown submission outcome: incomplete queue receipt; do not replay")
        return {k: data[k] for k in ("request_id", "status_url", "response_url")}

    async def result(self, receipt):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = await self.queue_request("GET", receipt["status_url"])
            if data.get("error") or data.get("status") in {"FAILED", "CANCELLED", "CANCELED"}:
                raise PilotError("fal generation failed; no automatic resubmission")
            status = data.get("status")
            if status == "COMPLETED":
                result = await self.queue_request("GET", receipt["response_url"])
                if result.get("error"):
                    raise PilotError("fal result failed; no automatic resubmission")
                return result
            if status not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise PilotError("Unexpected fal queue status")
            await asyncio.sleep(min(self.poll_interval, max(0, deadline-time.monotonic())))
        raise TimeoutError("fal job timed out; retain its receipt before any further action")

    async def image(self, url):
        parsed = urlparse(url)
        if parsed.scheme != "https" or not (parsed.hostname == "fal.media" or (parsed.hostname or "").endswith(".fal.media")):
            raise PilotError("Unexpected result image host")
        # Authorization is deliberately absent for CDN image retrieval.
        async with self.client.stream("GET", url, follow_redirects=False) as response:
            if response.status_code != 200:
                raise PilotError(f"Image download returned HTTP {response.status_code}")
            chunks, length = [], 0
            async for chunk in response.aiter_bytes():
                length += len(chunk)
                if length > 64*1024**2:
                    raise PilotError("Image exceeds 64 MiB limit")
                chunks.append(chunk)
        image = cv2.imdecode(np.frombuffer(b"".join(chunks), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise PilotError("Cannot decode generated image")
        return image


async def run_pilot(config, directory, backend, resume=False):
    directory = Path(directory)
    plan = plan_run(config)
    if directory.exists():
        if not resume:
            raise PilotError("Run directory already exists; use resume to recover without duplicate submissions")
        previous = json.loads((directory / "plan.json").read_text())
        if fingerprint(previous) != fingerprint(plan):
            raise PilotError("Source, mask or settings changed; resume refused")
    else:
        prepare(config, directory)
    source, mask, crop = inputs(config)
    x, y, w, h = crop
    rows = []
    reserved = 0.0
    try:
        for index, settings in enumerate(plan["candidates"], 1):
            take = directory / f"candidate-{index:02d}"
            take.mkdir(exist_ok=True)
            intent, receipt_path = take / "intent.json", take / "receipt.json"
            rejection_path = take / "rejection.json"
            raw_path = take / "generated.png"
            if intent.exists() and not rejection_path.exists():
                reserved += plan["estimated_candidate_usd"]
            if raw_path.exists():
                edit = read_image(raw_path)
                saved = json.loads((take / "generated.sha256.json").read_text())
                if fingerprint(edit) != saved["sha256"]:
                    raise PilotError("Cached candidate changed; resume refused")
            else:
                if intent.exists() and not receipt_path.exists() and not rejection_path.exists():
                    raise PilotError("Submission outcome unknown; recover in fal dashboard. No automatic replay.")
                if backend is None:
                    raise PilotError("Generated image is not cached; offline replay cannot submit or poll jobs")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                else:
                    if reserved + plan["estimated_candidate_usd"] > config.max_usd + 1e-9:
                        raise PilotError("Run budget exceeded before submission")
                    reserved += plan["estimated_candidate_usd"]
                    if rejection_path.exists():
                        rejection_path.unlink()  # Clear before POST so a new lost response stays unresolved.
                    save_json(intent, {"settings": settings, "estimated_reserved_usd": plan["estimated_candidate_usd"],
                                       "submission_outcome": "unknown_until_receipt_saved"})
                    append_json(directory / "events.jsonl", {"event": "submission_intent", "candidate": index})
                    payload = {k:v for k,v in settings.items() if k != "model"}
                    payload.update(image_url=data_uri(source[y:y+h, x:x+w]), mask_url=data_uri(mask[y:y+h, x:x+w]))
                    try:
                        receipt = await backend.submit(payload)
                    except FalHTTPError as exc:
                        if exc.status_code in {401,403,404,405,422}:
                            save_json(rejection_path, {"http_status": exc.status_code, "reason": str(exc), "accepted": False})
                            reserved -= plan["estimated_candidate_usd"]
                            append_json(directory / "events.jsonl", {"event": "submission_rejected", "candidate": index, "http_status": exc.status_code})
                        raise
                    save_json(receipt_path, receipt)
                result = await backend.result(receipt)
                if result.get("seed") != config.seed:
                    raise PilotError("Provider returned a different seed")
                if any(result.get("has_nsfw_concepts", [])):
                    raise PilotError("Provider flagged this result; stopped without retry")
                images = result.get("images", [])
                if len(images) != 1 or not images[0].get("url"):
                    raise PilotError("Expected exactly one generated image")
                edit = await backend.image(images[0]["url"])
                if edit.shape != (h,w,3):
                    raise PilotError("Provider changed output dimensions")
                write_image(raw_path, edit)
                save_json(take / "generated.sha256.json", {"sha256": fingerprint(edit)})
            output = composite(source, edit, mask, crop, config.feather_px)
            write_image(take / "composite.png", output)
            report = {"full_frame": screen_candidate(source, output, mask),
                      "raw_crop": screen_candidate(source[y:y+h,x:x+w], edit, mask[y:y+h,x:x+w])}
            save_json(take / "screening.json", report)
            append_json(directory / "events.jsonl", {"event": "candidate_ready", "candidate": index, "strength": settings["strength"]})
            rows.append((settings["strength"], output[y:y+h,x:x+w]))
            print(f"candidate {index}/{len(plan['candidates'])}: strength={settings['strength']:.2f}; awaiting visual review", flush=True)
            contact_sheet(source[y:y+h,x:x+w], rows, directory / "contact-sheet.png")
        state = {"status": "awaiting_review", "candidates": len(rows), "estimated_reserved_usd": round(reserved,6),
                 "plan_sha256": fingerprint(plan), "approved": False, "next_step": "Shane/Steve selects a kitchen material candidate before Seedance motion testing"}
        save_json(directory / "state.json", state)
        alert = directory / "alert.json"
        if alert.exists():
            alert.unlink()
        return state
    except Exception as exc:
        save_json(directory / "alert.json", {"status": "stopped", "exception_type": type(exc).__name__,
                                            "reason": str(exc) if isinstance(exc, PilotError) else "Transport or local operation failed; response bodies withheld",
                                            "estimated_reserved_usd": round(reserved,6), "replay": "Never resubmit an unknown request"})
        raise


def load_config(path):
    data = json.loads(path.read_text())
    for name in ("source", "mask"):
        p = Path(data[name])
        data[name] = str((path.parent / p).resolve()) if not p.is_absolute() else str(p)
    return PilotConfig(**data)


def resolve_fal_key():
    """Read one named login Keychain item; never print or persist its value."""
    if sys.platform == "darwin":
        command = ["/usr/bin/security", "find-generic-password", "-s", "soft-cactus.fal",
                   "-a", "FAL_KEY", "-w", str(Path.home() / "Library/Keychains/login.keychain-db")]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        except subprocess.TimeoutExpired:
            raise PilotError("Keychain access timed out. Resolve the macOS access prompt, then retry.") from None
        if result.returncode == 0:
            key = result.stdout.rstrip("\r\n")
            if not key:
                raise PilotError("The soft-cactus.fal Keychain item contains an empty password")
            return key
        if result.returncode != 44:  # errSecItemNotFound; other errors must not silently change credentials.
            raise PilotError(f"Keychain access failed (exit {result.returncode}). Unlock the login keychain and allow access to soft-cactus.fal.")
    load_dotenv(Path(__file__).with_name(".env"))
    key = os.environ.get("FAL_KEY")
    if not key:
        raise PilotError("No fal key found. Save a login Keychain password named soft-cactus.fal with account FAL_KEY; alternatively set FAL_KEY in local .env.")
    return key


async def live_run(config, directory, resume):
    key = resolve_fal_key()
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        backend = FalQueue(key, client, timeout=config.timeout_seconds)
        return await run_pilot(config, directory, backend, resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "resume", "replay"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "prepare":
            plan = prepare(config, args.run_dir)
            print(f"Prepared {len(plan['candidates'])} candidates; estimated inference ${plan['estimated_total_usd']:.3f}. No network calls.")
            return 0
        if args.command == "replay":
            if not args.run_dir.is_dir():
                raise PilotError("Replay requires an existing completed run")
            result = asyncio.run(run_pilot(config, args.run_dir, None, resume=True))
        else:
            result = asyncio.run(live_run(config, args.run_dir, args.command == "resume"))
        print(json.dumps(result, indent=2))
        return 0
    except (PilotError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}. Inspect alert.json; transport details withheld.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
