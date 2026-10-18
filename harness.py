"""Bounded Higgsfield MCP orchestration and streaming static-scene CV grading."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from jsonschema import validate


@dataclass
class Rubric:
    # Pixel units refer to the native decoded resolution; luma is 0..255.
    white_luma: int = 240
    min_blob_pixels: int = 64
    flat_laplacian_variance: float = 5.0
    max_flat_white_fraction: float = 0.003
    min_spatial_grain_std: float = 1.2
    max_temporal_std: float = 2.0
    max_flicker_luma: float = 2.0
    max_step_flow_px: float = 0.5
    max_anchor_flow_px: float = 0.75
    max_shimmer_luma: float = 3.0
    flow_width: int = 640

    def __post_init__(self):
        if not 0 <= self.white_luma <= 255 or self.flow_width < 32:
            raise ValueError("invalid white_luma or flow_width")
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("rubric values must be finite and nonnegative")


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if expected > 0 and count < expected:
            raise ValueError(f"Truncated video: decoded {count} of {expected} frames")
    finally:
        cap.release()


class Evaluator:
    def __init__(self, rubric: Rubric | None = None):
        self.r = rubric or Rubric()

    def specular(self, gray, mask):
        bright = ((gray >= self.r.white_luma) & mask).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
        lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        flat = 0
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < self.r.min_blob_pixels:
                continue
            interior = cv2.erode((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if interior.any() and float(lap[interior].var()) < self.r.flat_laplacian_variance:
                flat += int(interior.sum())
        return flat / int(mask.sum())

    def flow(self, a, b, mask, scale):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=2) / scale
        # Worst block's median catches local drift that a global mean conceals.
        scores = []
        for y in range(0, a.shape[0], 32):
            for x in range(0, a.shape[1], 32):
                roi = mask[y:y+32, x:x+32]
                if roi.sum() >= 16:
                    scores.append(float(np.median(magnitude[y:y+32, x:x+32][roi])))
        if not scores:
            raise ValueError("mask has insufficient flow coverage")
        return max(scores)

    def evaluate(self, frames: Iterable[np.ndarray], mask: np.ndarray | None = None):
        iterator = iter(frames)
        try:
            first = next(iterator)
        except StopIteration as e:
            raise ValueError("Need at least two decoded frames") from e
        if first.ndim != 2 or first.dtype != np.uint8:
            raise ValueError("frames must be uint8 grayscale")
        mask = np.ones(first.shape, bool) if mask is None else np.asarray(mask, bool)
        if mask.shape != first.shape or mask.sum() < 64:
            raise ValueError("mask must match frame and contain at least 64 pixels")
        scale = min(1.0, self.r.flow_width / first.shape[1])
        size = (round(first.shape[1] * scale), round(first.shape[0] * scale))
        small_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        anchor = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
        previous_small = anchor
        previous = first.astype(np.float32)
        mean = first[mask].astype(np.float64)
        m2 = np.zeros_like(mean)
        n = 1
        grains, flat = [], []
        step_flow = anchor_flow = flicker = shimmer = 0.0

        def spatial(gray):
            f = gray.astype(np.float32)
            low = cv2.GaussianBlur(f, (0, 0), 1.0)
            # Exclude structural boundaries from grain measurement.
            gx = cv2.Sobel(low, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(low, cv2.CV_32F, 0, 1)
            planes = mask & (np.hypot(gx, gy) < 20) & (gray < self.r.white_luma)
            if planes.sum() < 64:
                raise ValueError("Insufficient unmoving plane pixels for texture grading")
            return float((f - low)[planes].std())

        grains.append(spatial(first))
        flat.append(self.specular(first, mask))
        for gray in iterator:
            if gray.shape != first.shape or gray.dtype != np.uint8:
                raise ValueError("Frame dimensions or dtype changed")
            n += 1
            f = gray.astype(np.float32)
            values = gray[mask].astype(np.float64)
            delta = values - mean
            mean += delta / n
            m2 += delta * (values - mean)
            diff = (f - previous)[mask]
            shift = float(np.median(diff))
            flicker = max(flicker, abs(shift))
            shimmer = max(shimmer, float(np.sqrt(np.mean((diff - shift)**2))))
            small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            step_flow = max(step_flow, self.flow(previous_small, small, small_mask, scale))
            anchor_flow = max(anchor_flow, self.flow(anchor, small, small_mask, scale))
            previous_small, previous = small, f
            grains.append(spatial(gray))
            flat.append(self.specular(gray, mask))
        if n < 2:
            raise ValueError("Need at least two decoded frames")
        temporal_std = float(np.mean(np.sqrt(m2 / n)))
        spec = {"flat_white_fraction": max(flat), "passed": max(flat) <= self.r.max_flat_white_fraction}
        lum = {"spatial_grain_std": min(grains), "grain_std_range": max(grains) - min(grains),
               "temporal_std": temporal_std, "passed": min(grains) >= self.r.min_spatial_grain_std}
        temporal = {"step_flow_px": step_flow, "anchor_flow_px": anchor_flow,
                    "flicker_luma": flicker, "shimmer_luma": shimmer,
                    "passed": step_flow <= self.r.max_step_flow_px and anchor_flow <= self.r.max_anchor_flow_px
                    and flicker <= self.r.max_flicker_luma and shimmer <= self.r.max_shimmer_luma
                    and temporal_std <= self.r.max_temporal_std}
        return {"passed": all(x["passed"] for x in (spec, lum, temporal)), "frames": n,
                "width": first.shape[1], "height": first.shape[0], "mask_pixels": int(mask.sum()),
                "specular": spec, "luminance": lum, "temporal": temporal, "rubric": asdict(self.r)}


LOCK = "Camera locked on tripod. Static background geometry and fixed lighting. No camera movement."
TAGS = "matte rough micro-texture grain, physical drywall occlusion"


@dataclass
class State:
    scene: str = "Static minimalist indoor room"
    modifiers: str = "matte finishes, micro-surface imperfections, subtle camera grain"
    seed: int = 42
    strength: float = 0.35
    motion_prompt: str = LOCK

    def __post_init__(self):
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 0.35:
            raise ValueError("strength must be in [0, 0.35]")


def feedback(state, report):
    if not report["specular"]["passed"]:
        state.strength = max(0.0, round(state.strength - 0.05, 2))
    if not report["luminance"]["passed"]:
        if TAGS not in state.modifiers:
            state.modifiers += ", " + TAGS
        state.seed += 1
    if not report["temporal"]["passed"]:
        state.motion_prompt = LOCK + " No geometry deformation, exposure changes, flicker or texture shimmer."


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def run_loop(backend, state, directory, evaluate, max_loops=5):
    if not 1 <= max_loops <= 5:
        raise ValueError("max_loops must be 1..5 (includes initial attempt)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "iterations.jsonl").exists():
        raise ValueError("Use a fresh run directory to avoid mixing runs")
    try:
        await backend.preflight(state)
        for attempt in range(1, max_loops + 1):
            take = directory / f"attempt-{attempt:02d}"
            take.mkdir()
            save_json(take / "state.json", asdict(state))
            path = await backend.iteration(state, take)
            report = evaluate(path)
            save_json(take / "evaluation.json", report)
            append_json(directory / "iterations.jsonl", {"attempt": attempt, "state": asdict(state), "evaluation": report})
            print(json.dumps({"attempt": attempt, "passed": report["passed"]}), flush=True)
            if report["passed"]:
                save_json(directory / "result.json", {"status": "passed", "video": str(path), "attempt": attempt})
                return 0
            if attempt < max_loops:
                feedback(state, report)
        save_json(directory / "alert.json", {"reason": "validation_exhausted", "attempts": max_loops})
        print("ALERT: validation exhausted; see alert.json", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        # Avoid persisting transport exception text: it may contain signed URLs or credentials.
        save_json(directory / "alert.json", {"reason": "execution_error", "exception_type": type(exc).__name__})
        print(f"ALERT: execution failed ({type(exc).__name__}); see alert.json", file=sys.stderr, flush=True)
        raise


def validate_model(model, params):
    if model.get("error") or not model.get("id"):
        raise ValueError("Model unavailable in connected catalog")
    parameters = {p["name"]: p for p in model.get("parameters", [])}
    for key, value in params.items():
        if key in {"model", "prompt", "medias", "count", "get_cost"}:
            continue
        if key == "aspect_ratio":
            if value not in model.get("aspect_ratios", []):
                raise ValueError(f"{model['id']}: unsupported aspect_ratio {value}")
            continue
        if key not in parameters:
            raise ValueError(f"{model['id']}: unsupported parameter {key}")
        p = parameters[key]
        if "options" in p and value not in p["options"]:
            raise ValueError(f"{model['id']}: unsupported {key}={value}; allowed {p['options']}")
        if p.get("type") == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric {key}")
            if value < p.get("min", -math.inf) or value > p.get("max", math.inf):
                raise ValueError(f"Out of range {key}")
    for p in parameters.values():
        if p.get("required") == "required" and p["name"] not in params:
            raise ValueError(f"Missing parameter {p['name']}")
    roles = {r for m in model.get("medias", []) for r in m.get("roles", [])}
    for media in params.get("medias", []):
        if media["role"] not in roles:
            raise ValueError(f"Unsupported reference role {media['role']}")


class MCPBackend:
    """Official MCP ClientSession; no invented REST generation endpoints."""
    def __init__(self, session, config, directory):
        self.session, self.config, self.directory = session, config, Path(directory)
        self.tools = {}
        self.catalog = {}
        self.spent = 0.0

    async def call(self, name, arguments):
        if self.tools:
            if name not in self.tools:
                raise ValueError(f"MCP tool missing: {name}")
            validate(arguments, self.tools[name].inputSchema)
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"MCP tool {name} returned an error; no automatic resubmission")
        data = result.structuredContent
        if data is None:
            for part in result.content:
                if getattr(part, "type", None) == "text":
                    try:
                        candidate = json.loads(part.text)
                    except json.JSONDecodeError:
                        continue  # Non-JSON display text can accompany the structured result.
                    if isinstance(candidate, dict):
                        data = candidate
                        break
        if not isinstance(data, dict):
            raise RuntimeError(f"{name}: missing structured result")
        if data.get("error") or data.get("unlim_choice") or data.get("recovery_tool"):
            raise RuntimeError(f"{name}: rejected or requires host interaction; no resubmission")
        return data

    def payloads(self, state, soul_id="PREFLIGHT", anchor_id="PREFLIGHT"):
        models = self.config.get("models", {})
        return [
            ("image", {"model": models.get("soul", "soul_cinematic"), "prompt": state.scene + ", clean architectural lines, minimalist interior composition, f/4 aperture",
                       "aspect_ratio": "16:9", "seed": state.seed}),
            ("image", {"model": models.get("sunburst", "gpt_image_2_5"), "variant": "sunburst", "prompt": "Preserve composition, geometry and lighting. Enhance surface details: " + state.modifiers,
                       "medias": [{"role": "image_references", "value": soul_id}], "strength": state.strength, "seed": state.seed}),
            ("video", {"model": models.get("seedance", "seedance_2_5"), "prompt": state.motion_prompt,
                       "medias": [{"role": "start_image", "value": anchor_id}], "mode": "omni_reference",
                       "resolution": "1080p_pro", "duration": 15, "aspect_ratio": "16:9"})]

    async def preflight(self, state):
        listed = await self.session.list_tools()
        self.tools = {t.name: t for t in listed.tools}
        while listed.nextCursor:
            listed = await self.session.list_tools(cursor=listed.nextCursor)
            self.tools.update({t.name: t for t in listed.tools})
        issues = []
        for kind, params in self.payloads(state):
            model = await self.call("models_explore", {"action": "get", "model_id": params["model"]})
            self.catalog[params["model"]] = model
            try:
                validate_model(model, params)
                tool = "generate_" + kind
                if tool not in self.tools:
                    raise ValueError(f"Missing {tool}")
                validate({"params": params}, self.tools[tool].inputSchema)
            except (ValueError, KeyError) as exc:
                issues.append(str(exc))
        # Persist catalog without authentication data or private URL query parameters.
        save_json(self.directory / "preflight.json", {"models": self.catalog, "issues": issues})
        if issues:
            raise ValueError("Strict contract unavailable: " + "; ".join(issues))

    async def generate(self, kind, params):
        data = await self.call("generate_" + kind, {"params": params})
        jobs = data.get("results", [])
        ids = [j["id"] for j in jobs]
        # Record IDs before checking adjusted settings: accepted jobs may already cost credits.
        append_json(self.directory / "jobs.jsonl", {"kind": kind, "job_ids": ids})
        if data.get("adjustments"):
            raise RuntimeError(f"Generation adjusted requested settings; retained job IDs {ids}")
        if len(ids) != 1:
            raise RuntimeError("Unknown submission outcome: inspect jobs.jsonl; never blindly retry")
        url = await self.wait(ids[0])
        return ids[0], url

    async def wait(self, job_id):
        deadline = time.monotonic() + self.config.get("job_timeout_seconds", 1800)
        while time.monotonic() < deadline:
            data = await self.call("jobs_wait", {"jobs": [{"index": 0, "job_id": job_id}], "timeout_seconds": 15})
            jobs = data.get("jobs", [])
            if len(jobs) != 1 or jobs[0].get("job_id") != job_id:
                raise RuntimeError("Unexpected job status response")
            job = jobs[0]
            if job["status"] == "completed":
                if not job.get("result_url"):
                    raise RuntimeError("Completed job has no result URL")
                return job["result_url"]
            if job["status"] in {"failed", "canceled", "nsfw", "ip_detected", "lookup_failed"}:
                raise RuntimeError(f"Job {job_id}: {job['status']}")
            await asyncio.sleep(min(5, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"Job {job_id} timed out; recover this job before any resubmission")

    async def iteration(self, state, directory):
        refs = ["PREFLIGHT", "PREFLIGHT"]
        for index in range(3):
            kind, params = self.payloads(state, *refs)[index]
            validate_model(self.catalog[params["model"]], params)
            quote = await self.call("generate_" + kind, {"params": {**params, "get_cost": True}})
            cost = quote.get("cost", {}).get("credits_exact")
            budget = self.config.get("max_credits", 0)
            if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                raise RuntimeError("Missing valid cost quote")
            if self.spent + cost > budget:
                raise RuntimeError("Run credit budget exceeded before submission")
            self.spent += cost  # Reserve even if submission response is lost.
            append_json(self.directory / "spend.jsonl", {"stage": index, "reserved_credits": cost, "total": self.spent})
            job_id, url = await self.generate(kind, params)
            if index < 2:
                refs[index] = job_id
            else:
                await download(url, directory / "video.mp4")
        return directory / "video.mp4"


async def download(url, destination):
    if urlparse(url).scheme != "https":
        raise ValueError("Result URL must use HTTPS")
    partial = destination.with_suffix(".partial")
    limit = 2 * 1024**3
    total = 0
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > limit:
                        raise ValueError("Video download exceeds 2 GiB")
                    f.write(chunk)
    if not total:
        raise ValueError("Empty video download")
    partial.replace(destination)


def evaluate_file(path, rubric, mask_path=None):
    mask = None
    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Cannot decode mask")
        mask = mask > 0
    return Evaluator(rubric).evaluate(read_video(path), mask)


async def connected_run(config, directory, preflight_only):
    from contextlib import AsyncExitStack
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    load_dotenv(Path(__file__).with_name(".env"))
    async with AsyncExitStack() as stack:
        url = os.getenv("HIGGSFIELD_MCP_URL")
        command = os.getenv("HIGGSFIELD_MCP_COMMAND")
        if bool(url) == bool(command):
            raise ValueError("Set exactly one MCP URL or MCP command in .env")
        if url:
            if urlparse(url).scheme != "https":
                raise ValueError("MCP URL must use HTTPS")
            token = os.getenv("HIGGSFIELD_MCP_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            read, write, _ = await stack.enter_async_context(streamablehttp_client(url, headers=headers))
        else:
            argv = json.loads(command)
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                raise ValueError("MCP command must be a JSON array of strings")
            read, write = await stack.enter_async_context(stdio_client(StdioServerParameters(command=argv[0], args=argv[1:])))
        session = await stack.enter_async_context(ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)))
        await session.initialize()
        backend = MCPBackend(session, config, directory)
        state = State(**config.get("state", {}))
        if preflight_only:
            await backend.preflight(state)
            return 0
        rubric = Rubric(**config.get("rubric", {}))
        return await run_loop(backend, state, directory,
                              lambda p: evaluate_file(p, rubric, config.get("mask")), config.get("max_loops", 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("video", type=Path)
    ev.add_argument("--mask", type=Path)
    ev.add_argument("--rubric", type=Path)
    ev.add_argument("--report", type=Path, required=True)
    for cmd in ("preflight", "run"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    owns_directory = False
    try:
        if args.command == "evaluate":
            rubric = Rubric(**json.loads(args.rubric.read_text())) if args.rubric else Rubric()
            report = evaluate_file(args.video, rubric, args.mask)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.report, report)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 2
        config = json.loads(args.config.read_text())
        args.run_dir.mkdir(parents=True, exist_ok=False)
        owns_directory = True
        save_json(args.run_dir / "config.json", config)
        return asyncio.run(connected_run(config, args.run_dir, args.command == "preflight"))
    except Exception as exc:
        # Exception details from transport libraries may contain tokens. Keep CLI output safe.
        if owns_directory:
            save_json(args.run_dir / "alert.json", {"reason": "execution_error", "exception_type": type(exc).__name__})
        print(f"ERROR: {type(exc).__name__}. Inspect preflight.json/alert.json; transport details withheld.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
