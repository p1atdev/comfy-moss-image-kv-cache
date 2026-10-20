"""Measure BF16 joint/KV workflows through ComfyUI's local WebSocket API."""

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


async def generate(
    session: aiohttp.ClientSession,
    server: str,
    graph: dict[str, Any],
) -> dict[str, Any]:
    client = str(uuid.uuid4())
    async with session.ws_connect(f"{server}/ws?clientId={client}") as socket:
        async with session.post(
            f"{server}/prompt", json={"prompt": graph, "client_id": client}
        ) as response:
            queued = await response.json()
            if response.status != 200:
                raise RuntimeError(queued)
        prompt_id = queued["prompt_id"]
        sampler_start = sampler_end = image_end = None
        async for event in socket:
            if event.type != aiohttp.WSMsgType.TEXT:
                continue
            message = json.loads(event.data)
            data = message["data"]
            if data.get("prompt_id") != prompt_id:
                continue
            now = time.perf_counter()
            if message["type"] in ("execution_error", "execution_interrupted"):
                raise RuntimeError(data)
            if message["type"] == "executing":
                if data["node"] == "9":
                    sampler_start = now
                elif sampler_start is not None and sampler_end is None:
                    sampler_end = now
                if data["node"] is None:
                    break
            elif message["type"] == "executed" and data["node"] == "12":
                image_end = now
        if sampler_start is None or sampler_end is None or image_end is None:
            raise RuntimeError("Missing timing events; sampler must execute uncached.")
        async with session.get(f"{server}/history/{prompt_id}") as response:
            history = (await response.json())[prompt_id]
        return {
            "prompt_id": prompt_id,
            "sampler_node_seconds": sampler_end - sampler_start,
            "sampler_to_saved_image_seconds": image_end - sampler_start,
            "images": history["outputs"]["12"]["images"],
        }


async def main(args: argparse.Namespace) -> None:
    graph = json.loads(args.workflow.read_text())
    prompts = json.loads(args.prompts.read_text())
    variants = ("joint", "kv", "kv_corrected")
    result: dict[str, Any] = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "workflow_sha256": hashlib.sha256(args.workflow.read_bytes()).hexdigest(),
        "calibration_lora": args.calibration_lora or graph["3"]["inputs"]["lora_name"],
        "precision": "bfloat16",
        "dmd_strength": 0.7,
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "size": 1024,
        "text_timestep": 1000,
        "warmups": 1,
        "repeats": args.repeats,
        "timing_scope": "KSampler start to SaveImage completion, measured by local WebSocket events; includes fresh text KV construction, sampling, VAE decode and PNG saving; excludes text encoding and warm model loading",
        "prompts": prompts,
        "measurements": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{args.server}/queue") as response:
            queue = await response.json()
            if queue["queue_running"] or queue["queue_pending"]:
                raise RuntimeError("Wait for the ComfyUI queue to become empty.")
        async with session.get(f"{args.server}/system_stats") as response:
            stats = await response.json()
            result["gpu"] = stats["devices"][0]["name"]
            result["comfyui_version"] = stats["system"]["comfyui_version"]
            result["pytorch_version"] = stats["system"]["pytorch_version"]
        for case in prompts:
            for variant in variants:
                runs = []
                for repeat in range(args.repeats + 1):
                    current = copy.deepcopy(graph)
                    if args.calibration_lora:
                        current["3"]["inputs"]["lora_name"] = args.calibration_lora
                    if variant == "joint":
                        current["9"]["inputs"]["model"] = ["2", 0]
                        del current["3"], current["4"]
                    elif variant == "kv":
                        current["4"]["inputs"]["model"] = ["2", 0]
                        del current["3"]
                    current["6"]["inputs"]["text"] = case["prompt"]
                    # The same seed sequence is used for all three variants.
                    # Changing it invalidates ComfyUI's sampler output cache.
                    current["9"]["inputs"]["seed"] = args.seed - 1 + repeat
                    current["12"]["inputs"]["filename_prefix"] = (
                        f"{args.prefix}/{case['id']}_{variant}_{repeat}"
                    )
                    measured = await generate(session, args.server, current)
                    measured["seed"] = args.seed - 1 + repeat
                    print(
                        f"{case['id']} {variant} {'warmup' if repeat == 0 else repeat}: "
                        f"{measured['sampler_to_saved_image_seconds']:.3f}s",
                        flush=True,
                    )
                    if repeat == 0:
                        warmup = measured
                    else:
                        runs.append(measured)
                times = [run["sampler_to_saved_image_seconds"] for run in runs]
                result["measurements"][f"{case['id']}_{variant}"] = {
                    "warmup": warmup,
                    "runs": runs,
                    "median_seconds": statistics.median(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                }
                args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--workflow",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/moss_kv_v2_api.json",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/benchmark_prompts.json",
    )
    parser.add_argument("--prefix", default="benchmark_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-lora")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parsed = parser.parse_args()
    if parsed.repeats < 1:
        parser.error("repeats must be positive")
    asyncio.run(main(parsed))
