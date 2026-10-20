"""Submit the example through ComfyUI's bundled HTTP API and wait for output."""

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request


def api(server: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = request.Request(
        server.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-strength", type=float, default=1.0)
    parser.add_argument("--prefix", default="MOSS_KV_v2")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    prompt = json.loads((root / "moss_kv_v2_api.json").read_text())
    workflow = json.loads((root / "moss_kv_v2.json").read_text())
    prompt["9"]["inputs"]["seed"] = args.seed
    prompt["3"]["inputs"]["strength_model"] = args.calibration_strength
    prompt["12"]["inputs"]["filename_prefix"] = args.prefix
    for node in workflow["nodes"]:
        if node["id"] == 9:
            node["widgets_values"][0] = args.seed
        elif node["id"] == 3:
            node["widgets_values"][1] = args.calibration_strength
        elif node["id"] == 12:
            node["widgets_values"][0] = args.prefix
    queued = api(
        args.server,
        "/prompt",
        {"prompt": prompt, "extra_data": {"extra_pnginfo": {"workflow": workflow}}},
    )
    prompt_id = queued["prompt_id"]
    print(f"Queued {prompt_id}", flush=True)
    started = time.monotonic()
    while time.monotonic() - started < args.timeout:
        history = api(args.server, f"/history/{prompt_id}")
        if prompt_id in history:
            result = history[prompt_id]
            if result["status"]["status_str"] != "success":
                raise RuntimeError(json.dumps(result["status"], indent=2))
            print(json.dumps(result["outputs"], indent=2))
            print(f"Completed in {time.monotonic() - started:.2f} seconds")
            return
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} is still pending; it was not cancelled.")


if __name__ == "__main__":
    main()
