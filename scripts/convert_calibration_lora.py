"""Export a trained calibration checkpoint for ComfyUI's standard LoRA loader."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert_checkpoint(source: Path, target: Path) -> None:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        with safe_open(target, framework="pt") as file:
            if (file.metadata() or {}).get("source_sha256") != digest:
                raise ValueError(f"Existing LoRA has a different source: {target}")
        return
    with safe_open(source, framework="pt") as file:
        metadata = file.metadata() or {}
        if metadata.get("format") != "moss-image-calibration-v1":
            raise ValueError("Expected a moss-image-calibration-v1 checkpoint.")
        specs = json.loads(metadata["adapters"])
        expected = {
            f"{name}.{suffix}"
            for name in specs
            for suffix in ("calibration_A", "calibration_B")
        }
        if set(file.keys()) != expected or not specs:
            raise ValueError(
                "Calibration tensor keys do not match its adapter metadata."
            )
        lora = {}
        for name, spec in specs.items():
            if (
                re.fullmatch(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)", name)
                is None
            ):
                raise ValueError(f"Unsupported calibration target: {name}")
            rank, alpha = spec["rank"], spec["alpha"]
            a, b = (
                file.get_tensor(f"{name}.calibration_A"),
                file.get_tensor(f"{name}.calibration_B"),
            )
            if (
                not isinstance(rank, int)
                or rank < 1
                or not math.isfinite(alpha)
                or a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != rank
                or b.shape[1] != rank
            ):
                raise ValueError(f"Invalid rank, alpha or tensor shapes: {name}")
            prefix = f"diffusion_model.{name}"
            lora[f"{prefix}.lora_down.weight"] = a
            lora[f"{prefix}.lora_up.weight"] = b
            lora[f"{prefix}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.safetensors")
    save_file(
        lora,
        temporary,
        metadata={
            "format": "pt",
            "source_sha256": digest,
            "source_recipe": metadata["recipe"],
        },
    )
    temporary.replace(target)
    print(f"Converted {len(specs)} projections: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    convert_checkpoint(args.source, args.target)


if __name__ == "__main__":
    main()
