"""Make local Hugging Face shards available to standard ComfyUI loaders."""

import argparse
import gc
from pathlib import Path

from convert_calibration_lora import convert_checkpoint
from safetensors import safe_open
from safetensors.torch import save_file


def consolidate(folder: Path, target: Path, *, key_prefix: str = "") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = str(folder.resolve())
    if target.exists():
        with safe_open(target, framework="pt") as file:
            metadata = file.metadata() or {}
            if (
                metadata.get("source") != source
                or metadata.get("key_prefix", "") != key_prefix
            ):
                raise ValueError(f"Existing model has a different source: {target}")
        print(f"Already prepared: {target}", flush=True)
        return
    state = {}
    for shard in sorted(folder.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as file:
            for key in file.keys():
                name = key_prefix + key
                if name in state:
                    raise ValueError(f"Duplicate tensor in source shards: {key}")
                state[name] = file.get_tensor(key)
    if not state:
        raise FileNotFoundError(f"No safetensors in {folder}")
    print(f"Consolidating {len(state)} tensors to {target}", flush=True)
    temporary = target.with_suffix(".partial.safetensors")
    save_file(state, temporary, metadata={"source": source, "key_prefix": key_prefix})
    temporary.replace(target)
    del state
    gc.collect()


def link(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise ValueError(f"Existing model points to a different source: {target}")
        return
    target.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dmd-lora", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    args = parser.parse_args()
    consolidate(
        args.source / "transformer",
        args.models_dir / "diffusion_models/moss_image_bf16.safetensors",
    )
    consolidate(
        args.source / "text_encoder",
        args.models_dir / "text_encoders/qwen3_5_9b_moss.safetensors",
        key_prefix="model.",
    )
    link(
        args.source / "vae/diffusion_pytorch_model.safetensors",
        args.models_dir / "vae/moss_flux2_vae.safetensors",
    )
    link(args.dmd_lora, args.models_dir / "loras/moss_dmd_8step.safetensors")
    for step in (100, 500):
        convert_checkpoint(
            args.training_run / f"checkpoints/step_{step:06d}/calibration.safetensors",
            args.models_dir / f"loras/moss_kv_cal{step}.safetensors",
        )
    print(f"Models ready: {args.models_dir}", flush=True)


if __name__ == "__main__":
    main()
