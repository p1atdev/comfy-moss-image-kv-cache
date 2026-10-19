import json
from pathlib import Path

import comfy.lora
import comfy.model_base
import comfy.supported_models
import folder_paths
import pytest
import torch
from comfy.model_patcher import ModelPatcher
from convert_calibration_lora import convert_checkpoint
from safetensors.torch import load_file, save_file

import nodes


@pytest.mark.parametrize("fused", [False, True])
def test_standard_loader_applies_converted_delta(tmp_path: Path, fused: bool) -> None:
    torch.manual_seed(2)
    name = "layers.0.proj.q"
    a, b = torch.randn(2, 32), torch.randn(32, 2)
    source, target = (
        tmp_path / "calibration.safetensors",
        tmp_path / "standard_lora.safetensors",
    )
    save_file(
        {f"{name}.calibration_A": a, f"{name}.calibration_B": b},
        source,
        metadata={
            "format": "moss-image-calibration-v1",
            "adapters": json.dumps({name: {"rank": 2, "alpha": 4}}),
            "recipe": "{}",
        },
    )
    convert_checkpoint(source, target)
    exported = load_file(target)
    torch.testing.assert_close(
        exported[f"diffusion_model.{name}.lora_down.weight"], a, atol=0, rtol=0
    )
    torch.testing.assert_close(
        exported[f"diffusion_model.{name}.lora_up.weight"], b, atol=0, rtol=0
    )
    config = comfy.supported_models.MOSSImage(
        {
            "dim": 32,
            "ffn_dim": 48,
            "num_heads": 2,
            "num_layers": 1,
            "text_dim": 12,
            "latent_channels": 8,
            "fused_qkv": fused,
        }
    )
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.MOSSImage(config, device="cpu")
    patcher = ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    folder_paths.add_model_folder_path("loras", str(tmp_path))
    patched = nodes.LoraLoaderModelOnly().load_lora_model_only(
        patcher, target.name, 0.7
    )[0]
    key = "diffusion_model.layers.0.proj." + ("qkv" if fused else "q") + ".weight"
    assert set(patched.patches) == {key}
    assert patcher.patches == {}
    weight = torch.randn(96 if fused else 32, 32)
    actual = comfy.lora.calculate_weight(patched.patches[key], weight.clone(), key)
    expected = weight.clone()
    expected[:32] += (b @ a) * (0.7 * 4 / 2)
    torch.testing.assert_close(actual, expected)


def test_converter_rejects_other_formats(tmp_path: Path) -> None:
    source, target = tmp_path / "bad.safetensors", tmp_path / "result.safetensors"
    save_file({"weight": torch.ones(2, 2)}, source)
    with pytest.raises(ValueError, match="moss-image-calibration"):
        convert_checkpoint(source, target)
    assert not target.exists()
