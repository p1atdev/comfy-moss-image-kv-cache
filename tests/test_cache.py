from unittest.mock import patch

import comfy.conds
import comfy.model_base
import comfy.ops
import comfy.supported_models
import pytest
import torch
from comfy.ldm.moss_image.model import (
    MOSSImage,
    rope_split_half,
    situ,
    timestep_embedding,
)
from comfy.model_patcher import ModelPatcher
from moss_kv_test.model import cached_extra_conds, cached_forward, prepare_text_kv_cache
from moss_kv_test.nodes import MOSSImageKVCache
from torch.nn import functional as F


def initialize(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        if "norm" in name:
            parameter.fill_(1.0)
        else:
            parameter.copy_(torch.randn_like(parameter) * 0.1)


def tiny_model(fused: bool) -> MOSSImage:
    model = MOSSImage(
        dim=32,
        ffn_dim=48,
        num_heads=2,
        num_layers=3,
        text_dim=12,
        latent_channels=8,
        fused_qkv=fused,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    initialize(model)
    return model


def isolated_reference(
    model: MOSSImage,
    image: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    fixed_time: float,
) -> torch.Tensor:
    batch, _, height, width = image.shape
    count = height * width
    pe = model.rope(height, width, context.shape[1], image.device)
    vec = model.time_embedding(timestep_embedding(timestep, 256))
    text_vec = model.time_embedding(
        timestep_embedding(torch.full((batch,), fixed_time / 1000), 256)
    )
    img_mod = (
        model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    )
    txt_mod = (
        model.shared_block_modulation(text_vec)
        .unflatten(-1, (6, -1))
        .unsqueeze(2)
        .unbind(1)
    )
    mod = [
        torch.cat((a.expand(-1, count, -1), b.expand(-1, context.shape[1], -1)), dim=1)
        for a, b in zip(img_mod, txt_mod)
    ]
    head_shift, head_scale = (
        model.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    )
    x = torch.cat(
        (
            model.video_embedder(image.flatten(2).transpose(1, 2)),
            model.text_embedder(context, pe[:, count:]),
        ),
        dim=1,
    )
    mask = torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool)
    mask[count:, :count] = False
    for block in model.layers:
        y = block.norm1(x) * (1 + mod[1]) + mod[0]
        if block.fused_qkv:
            q, k, v = block.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
        shape = (batch, x.shape[1], block.num_heads, -1)
        q = rope_split_half(block.q_norm(q.reshape(shape)), pe).transpose(1, 2)
        k = rope_split_half(block.k_norm(k.reshape(shape)), pe).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)
        out = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            .transpose(1, 2)
            .flatten(2)
        )
        x = x + block.proj["o"](out) * mod[2]
        y = block.norm2(x) * (1 + mod[4]) + mod[3]
        hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, -1)
        x = x + block.proj["ffn_down"](situ(hidden, gate)) * mod[5]
    out = model.final_linear_video(
        model.final_norm_video(x[:, :count]) * (1 + head_scale) + head_shift
    )
    return out.transpose(1, 2).reshape(batch, 8, height, width)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fixed_time", [0.0, 1000.0])
def test_cached_matches_isolated_recomputation(fused: bool, fixed_time: float) -> None:
    torch.manual_seed(8)
    torch.set_num_threads(2)
    model = tiny_model(fused)
    context = torch.randn(1, 7, 12)
    image = torch.randn(1, 8, 2, 3)
    with patch.object(
        model.text_embedder, "forward", wraps=model.text_embedder.forward
    ) as encode:
        cache = prepare_text_kv_cache(model, context, fixed_time)
        for value in (1.0, 0.2):
            timestep = torch.tensor([value])
            actual = cached_forward(model, image, timestep, moss_text_kv=cache)
            expected = isolated_reference(model, image, timestep, context, fixed_time)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        assert encode.call_count == 3  # one cached encoding, two reference encodings
    for tensor in cache:
        assert tensor.storage_offset() == 0
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )


def test_condition_batching_preserves_each_prompt_cache() -> None:
    torch.manual_seed(12)
    model = tiny_model(False)
    image = torch.randn(2, 8, 2, 3)
    contexts = [torch.randn(1, 5, 12), torch.randn(1, 5, 12)]
    caches = [prepare_text_kv_cache(model, context, 1000) for context in contexts]
    conditions = [comfy.conds.CONDList(cache) for cache in caches]
    assert conditions[0].can_concat(conditions[1])
    batched = conditions[0].concat([conditions[1]])
    actual = cached_forward(
        model, image, torch.tensor([0.4, 0.4]), moss_text_kv=batched
    )
    for index in range(2):
        expected = isolated_reference(
            model, image[index : index + 1], torch.tensor([0.4]), contexts[index], 1000
        )
        torch.testing.assert_close(
            actual[index : index + 1], expected, atol=2e-5, rtol=2e-5
        )
    other = comfy.conds.CONDList(
        prepare_text_kv_cache(model, torch.randn(1, 3, 12), 1000)
    )
    assert not conditions[0].can_concat(other)
    repeated = conditions[0].process_cond(batch_size=3).cond
    assert all(tensor.shape[0] == 3 for tensor in repeated)


def test_preprocessing_is_fresh_and_patcher_clone_is_independent() -> None:
    config = comfy.supported_models.MOSSImage(
        {
            "dim": 32,
            "ffn_dim": 48,
            "num_heads": 2,
            "num_layers": 3,
            "text_dim": 12,
            "latent_channels": 8,
        }
    )
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.MOSSImage(config, device="cpu")
    initialize(model)
    patcher = ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    output = MOSSImageKVCache.execute(patcher, text_timestep=1000)
    patched = output.result[0]
    assert patcher.object_patches == {}
    assert set(patched.object_patches) == {"extra_conds", "diffusion_model._forward"}
    kwargs = {
        "cross_attn": torch.randn(1, 5, 12),
        "device": torch.device("cpu"),
        "noise": torch.randn(1, 8, 2, 3),
    }
    a = cached_extra_conds(model, text_timestep=1000, **kwargs)
    b = cached_extra_conds(model, text_timestep=1000, **kwargs)
    assert "c_crossattn" not in a
    assert a["moss_text_kv"].cond[0].data_ptr() != b["moss_text_kv"].cond[0].data_ptr()
    torch.testing.assert_close(a["moss_text_kv"].cond[0], b["moss_text_kv"].cond[0])
