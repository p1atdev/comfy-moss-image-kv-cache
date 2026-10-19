"""Fixed-timestep text KV conditioning for ComfyUI's native MOSS-Image model."""

from typing import Any

import comfy.conds
import comfy.model_base
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
import torch
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.ldm.moss_image.model import (
    Block,
    MOSSImage,
    gated_residual,
    modulated_rms_norm,
    rope_split_half,
    timestep_embedding,
)


def prepare_attention(
    block: Block, x: torch.Tensor, pe: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block.fused_qkv:
        q, k, v = block.proj["qkv"](x).chunk(3, dim=-1)
    else:
        q, k, v = block.proj["q"](x), block.proj["k"](x), block.proj["v"](x)
    batch, length, _ = q.shape
    q = q.view(batch, length, block.num_heads, -1)
    k = k.view(batch, length, block.num_heads, -1)
    # Expose the rotated K used by native attention so it can be reused.
    if comfy.model_management.in_training:
        q = rope_split_half(block.q_norm(q), pe)
        k = rope_split_half(block.k_norm(k), pe)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(
            block.q_norm, q, offloadable=True
        )
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(
            block.k_norm, k, offloadable=True
        )
        q, k = comfy.quant_ops.ck.rms_rope_split_half(
            q, k, pe, q_scale, k_scale, block.q_norm.eps
        )
        comfy.ops.uncast_bias_weight(block.q_norm, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(block.k_norm, k_scale, None, k_stream)
    return q.flatten(2), k.flatten(2), v


def attend(
    block: Block,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    options: dict[str, Any],
) -> torch.Tensor:
    return optimized_attention(
        AttentionTensorContainer(q),
        AttentionTensorContainer(k),
        AttentionTensorContainer(v),
        block.num_heads,
        transformer_options=options,
        preferred_attention=block.comfy_attention,
    )


def finish_block(
    block: Block,
    x: torch.Tensor,
    attention: torch.Tensor,
    mod: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    _, _, gate_a, shift_f, scale_f, gate_f = mod
    x = gated_residual(x, block.proj["o"](attention), gate_a)
    y = modulated_rms_norm(x, block.norm2, scale_f, shift_f)
    return gated_residual(
        x,
        comfy.ops.linear_input_act(
            block.proj["ffn_down"], block.proj["ffn_up_gate"](y), "situ"
        ),
        gate_f,
    )


def prepare_text_kv_cache(
    model: MOSSImage,
    context: torch.Tensor,
    text_timestep: float,
) -> list[torch.Tensor]:
    options: dict[str, Any] = {}
    pe = model.rope(1, 1, context.shape[1], context.device)[:, 1:]
    # Native MOSS uses normalized time; timestep_embedding multiplies it by 1000.
    timestep = torch.full(
        (context.shape[0],),
        text_timestep / 1000,
        device=context.device,
        dtype=torch.float32,
    )
    vec = model.time_embedding(timestep_embedding(timestep, 256).to(context.dtype))
    mod = (
        model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    )
    text = model.text_embedder(context, pe, options)
    cache = []
    queue = comfy.model_prefetch.make_prefetch_queue(
        list(model.layers), context.device, options
    )
    for index, block in enumerate(model.layers):
        comfy.model_prefetch.prefetch_queue_pop(
            queue, context.device, block, context.dtype
        )
        y = modulated_rms_norm(text, block.norm1, mod[1], mod[0])
        q, k, v = prepare_attention(block, y, pe)
        cache.extend((k.clone(), v.clone()))
        if index + 1 < len(model.layers):
            text = finish_block(block, text, attend(block, q, k, v, options), mod)
    comfy.model_prefetch.prefetch_queue_pop(queue, context.device, None)
    return cache


def cached_extra_conds(
    model: comfy.model_base.MOSSImage,
    *,
    text_timestep: float,
    **kwargs: Any,
) -> dict[str, Any]:
    out = comfy.model_base.MOSSImage.extra_conds(model, **kwargs)
    context = kwargs.get("cross_attn")
    if context is None:
        return out
    if kwargs.get("hooks") is not None:
        raise ValueError(
            "MOSS text KV caching requires fixed LoRA weights; conditioning hooks change them during sampling."
        )
    context = context.to(device=kwargs["device"], dtype=model.get_dtype_inference())
    cache = prepare_text_kv_cache(model.diffusion_model, context, text_timestep)
    out.pop("c_crossattn", None)
    out["moss_text_kv"] = comfy.conds.CONDList(cache)
    return out


def image_block(
    block: Block,
    x: torch.Tensor,
    mod: tuple[torch.Tensor, ...],
    pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    options: dict[str, Any],
) -> torch.Tensor:
    y = modulated_rms_norm(x, block.norm1, mod[1], mod[0])
    q, k, v = prepare_attention(block, y, pe)
    k = torch.cat((k, key), dim=1)
    v = torch.cat((v, value), dim=1)
    return finish_block(block, x, attend(block, q, k, v, options), mod)


def cached_forward(
    model: MOSSImage,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor | None = None,
    transformer_options: dict[str, Any] | None = None,
    moss_text_kv: list[torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if moss_text_kv is None:
        raise ValueError(
            "MOSS text KV conditioning is missing; sample the patched MODEL through a ComfyUI sampler."
        )
    batch, _, height, width = x.shape
    dtype, device = x.dtype, x.device
    options = {} if transformer_options is None else transformer_options.copy()
    pe = model.rope(height, width, moss_text_kv[0].shape[1], device)[
        :, : height * width
    ]
    vec = model.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    mod = (
        model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    )
    head_shift, head_scale = (
        model.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    )
    x = model.video_embedder(x.flatten(2).transpose(1, 2))
    replacements = options.get("patches_replace", {}).get("dit", {})
    options["total_blocks"] = len(model.layers)
    options["block_type"] = "single"
    queue = comfy.model_prefetch.make_prefetch_queue(
        list(model.layers), device, options
    )
    comfy.model_prefetch.malloc_graph_begin(device)
    try:
        for index, block in enumerate(model.layers):
            comfy.model_prefetch.prefetch_queue_pop(
                queue, device, block, dtype, malloc_scope="block"
            )
            options["block_index"] = index
            key, value = moss_text_kv[2 * index : 2 * index + 2]
            if ("single_block", index) in replacements:

                def original_block(args: dict[str, Any]) -> dict[str, torch.Tensor]:
                    return {
                        "img": image_block(
                            block,
                            args["img"],
                            args["mod"],
                            args["pe"],
                            key,
                            value,
                            args["transformer_options"],
                        )
                    }

                x = replacements[("single_block", index)](
                    {
                        "img": x,
                        "vec": vec,
                        "mod": mod,
                        "pe": pe,
                        "transformer_options": options,
                    },
                    {"original_block": original_block},
                )["img"]
            else:
                x = image_block(block, x, mod, pe, key, value, options)
        comfy.model_prefetch.prefetch_queue_pop(
            queue, device, None, malloc_scope="block"
        )
    finally:
        comfy.model_prefetch.malloc_graph_end()
    out = model.final_linear_video(
        modulated_rms_norm(x, model.final_norm_video, head_scale, head_shift)
    )
    return out.transpose(1, 2).reshape(batch, model.latent_channels, height, width)
