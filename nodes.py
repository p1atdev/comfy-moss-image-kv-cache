from functools import partial
from types import MethodType

from comfy.ldm.moss_image.model import MOSSImage
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io

from .model import cached_extra_conds, cached_forward


class MOSSImageKVCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MOSSImageKVCache",
            display_name="MOSS-Image Text KV Cache",
            category="model/patches/moss_image",
            description="Cache text K/V once per sampling run using isolated text attention and a fixed text timestep. This changes joint attention behavior.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "text_timestep",
                    default=1000.0,
                    min=0.0,
                    max=1000.0,
                    step=1.0,
                    tooltip="1000 matches the trained correction LoRA.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls, model: ModelPatcher, text_timestep: float = 1000.0
    ) -> io.NodeOutput:
        transformer = model.get_model_object("diffusion_model")
        if not isinstance(transformer, MOSSImage):
            raise ValueError("This node requires a native MOSS-Image MODEL.")
        patched = model.clone()
        patched.add_object_patch(
            "extra_conds",
            partial(cached_extra_conds, model.model, text_timestep=text_timestep),
        )
        patched.add_object_patch(
            "diffusion_model._forward", MethodType(cached_forward, transformer)
        )
        return io.NodeOutput(patched)


class MOSSKVExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MOSSImageKVCache]
