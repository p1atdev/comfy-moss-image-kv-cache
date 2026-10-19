from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nodes import MOSSKVExtension


async def comfy_entrypoint() -> MOSSKVExtension:
    from .nodes import MOSSKVExtension

    return MOSSKVExtension()
