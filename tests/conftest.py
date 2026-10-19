import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("COMFYUI_ROOT", str(ROOT.parents[1])))
sys.path.insert(0, str(ROOT / "scripts"))

import comfy.cli_args  # noqa: E402 -- resolve imports from the host ComfyUI checkout

comfy.cli_args.args.cpu = True
comfy.cli_args.args.use_pytorch_cross_attention = True

spec = importlib.util.spec_from_file_location(
    "moss_kv_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
