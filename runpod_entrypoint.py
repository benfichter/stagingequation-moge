from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from runpod.serverless import start


def _load_handler() -> Callable[..., Any]:
    target = os.getenv("RUNPOD_HANDLER", "").strip()
    if not target:
        raise RuntimeError("RUNPOD_HANDLER is not set.")

    if ":" in target:
        module_path, attr_path = target.split(":", 1)
    else:
        module_path, attr_path = target.rsplit(".", 1)

    module = importlib.import_module(module_path)
    handler: Any = module
    for part in attr_path.split("."):
        handler = getattr(handler, part)
    return handler


def main() -> None:
    handler = _load_handler()
    start({"handler": handler})


if __name__ == "__main__":
    main()
