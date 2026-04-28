#!/usr/bin/env python3

import importlib
import platform
import sys
from typing import Optional


def _check_import(name: str) -> Optional[str]:
    try:
        importlib.import_module(name)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{name}: {e}"


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    missing = []
    for mod in ["PIL", "yaml", "lxml", "cv2", "torch", "ultralytics"]:
        err = _check_import(mod)
        if err:
            missing.append(err)

    if missing:
        print("ERROR: missing/failed imports")
        for m in missing:
            print(f"- {m}")
        return 1

    import torch

    mps = getattr(torch.backends, "mps", None)
    mps_available = bool(mps and mps.is_available())
    print(f"torch={torch.__version__}")
    print(f"mps_available={mps_available}")

    if not mps_available:
        print("WARN: torch MPS backend not available; training may fall back to CPU")

    try:
        from ultralytics import YOLO

        _ = YOLO("yolov8n.pt")
        print("ultralytics_model_load=ok")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: ultralytics model load failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
