#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLO inference on a single image")
    ap.add_argument("--model", type=Path, required=True, help="Path to YOLO .pt weights")
    ap.add_argument("--image", type=Path, required=True, help="Input image")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default: 0.7)")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(".tmp/predictions"),
        help="Output directory (default: .tmp/predictions)",
    )
    args = ap.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        return 2
    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}")
        return 2

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.image),
        conf=float(args.conf),
        iou=float(args.iou),
        verbose=False,
    )
    r = results[0]

    names = getattr(r, "names", {}) or {}
    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy().tolist()
        confs = b.conf.cpu().numpy().tolist()
        clss = b.cls.cpu().numpy().tolist()
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            cls_i = int(cls)
            detections.append(
                {
                    "label_id": cls_i,
                    "label": str(names.get(cls_i, cls_i)),
                    "confidence": float(c),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    json_path = args.out_dir / f"{stem}.detections.json"
    overlay_path = args.out_dir / f"{stem}.overlay.jpg"

    payload = {
        "image": str(args.image),
        "model": str(args.model),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "detections": detections,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        import cv2

        overlay = r.plot()  # BGR numpy array
        cv2.imwrite(str(overlay_path), overlay)
    except Exception:
        overlay_path = None

    print(str(json_path))
    if overlay_path:
        print(str(overlay_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
