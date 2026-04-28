#!/usr/bin/env python3
"""Generate detection example images for the IEEE paper."""

import cv2
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    model_path = Path("runs/detect/output/yolo/fine_k8_more30/weights/best.pt")
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 1

    model = YOLO(str(model_path))

    sample_images = [
        ".tmp/gpr_data/GPR_data/Utilities/008.jpg",
        ".tmp/gpr_data/GPR_data/Utilities/030.jpg",
        ".tmp/gpr_data/GPR_data/Utilities/065.jpg",
    ]

    output_dir = Path(".tmp/paper_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(sample_images):
        img = Path(img_path)
        if not img.exists():
            print(f"Skip: {img} not found")
            continue

        results = model.predict(source=str(img), conf=0.25, verbose=False)
        r = results[0]

        img_cv = cv2.imread(str(img))
        annotated = r.plot(img=img_cv)

        out_path = output_dir / f"detection{i+1}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"Written: {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())