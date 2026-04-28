#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply deterministic denoise/enhancement to a B-scan image")
    ap.add_argument("--inp", type=Path, required=True, help="Input image path")
    ap.add_argument("--out", type=Path, required=True, help="Output image path")
    ap.add_argument(
        "--steps",
        type=str,
        default="median,clahe",
        help="Comma-separated steps: none,median,nlmeans,clahe (default: median,clahe)",
    )
    ap.add_argument("--median_ksize", type=int, default=3, help="Median blur ksize (odd, default: 3)")
    ap.add_argument("--nl_h", type=float, default=10.0, help="NLMeans h (default: 10.0)")
    ap.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clipLimit (default: 2.0)")
    ap.add_argument("--clahe_grid", type=int, default=8, help="CLAHE tileGridSize (default: 8)")
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional JSON metadata output path",
    )
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}")
        return 2

    import cv2

    img = cv2.imread(str(args.inp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: failed to read image: {args.inp}")
        return 3

    applied = []
    steps = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    if "none" in steps:
        steps = []

    out = img
    for s in steps:
        if s == "median":
            k = args.median_ksize
            if k < 1 or k % 2 == 0:
                print("ERROR: --median_ksize must be odd and >= 1")
                return 2
            out = cv2.medianBlur(out, k)
            applied.append({"step": "median", "ksize": k})
        elif s == "nlmeans":
            out = cv2.fastNlMeansDenoising(out, None, float(args.nl_h), 7, 21)
            applied.append({"step": "nlmeans", "h": float(args.nl_h), "templateWindowSize": 7, "searchWindowSize": 21})
        elif s == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=float(args.clahe_clip),
                tileGridSize=(int(args.clahe_grid), int(args.clahe_grid)),
            )
            out = clahe.apply(out)
            applied.append({"step": "clahe", "clipLimit": float(args.clahe_clip), "tileGrid": int(args.clahe_grid)})
        else:
            print(f"ERROR: unknown step: {s}")
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.out), out)
    if not ok:
        print(f"ERROR: failed to write output image: {args.out}")
        return 4

    meta = {
        "input": str(args.inp),
        "output": str(args.out),
        "shape": [int(out.shape[0]), int(out.shape[1])],
        "steps": applied,
    }
    if args.meta is not None:
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
