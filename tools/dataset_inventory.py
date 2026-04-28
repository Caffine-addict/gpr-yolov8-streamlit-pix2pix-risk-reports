#!/usr/bin/env python3

import argparse
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def _iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _iter_voc_xmls(root: Path):
    for p in root.rglob("*.xml"):
        if "VOC_XML_format" in str(p):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory GPR image+VOC dataset")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(".tmp/gpr_data/GPR_data"),
        help="Dataset root (default: .tmp/gpr_data/GPR_data)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=50,
        help="How many images to sample for size/mode (default: 50)",
    )
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        print(f"ERROR: root not found: {root}")
        return 2

    images = list(_iter_images(root))
    xmls = list(_iter_voc_xmls(root))
    print(f"root={root}")
    print(f"image_count={len(images)}")
    print(f"voc_xml_count={len(xmls)}")

    label_counts: Counter[str] = Counter()
    parse_errors = 0
    for x in xmls:
        try:
            t = ET.parse(x)
            for obj in t.getroot().findall("object"):
                name = obj.findtext("name")
                if name:
                    label_counts[name] += 1
        except Exception:
            parse_errors += 1

    if parse_errors:
        print(f"voc_parse_errors={parse_errors}")

    if label_counts:
        print("labels=")
        for k, v in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"- {k}: {v}")

    # Sample image sizes
    try:
        from PIL import Image

        sizes: Counter[tuple[int, int]] = Counter()
        modes: Counter[str] = Counter()
        for p in images[: max(0, args.sample)]:
            try:
                im = Image.open(p)
                sizes[im.size] += 1
                modes[im.mode] += 1
            except Exception:
                pass
        if sizes:
            print("sample_sizes=")
            for sz, c in sizes.most_common(10):
                print(f"- {sz}: {c}")
        if modes:
            print("sample_modes=")
            for m, c in modes.most_common(10):
                print(f"- {m}: {c}")
    except Exception as e:
        print(f"WARN: pillow not available for sampling: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
