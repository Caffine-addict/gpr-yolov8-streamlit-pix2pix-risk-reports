#!/usr/bin/env python3

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class VocBox:
    cls: int
    x_center: float
    y_center: float
    w: float
    h: float


def _safe_stem(p: Path) -> str:
    # Keep stem as-is; YOLO label file can contain spaces in filename if we write by path.
    # We always write labels next to copied image with the same stem.
    return p.stem


def _norm_label(name: str) -> str:
    if name.lower() == "utility":
        return "utility"
    if name.lower() == "cavities":
        return "cavities"
    return name.strip()


def _parse_voc(xml_path: Path, class_to_id: dict[str, int]) -> tuple[str, list[VocBox]]:
    t = ET.parse(xml_path)
    r = t.getroot()
    filename = r.findtext("filename")
    if not filename:
        raise ValueError("VOC missing <filename>")

    w = int(r.findtext("size/width") or "0")
    h = int(r.findtext("size/height") or "0")
    if w <= 0 or h <= 0:
        raise ValueError("VOC missing/invalid size")

    boxes: list[VocBox] = []
    for obj in r.findall("object"):
        raw = obj.findtext("name") or ""
        label = _norm_label(raw)
        if label not in class_to_id:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin") or "0")
        ymin = float(bb.findtext("ymin") or "0")
        xmax = float(bb.findtext("xmax") or "0")
        ymax = float(bb.findtext("ymax") or "0")
        xmin = max(0.0, min(xmin, w))
        xmax = max(0.0, min(xmax, w))
        ymin = max(0.0, min(ymin, h))
        ymax = max(0.0, min(ymax, h))
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        if bw <= 0 or bh <= 0:
            continue
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        nw = bw / w
        nh = bh / h
        boxes.append(VocBox(class_to_id[label], xc, yc, nw, nh))

    return filename, boxes


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_labels(dst: Path, boxes: list[VocBox]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for b in boxes:
            f.write(
                f"{b.cls} {b.x_center:.6f} {b.y_center:.6f} {b.w:.6f} {b.h:.6f}\n"
            )


def _write_dataset_yaml(out_dir: Path, names: list[str]) -> None:
    # Ultralytics YOLO data.yaml
    yaml_path = out_dir / "gpr.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Pascal VOC XML dataset to YOLO format")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(".tmp/gpr_data/GPR_data"),
        help="Dataset root (default: .tmp/gpr_data/GPR_data)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(".tmp/yolo_dataset"),
        help="Output directory (default: .tmp/yolo_dataset)",
    )
    ap.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="Validation fraction (default: 0.2)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = ap.parse_args()

    root = args.root
    out_dir = args.out
    if not root.exists():
        print(f"ERROR: root not found: {root}")
        return 2

    if not (0.0 < args.val < 1.0):
        print("ERROR: --val must be between 0 and 1")
        return 2

    # Canonical class list (normalized)
    class_names = ["cavities", "utility"]
    class_to_id = {n: i for i, n in enumerate(class_names)}

    # Collect annotated images from augmented_* folders
    annotated = []
    for ann_dir in [
        root / "augmented_cavities" / "annotations" / "VOC_XML_format",
        root / "augmented_utilities" / "annotations" / "VOC_XML_format",
    ]:
        if not ann_dir.exists():
            continue
        for xml_path in ann_dir.glob("*.xml"):
            try:
                filename, boxes = _parse_voc(xml_path, class_to_id)
            except Exception:
                continue

            # Image path is next to annotation folder's parent
            img_path = (ann_dir.parent.parent / filename).resolve()
            if not img_path.exists():
                # Some datasets keep images elsewhere; try relative to dataset root
                alt = (root / filename).resolve()
                img_path = alt if alt.exists() else img_path
            if img_path.exists():
                annotated.append((img_path, boxes))

    # Add negative images
    negatives = []
    for neg_dir in [root / "intact", root / "augmented_intact"]:
        if not neg_dir.exists():
            continue
        for p in neg_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                negatives.append(p.resolve())

    all_items = [(p, b) for (p, b) in annotated] + [(p, []) for p in negatives]
    if not all_items:
        print("ERROR: no images found to convert")
        return 3

    random.seed(args.seed)
    random.shuffle(all_items)
    n_val = int(len(all_items) * args.val)
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    # Clear output dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def write_split(split: str, items: list[tuple[Path, list[VocBox]]]):
        for src_img, boxes in items:
            stem = _safe_stem(src_img)
            dst_img = out_dir / "images" / split / src_img.name
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
            _copy_image(src_img, dst_img)
            _write_labels(dst_lbl, boxes)

    write_split("train", train_items)
    write_split("val", val_items)

    _write_dataset_yaml(out_dir, class_names)
    print(f"out={out_dir}")
    print(f"train_images={len(train_items)}")
    print(f"val_images={len(val_items)}")
    print(f"classes={class_names}")
    print(f"data_yaml={out_dir / 'gpr.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
