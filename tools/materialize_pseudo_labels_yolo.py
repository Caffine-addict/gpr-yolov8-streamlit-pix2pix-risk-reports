#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class YoloBox:
    cls: int
    x_center: float
    y_center: float
    w: float
    h: float


def _read_cluster_mapping(path: Path) -> dict[int, str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in d.items():
        out[int(k)] = str(v)
    return out


def _load_pseudo_jsonl(path: Path) -> dict[tuple[str, int, int, int, int], int]:
    # Keyed by (image_path, x1,y1,x2,y2) -> cluster_id
    m: dict[tuple[str, int, int, int, int], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = str(r["image"])
        x1, y1, x2, y2 = [int(v) for v in r["bbox_xyxy"]]
        cid = int(r["cluster_id"])
        m[(img, x1, y1, x2, y2)] = cid
    return m


def _voc_boxes(xml_path: Path) -> tuple[str, int, int, list[tuple[str, tuple[int, int, int, int]]]]:
    t = ET.parse(xml_path)
    r = t.getroot()
    filename = r.findtext("filename") or ""
    w = int(r.findtext("size/width") or "0")
    h = int(r.findtext("size/height") or "0")
    objs = []
    for obj in r.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = int(float(bb.findtext("xmin") or "0"))
        y1 = int(float(bb.findtext("ymin") or "0"))
        x2 = int(float(bb.findtext("xmax") or "0"))
        y2 = int(float(bb.findtext("ymax") or "0"))
        objs.append((name, (x1, y1, x2, y2)))
    return filename, w, h, objs


def _to_yolo(b: tuple[int, int, int, int], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    xc = (float(x1) + float(x2)) / 2.0 / float(w)
    yc = (float(y1) + float(y2)) / 2.0 / float(h)
    return xc, yc, bw / float(w), bh / float(h)


def _write_labels(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for b in boxes:
            f.write(
                f"{b.cls} {b.x_center:.6f} {b.y_center:.6f} {b.w:.6f} {b.h:.6f}\n"
            )


def _write_yaml(out_dir: Path, names: list[str]) -> None:
    p = out_dir / "gpr_fine.yaml"
    p.write_text(
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
    ap = argparse.ArgumentParser(
        description="Materialize fine-grained YOLO labels using pseudo clusters for utilities"
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(".tmp/gpr_data/GPR_data"),
        help="Dataset root (default: .tmp/gpr_data/GPR_data)",
    )
    ap.add_argument(
        "--pseudo_jsonl",
        type=Path,
        default=Path("output/pseudo_labels/utilities.clusters.jsonl"),
        help="Pseudo JSONL (default: output/pseudo_labels/utilities.clusters.jsonl)",
    )
    ap.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="JSON mapping: cluster_id -> label_name (e.g., {\"0\":\"rebar\"})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(".tmp/yolo_dataset_fine"),
        help="Output YOLO dataset dir (default: .tmp/yolo_dataset_fine)",
    )
    ap.add_argument("--val", type=float, default=0.2, help="Val fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"ERROR: root not found: {args.root}")
        return 2
    if not args.pseudo_jsonl.exists():
        print(f"ERROR: pseudo_jsonl not found: {args.pseudo_jsonl}")
        return 2
    if not args.mapping.exists():
        print(f"ERROR: mapping not found: {args.mapping}")
        return 2

    mapping = _read_cluster_mapping(args.mapping)
    pseudo = _load_pseudo_jsonl(args.pseudo_jsonl)

    # Class list: cavities + mapped utility classes (sorted stable)
    util_names = sorted(set(mapping.values()))
    class_names = ["cavities"] + util_names
    class_to_id = {n: i for i, n in enumerate(class_names)}

    # Gather annotated items from both VOC sources
    items = []

    # cavities VOC
    cav_voc = args.root / "augmented_cavities" / "annotations" / "VOC_XML_format"
    cav_img_dir = args.root / "augmented_cavities"
    if cav_voc.exists():
        for xml in cav_voc.glob("*.xml"):
            try:
                fn, w, h, objs = _voc_boxes(xml)
            except Exception:
                continue
            img = (cav_img_dir / fn).resolve()
            if not img.exists() or w <= 0 or h <= 0:
                continue
            yboxes: list[YoloBox] = []
            for name, bb in objs:
                if name.lower() != "cavities":
                    continue
                xc, yc, bw, bh = _to_yolo(bb, w, h)
                yboxes.append(YoloBox(class_to_id["cavities"], xc, yc, bw, bh))
            items.append((img, yboxes))

    # utilities VOC + pseudo mapping
    util_voc = args.root / "augmented_utilities" / "annotations" / "VOC_XML_format"
    util_img_dir = args.root / "augmented_utilities"
    if util_voc.exists():
        for xml in util_voc.glob("*.xml"):
            try:
                fn, w, h, objs = _voc_boxes(xml)
            except Exception:
                continue
            img = (util_img_dir / fn).resolve()
            if not img.exists() or w <= 0 or h <= 0:
                continue
            yboxes: list[YoloBox] = []
            for name, bb in objs:
                if name.lower() != "utility":
                    continue
                key = (str(img), int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                cid = pseudo.get(key)
                if cid is None:
                    continue
                mapped_name = mapping.get(int(cid))
                if not mapped_name:
                    continue
                if mapped_name not in class_to_id:
                    continue
                xc, yc, bw, bh = _to_yolo(bb, w, h)
                yboxes.append(YoloBox(class_to_id[mapped_name], xc, yc, bw, bh))
            items.append((img, yboxes))

    # Add negative images
    negatives = []
    for neg_dir in [args.root / "intact", args.root / "augmented_intact"]:
        if not neg_dir.exists():
            continue
        for p in neg_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                negatives.append(p.resolve())
    items.extend([(p, []) for p in negatives])

    if not items:
        print("ERROR: no items collected")
        return 3

    random.seed(args.seed)
    random.shuffle(items)
    n_val = int(len(items) * float(args.val))
    val_items = items[:n_val]
    train_items = items[n_val:]

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def write_split(split: str, split_items):
        for img, yboxes in split_items:
            dst_img = out / "images" / split / img.name
            shutil.copy2(img, dst_img)
            dst_lbl = out / "labels" / split / f"{img.stem}.txt"
            _write_labels(dst_lbl, yboxes)

    write_split("train", train_items)
    write_split("val", val_items)
    _write_yaml(out, class_names)

    print(f"out={out}")
    print(f"data_yaml={out / 'gpr_fine.yaml'}")
    print(f"classes={class_names}")
    print(f"train_images={len(train_items)}")
    print(f"val_images={len(val_items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
