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
class CropItem:
    image_path: Path
    xml_path: Path
    bbox_xyxy: tuple[int, int, int, int]


def _iter_voc_xmls(voc_dir: Path):
    for p in sorted(voc_dir.glob("*.xml")):
        yield p


def _parse_utility_boxes(xml_path: Path) -> tuple[str, list[tuple[int, int, int, int]]]:
    t = ET.parse(xml_path)
    r = t.getroot()
    filename = r.findtext("filename")
    if not filename:
        raise ValueError("VOC missing <filename>")

    w = int(r.findtext("size/width") or "0")
    h = int(r.findtext("size/height") or "0")
    if w <= 0 or h <= 0:
        raise ValueError("VOC missing/invalid size")

    boxes = []
    for obj in r.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name.lower() != "utility":
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = int(float(bb.findtext("xmin") or "0"))
        ymin = int(float(bb.findtext("ymin") or "0"))
        xmax = int(float(bb.findtext("xmax") or "0"))
        ymax = int(float(bb.findtext("ymax") or "0"))
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(1, min(xmax, w))
        ymax = max(1, min(ymax, h))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((xmin, ymin, xmax, ymax))

    return filename, boxes


def _load_crop_gray(image_path: Path, bbox: tuple[int, int, int, int], out_size: int):
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"failed to read image: {image_path}")
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("empty crop")
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def _embed_batch(crops_gray, device: str):
    import numpy as np
    import torch
    import torchvision

    # (N, H, W) uint8 -> (N, 3, H, W) float32 normalized
    x = np.stack(crops_gray, axis=0).astype("float32") / 255.0
    x = np.repeat(x[:, None, :, :], 3, axis=1)
    t = torch.from_numpy(x)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    t = (t - mean) / std

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    backbone.eval()

    dev = torch.device(device)
    backbone.to(dev)
    t = t.to(dev)

    with torch.no_grad():
        y = backbone(t)  # (N, 512, 1, 1)
    emb = y.flatten(1).detach().cpu().numpy()
    return emb


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create pseudo-label clusters for Utility using VOC boxes + ResNet18 embeddings"
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(".tmp/gpr_data/GPR_data"),
        help="Dataset root (default: .tmp/gpr_data/GPR_data)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of clusters (default: 3)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output/pseudo_labels"),
        help="Output directory (default: output/pseudo_labels)",
    )
    ap.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="Crop resize (square) (default: 128)",
    )
    ap.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Limit number of crops (0 = no limit)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="mps",
        help="torch device (default: mps)",
    )
    ap.add_argument(
        "--preview_per_cluster",
        type=int,
        default=40,
        help="Number of crop previews per cluster (default: 40)",
    )
    args = ap.parse_args()

    if args.k < 2:
        print("ERROR: --k must be >= 2")
        return 2

    voc_dir = args.root / "augmented_utilities" / "annotations" / "VOC_XML_format"
    img_dir = args.root / "augmented_utilities"
    if not voc_dir.exists() or not img_dir.exists():
        print("ERROR: expected augmented_utilities VOC dataset not found")
        print(f"- voc_dir={voc_dir}")
        print(f"- img_dir={img_dir}")
        return 2

    items: list[CropItem] = []
    for xml_path in _iter_voc_xmls(voc_dir):
        try:
            filename, boxes = _parse_utility_boxes(xml_path)
        except Exception:
            continue
        image_path = (img_dir / filename).resolve()
        if not image_path.exists():
            continue
        for b in boxes:
            items.append(CropItem(image_path=image_path, xml_path=xml_path, bbox_xyxy=b))

    if not items:
        print("ERROR: no Utility boxes found")
        return 3

    random.seed(args.seed)
    random.shuffle(items)
    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    # Load crops
    crops = []
    kept = []
    for it in items:
        try:
            crop = _load_crop_gray(it.image_path, it.bbox_xyxy, args.crop_size)
        except Exception:
            continue
        crops.append(crop)
        kept.append(it)

    if not kept:
        print("ERROR: failed to load any crops")
        return 3

    # Embed (single batch to keep deterministic; dataset is small enough)
    emb = _embed_batch(crops, args.device)

    # Cluster (sanitize numeric issues first)
    import numpy as np
    import warnings
    import importlib

    PCA = importlib.import_module("sklearn.decomposition").PCA
    KMeans = importlib.import_module("sklearn.cluster").KMeans
    StandardScaler = importlib.import_module("sklearn.preprocessing").StandardScaler

    emb = emb.astype("float64", copy=False)
    if not np.isfinite(emb).all():
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    emb = StandardScaler(with_mean=True, with_std=True).fit_transform(emb)
    if not np.isfinite(emb).all():
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    z = None
    cluster_ids = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pca = PCA(
                n_components=min(50, emb.shape[1]),
                random_state=args.seed,
                svd_solver="full",
            )
            z = pca.fit_transform(emb)

            if not np.isfinite(z).all():
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
            cluster_ids = km.fit_predict(z)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: clustering exception: {e}")
        return 3

    if z is None or cluster_ids is None:
        print("ERROR: clustering failed")
        return 3

    # Outputs
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "utilities.clusters.jsonl"
    previews_dir = out_dir / "cluster_previews"
    if previews_dir.exists():
        shutil.rmtree(previews_dir)
    previews_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it, cid in zip(kept, cluster_ids):
            rec = {
                "image": str(it.image_path),
                "annotation": str(it.xml_path),
                "bbox_xyxy": list(map(int, it.bbox_xyxy)),
                "cluster_id": int(cid),
                "label_source": "pseudo",
                "method": "resnet18_embed+pca+kmeans",
                "k": int(args.k),
            }
            f.write(json.dumps(rec) + "\n")

    # Save preview crops
    import cv2

    per_cluster = {i: 0 for i in range(args.k)}
    for it, cid in zip(kept, cluster_ids):
        if per_cluster[int(cid)] >= args.preview_per_cluster:
            continue
        try:
            crop = _load_crop_gray(it.image_path, it.bbox_xyxy, args.crop_size)
        except Exception:
            continue
        dst = previews_dir / f"cluster_{int(cid)}"
        dst.mkdir(parents=True, exist_ok=True)
        out_name = f"{it.image_path.stem}__{it.bbox_xyxy[0]}_{it.bbox_xyxy[1]}_{it.bbox_xyxy[2]}_{it.bbox_xyxy[3]}.jpg"
        cv2.imwrite(str(dst / out_name), crop)
        per_cluster[int(cid)] += 1

    # Quick stats
    counts = {i: 0 for i in range(args.k)}
    for cid in cluster_ids:
        counts[int(cid)] += 1

    print(f"jsonl={jsonl_path}")
    print(f"previews={previews_dir}")
    print(f"counts={counts}")
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Utility Pseudo-Labels (Clusters)",
                "",
                "This output is generated via feature extraction + clustering.",
                "You must map cluster IDs to semantic labels (e.g. rebar/steel/pipeline).",
                "",
                f"- JSONL: `{jsonl_path}`",
                f"- Previews: `{previews_dir}`",
                f"- k: `{args.k}`",
                f"- counts: `{counts}`",
                "",
                "Next: inspect `cluster_previews/cluster_0`, `cluster_1`, `cluster_2` and tell me which is which.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"readme={readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
