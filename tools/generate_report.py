#!/usr/bin/env python3

from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Allow running as a script from any working directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from backend.report_schema import (
    ReportValidationError,
    SCHEMA_VERSION,
    validate_batch_summary_v1,
    validate_image_report_v1,
)


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round1(x: Any) -> str:
    try:
        return f"{float(x):.1f}"
    except Exception:
        return "0.0"


def render_image_markdown(report: Dict[str, Any]) -> str:
    """Render Markdown from an already-built ImageReportV1 dict.

    Presentation-only rounding:
    - bbox values are rounded to 1 decimal place in tables
    - JSON remains full precision
    """
    # Validate before render to keep outputs consistent.
    validate_image_report_v1(report, eps=1e-6)

    img = report["image"]
    inf = report["inference"]
    enh = report["enhancement"]
    stats = report["stats"]
    risk = report["risk"]

    lines: List[str] = []
    lines.append("# GPR B-scan Report")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- schema_version: `{report['schema_version']}`")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append("")

    lines.append("## Image")
    lines.append(f"- image_name: `{img['image_name']}`")
    lines.append(f"- source_type: `{img['source_type']}`")
    lines.append(f"- source_path: `{img['source_path']}`")
    lines.append(f"- width_px: `{img['width_px']}`")
    lines.append(f"- height_px: `{img['height_px']}`")
    lines.append("")

    lines.append("## Inference")
    lines.append(f"- weights_path: `{inf['weights_path']}`")
    lines.append(f"- device: `{inf['device']}`")
    lines.append(f"- conf_threshold: `{inf['conf_threshold']}`")
    lines.append(f"- iou_threshold: `{inf['iou_threshold']}`")
    lines.append(f"- certain_threshold: `{inf['certain_threshold']}`")
    lines.append("")

    lines.append("## Enhancement")
    steps = enh.get("steps", []) or []
    if not steps:
        lines.append("- steps: (none)")
    else:
        lines.append("- steps:")
        for s in steps:
            lines.append(f"  - `{s.get('step')}`: `{s.get('params')}`")
    lines.append("")

    lines.append("## Detections Summary")
    lines.append(f"- detections_total: `{stats['detections_total']}`")
    lines.append(f"- certain_count (>= {inf['certain_threshold']}): `{stats['certain_count']}`")
    lines.append(f"- uncertain_count (< {inf['certain_threshold']}): `{stats['uncertain_count']}`")
    lines.append("")

    per_class = stats.get("per_class", []) or []
    if per_class:
        lines.append("## Per-Class Statistics (All Detections)")
        lines.append("| class | count | avg_conf | min_conf | max_conf |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sorted(per_class, key=lambda r: (-int(r.get("count", 0)), str(r.get("class", "")))):
            cls = _md_escape(row.get("class", "unknown"))
            lines.append(
                "| `{}` | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    cls,
                    int(row.get("count", 0)),
                    float(row.get("avg_confidence", 0.0)),
                    float(row.get("min_confidence", 0.0)),
                    float(row.get("max_confidence", 0.0)),
                )
            )
        lines.append("")

    dets = report.get("detections", []) or []
    lines.append("## Detections (All)")
    lines.append("| label | confidence | x1 | y1 | x2 | y2 | certain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    thr = float(inf["certain_threshold"])
    for d in sorted(dets, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
        label = _md_escape(str(d.get("label", "unknown")))
        conf = float(d.get("confidence", 0.0))
        x1, y1, x2, y2 = (d.get("bbox_xyxy") or [0, 0, 0, 0])
        is_certain = conf >= thr
        lines.append(
            f"| `{label}` | {conf:.4f} | {_round1(x1)} | {_round1(y1)} | {_round1(x2)} | {_round1(y2)} | {str(is_certain).lower()} |"
        )
    lines.append("")

    lines.append("## Risk")
    lines.append(f"- risk_score: `{float(risk['risk_score']):.4f}`")
    lines.append(f"- risk_level: `{risk['risk_level']}`")
    lines.append(f"- interpretation: {risk['interpretation']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _allocate_versioned_bundle(reports_dir: Path, base_stem: str, exts: Sequence[str]) -> Dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    exts = [e.lstrip(".") for e in exts]

    def paths_for_suffix(suffix: str) -> Dict[str, Path]:
        stem = f"{base_stem}{suffix}"
        return {ext: (reports_dir / f"{stem}.{ext}") for ext in exts}

    # v1 is no suffix
    candidate = paths_for_suffix("")
    if not any(p.exists() for p in candidate.values()):
        return candidate
    for i in range(2, 10_000):
        cand = paths_for_suffix(f"_v{i}")
        if not any(p.exists() for p in cand.values()):
            return cand
    raise RuntimeError("failed to allocate versioned report filenames")


def write_image_report_bundle(
    *,
    report: Dict[str, Any],
    original_rgb: Any,
    enhanced_rgb: Any,
    overlay_rgb: Any,
    reports_dir: Path = Path("output/reports"),
    base_name: Optional[str] = None,
    eps: float = 1e-6,
) -> Dict[str, Path]:
    """Write JSON/MD/PDF for a single ImageReportV1.

    Order is enforced:
    1) validate report model
    2) render+write Markdown
    3) write JSON
    4) write PDF (uses the same Markdown content)
    """
    validate_image_report_v1(report, eps=eps)
    image_name = str(report["image"]["image_name"])
    stem = f"{base_name or image_name}_report"
    out = _allocate_versioned_bundle(reports_dir, stem, ["json", "md", "pdf"])

    md = render_image_markdown(report)
    out["md"].write_text(md, encoding="utf-8")
    out["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_image_pdf(report=report, markdown_text=md, original_rgb=original_rgb, enhanced_rgb=enhanced_rgb, overlay_rgb=overlay_rgb, out_path=out["pdf"])
    return out


def _numpy_rgb_to_png_bytes(arr: Any) -> bytes:
    from io import BytesIO
    from PIL import Image

    im = Image.fromarray(arr)
    bio = BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _write_image_pdf(
    *,
    report: Dict[str, Any],
    markdown_text: str,
    original_rgb: Any,
    enhanced_rgb: Any,
    overlay_rgb: Any,
    out_path: Path,
) -> None:
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image
    from reportlab.lib import colors
    from PIL import Image as PILImage

    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("GPR B-scan Report", styles["Title"]))
    story.append(Spacer(1, 8))

    # Images (original/enhanced/overlay)
    def img_block(title: str, rgb):
        story.append(Paragraph(title, styles["Heading2"]))
        png = _numpy_rgb_to_png_bytes(rgb)
        # Scale to fit page width.
        with PILImage.open(BytesIO(png)) as im:
            w, h = im.size
        max_w = 500
        scale = min(1.0, float(max_w) / float(w))
        img_flow = Image(BytesIO(png), width=w * scale, height=h * scale)
        story.append(img_flow)
        story.append(Spacer(1, 8))

    img_block("Original", original_rgb)
    img_block("Enhanced", enhanced_rgb)
    img_block("Prediction Overlay", overlay_rgb)

    # Per-class stats table
    stats = report["stats"]
    per = stats.get("per_class", []) or []
    if per:
        story.append(Paragraph("Per-Class Statistics (All Detections)", styles["Heading2"]))
        data = [["class", "count", "avg_conf", "min_conf", "max_conf"]]
        for row in sorted(per, key=lambda r: (-int(r.get("count", 0)), str(r.get("class", "")))):
            data.append(
                [
                    str(row.get("class")),
                    str(int(row.get("count", 0))),
                    f"{float(row.get('avg_confidence', 0.0)):.4f}",
                    f"{float(row.get('min_confidence', 0.0)):.4f}",
                    f"{float(row.get('max_confidence', 0.0)):.4f}",
                ]
            )
        t = Table(data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

    # Risk section
    risk = report["risk"]
    story.append(Paragraph("Risk", styles["Heading2"]))
    story.append(Paragraph(f"Risk level: <b>{risk['risk_level']}</b>", styles["BodyText"]))
    story.append(Paragraph(f"Risk score: <b>{float(risk['risk_score']):.4f}</b>", styles["BodyText"]))
    story.append(Paragraph(f"Interpretation: {risk['interpretation']}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Markdown (audit) - exact content
    story.append(Paragraph("Markdown (Audit)", styles["Heading2"]))
    story.append(Preformatted(markdown_text, styles["Code"]))

    doc.build(story)


def write_batch_summary_bundle(
    *,
    batch_summary: Dict[str, Any],
    reports_dir: Path = Path("output/reports"),
    eps: float = 1e-6,
) -> Dict[str, Path]:
    validate_batch_summary_v1(batch_summary, eps=eps)
    out = _allocate_versioned_bundle(reports_dir, "batch_summary", ["json", "pdf"])
    out["json"].write_text(json.dumps(batch_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_batch_summary_pdf(batch_summary=batch_summary, out_path=out["pdf"])
    return out


def _write_batch_summary_pdf(*, batch_summary: Dict[str, Any], out_path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF

    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("GPR Batch Summary", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"generated_at_utc: {batch_summary['generated_at_utc']}", styles["BodyText"]))
    story.append(Spacer(1, 8))

    inf = batch_summary["inference"]
    story.append(Paragraph("Inference", styles["Heading2"]))
    story.append(
        Paragraph(
            f"weights_path={inf['weights_path']} | device={inf['device']} | conf={inf['conf_threshold']} | iou={inf['iou_threshold']} | certain_threshold={inf['certain_threshold']}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10))

    # Aggregated per-class
    story.append(Paragraph("Aggregated Per-Class Statistics", styles["Heading2"]))
    per = batch_summary["aggregated"].get("per_class", []) or []
    data = [["class", "count", "avg_conf", "min_conf", "max_conf"]]
    for row in sorted(per, key=lambda r: (-int(r.get("count", 0)), str(r.get("class", "")))):
        data.append(
            [
                str(row.get("class")),
                str(int(row.get("count", 0))),
                f"{float(row.get('avg_confidence', 0.0)):.4f}",
                f"{float(row.get('min_confidence', 0.0)):.4f}",
                f"{float(row.get('max_confidence', 0.0)):.4f}",
            ]
        )
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # Risk distribution
    story.append(Paragraph("Risk Distribution", styles["Heading2"]))
    rd = batch_summary.get("risk_distribution", []) or []
    data = [["risk_level", "count", "mean_risk_score"]]
    counts = {}
    for row in sorted(rd, key=lambda r: r.get("risk_level")):
        data.append([str(row["risk_level"]), str(int(row["count"])), f"{float(row['mean_risk_score']):.4f}"])
        counts[str(row["risk_level"])] = int(row["count"])
    t2 = Table(data, hAlign="LEFT")
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 10))

    # Lightweight bar chart (vector, no embedded images)
    total = sum(counts.get(k, 0) for k in ["LOW", "MEDIUM", "HIGH"]) or 1
    w = 500
    h = 60
    d = Drawing(w, h)
    x = 0
    palette = {"LOW": colors.HexColor("#2c7a7b"), "MEDIUM": colors.HexColor("#b7791f"), "HIGH": colors.HexColor("#c53030")}
    for lvl in ["LOW", "MEDIUM", "HIGH"]:
        c = counts.get(lvl, 0)
        bw = int(w * (c / total))
        d.add(Rect(x, 20, bw, 20, fillColor=palette[lvl], strokeColor=palette[lvl]))
        d.add(String(x + 2, 42, f"{lvl}: {c}", fontSize=8))
        x += bw
    story.append(Paragraph("Risk Level Counts (Bar)", styles["BodyText"]))
    story.append(Spacer(1, 4))
    story.append(d)
    story.append(Spacer(1, 12))

    # Per-image summary table
    story.append(Paragraph("Per-Image Summary (Sorted by Risk)", styles["Heading2"]))
    images = list(batch_summary["batch"].get("images", []) or [])
    images_sorted = sorted(images, key=lambda r: float(r.get("risk_score", 0.0)), reverse=True)
    data = [["image_name", "risk_score", "risk_level", "detections", "certain", "uncertain"]]
    for r in images_sorted:
        data.append(
            [
                str(r.get("image_name")),
                f"{float(r.get('risk_score', 0.0)):.4f}",
                str(r.get("risk_level")),
                str(int(r.get("detections_total", 0))),
                str(int(r.get("certain_count", 0))),
                str(int(r.get("uncertain_count", 0))),
            ]
        )
    t3 = Table(data, hAlign="LEFT")
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t3)

    doc.build(story)


def _coerce_detections_for_schema(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in dets:
        out.append(
            {
                "label_id": int(d.get("label_id", 0)),
                "label": str(d.get("label", "unknown")),
                "confidence": float(d.get("confidence", 0.0)),
                "bbox_xyxy": [float(x) for x in (d.get("bbox_xyxy") or [0, 0, 0, 0])],
                "bbox_units": "pixel",
                "depth_estimate": None,
            }
        )
    return out


def main() -> int:
    """CLI: generate Markdown (and optional PDF/JSON) from an existing detections JSON.

    This CLI remains a convenience wrapper. The Streamlit app calls the helpers directly.
    """
    ap = argparse.ArgumentParser(description="Generate GPR report artifacts from detection JSON")
    ap.add_argument("--image", type=Path, required=True, help="Original image")
    ap.add_argument("--enhanced", type=Path, default=None, help="Enhanced/denoised image (optional)")
    ap.add_argument("--overlay", type=Path, default=None, help="Overlay image (optional; required for PDF)")
    ap.add_argument("--detections", type=Path, required=True, help="Detections JSON")
    ap.add_argument("--reports_dir", type=Path, default=Path("output/reports"), help="Output reports directory")
    ap.add_argument("--name", type=str, default=None, help="Base image name for report filenames")
    ap.add_argument("--conf", type=float, default=0.25, help="Inference conf threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="Inference IoU threshold")
    ap.add_argument("--certain_threshold", type=float, default=0.75, help="Threshold for certain vs uncertain")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"], help="Device label (metadata only)")
    ap.add_argument("--write_pdf", action="store_true", help="Write PDF (requires --enhanced and --overlay)")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}")
        return 2
    if not args.detections.exists():
        print(f"ERROR: detections not found: {args.detections}")
        return 2

    det_payload = json.loads(args.detections.read_text(encoding="utf-8"))
    dets = _coerce_detections_for_schema(list(det_payload.get("detections", []) or []))

    # image size
    from PIL import Image

    with Image.open(args.image) as im:
        w, h = im.size

    # enhancement steps unknown in CLI unless provided; keep empty list.
    enhancement = {"steps": []}

    from backend.risk_engine import evaluate_risk
    from backend.pipeline import summarize_detections

    stats = summarize_detections({"detections": dets}, certain_threshold=float(args.certain_threshold))
    risk = evaluate_risk(dets)
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "image",
        "generated_at_utc": _now_utc_iso(),
        "image": {
            "image_name": args.name or args.image.stem,
            "source_type": "path",
            "source_path": str(args.image),
            "width_px": int(w),
            "height_px": int(h),
        },
        "inference": {
            "model_type": "yolo",
            "weights_path": str(det_payload.get("model", "")),
            "device": str(args.device),
            "conf_threshold": float(args.conf),
            "iou_threshold": float(args.iou),
            "certain_threshold": float(args.certain_threshold),
        },
        "enhancement": enhancement,
        "detections": dets,
        "stats": stats,
        "risk": risk,
    }

    # Allocate one consistent version across all outputs.
    exts = ["md", "json"] + (["pdf"] if args.write_pdf else [])
    bundle = _allocate_versioned_bundle(args.reports_dir, f"{report['image']['image_name']}_report", exts)

    # Render/write markdown first.
    md = render_image_markdown(report)
    bundle["md"].write_text(md, encoding="utf-8")
    print(str(bundle["md"]))

    # JSON next.
    bundle["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(bundle["json"]))

    # PDF last (optional).
    if args.write_pdf:
        if args.enhanced is None or args.overlay is None:
            print("ERROR: --write_pdf requires --enhanced and --overlay")
            return 2
        if not args.enhanced.exists() or not args.overlay.exists():
            print("ERROR: enhanced/overlay path missing")
            return 2
        import cv2

        orig_g = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        enh_g = cv2.imread(str(args.enhanced), cv2.IMREAD_GRAYSCALE)
        ov_bgr = cv2.imread(str(args.overlay), cv2.IMREAD_COLOR)
        if orig_g is None or enh_g is None or ov_bgr is None:
            print("ERROR: failed to read image/enhanced/overlay")
            return 2

        orig = cv2.cvtColor(orig_g, cv2.COLOR_GRAY2RGB)
        enh_img = cv2.cvtColor(enh_g, cv2.COLOR_GRAY2RGB)
        ov = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)
        _write_image_pdf(report=report, markdown_text=md, original_rgb=orig, enhanced_rgb=enh_img, overlay_rgb=ov, out_path=bundle["pdf"])
        print(str(bundle["pdf"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
