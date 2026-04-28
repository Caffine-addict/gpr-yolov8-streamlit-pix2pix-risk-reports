#!/usr/bin/env python3

"""Lightweight smoke tests for the GPR Streamlit pipeline.

This is not a full unit-test suite. It is intended to quickly verify:
- imports and syntax
- enhancement + inference can run
- strict report schema validation works
- report artifacts can be generated
- bbox strict validation fails as expected

Run from an environment that has torch/ultralytics/opencv/reportlab installed, e.g.:

  /Users/pritamwani/.venv/bin/python smoke_tests.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_imports() -> None:
    import backend.pipeline  # noqa: F401
    import backend.risk_engine  # noqa: F401
    import backend.report_schema  # noqa: F401
    import backend.gan_augmentor  # noqa: F401
    import tools.generate_report  # noqa: F401
    import frontend.streamlit_app  # noqa: F401


def test_syntax_compile() -> None:
    import py_compile

    for p in [
        *ROOT.glob("backend/*.py"),
        *ROOT.glob("frontend/*.py"),
        ROOT / "tools" / "generate_report.py",
        ROOT / "tools" / "predict_yolo.py",
        ROOT / "tools" / "denoise_image.py",
    ]:
        py_compile.compile(str(p), doraise=True)


def test_end_to_end_reports() -> None:
    from backend.pipeline import (
        InferenceParams,
        build_batch_summary_model,
        build_image_report_model,
        decode_image_gray,
        gray_to_rgb,
        run_enhancement,
        run_inference,
        summarize_detections,
    )
    from backend.risk_engine import evaluate_risk
    from backend.report_schema import ReportValidationError, validate_batch_summary_v1, validate_image_report_v1
    from tools.generate_report import write_batch_summary_bundle, write_image_report_bundle

    # Prefer existing sample images if present.
    candidates = [
        Path(".tmp/yolo_dataset/images/val/001_aug_3.jpg"),
        Path(".tmp/yolo_dataset_fine/images/val/004_aug_2.jpg"),
    ]
    imgs = [p for p in candidates if p.exists()]
    _assert(len(imgs) >= 1, "No sample images found under .tmp/yolo_dataset*/images/val")

    params = InferenceParams(conf_threshold=0.25, iou_threshold=0.7, certain_threshold=0.75, device="mps")

    per_reports = []
    for img in imgs[:2]:
        b = img.read_bytes()
        orig_g = decode_image_gray(b)
        h, w = orig_g.shape
        orig_rgb = gray_to_rgb(orig_g)

        enh_g, enh_meta = run_enhancement(b)
        enh_rgb = gray_to_rgb(enh_g)

        det_payload, overlay_rgb = run_inference(enh_g, params=params)
        det_payload["image"] = img.stem

        stats = summarize_detections(det_payload, certain_threshold=params.certain_threshold)
        risk = evaluate_risk(det_payload["detections"])

        report = build_image_report_model(
            image_name=img.stem,
            source_type="path",
            source_path=str(img),
            width_px=w,
            height_px=h,
            params=params,
            enhancement_meta=enh_meta,
            detections_payload=det_payload,
            stats=stats,
            risk=risk,
        )
        validate_image_report_v1(report)
        paths = write_image_report_bundle(
            report=report,
            original_rgb=orig_rgb,
            enhanced_rgb=enh_rgb,
            overlay_rgb=overlay_rgb,
            base_name=img.stem,
        )
        per_reports.append((report, paths))

    # Build batch summary only if we have >=2
    if len(per_reports) >= 2:
        pooled = {}
        levels = ["LOW", "MEDIUM", "HIGH"]
        by = {k: [] for k in levels}
        per_rows = []
        for report, paths in per_reports:
            for d in report["detections"]:
                pooled.setdefault(d["label"], []).append(float(d["confidence"]))
            by[report["risk"]["risk_level"]].append(float(report["risk"]["risk_score"]))
            per_rows.append(
                {
                    "image_name": report["image"]["image_name"],
                    "report_json_path": str(paths["json"]),
                    "risk_score": float(report["risk"]["risk_score"]),
                    "risk_level": str(report["risk"]["risk_level"]),
                    "detections_total": int(report["stats"]["detections_total"]),
                    "certain_count": int(report["stats"]["certain_count"]),
                    "uncertain_count": int(report["stats"]["uncertain_count"]),
                }
            )

        agg = []
        for cls in sorted(pooled):
            vals = pooled[cls]
            agg.append(
                {
                    "class": cls,
                    "count": len(vals),
                    "avg_confidence": sum(vals) / len(vals),
                    "min_confidence": min(vals),
                    "max_confidence": max(vals),
                }
            )

        rd = []
        for lvl in levels:
            scores = by[lvl]
            rd.append(
                {
                    "risk_level": lvl,
                    "count": len(scores),
                    "mean_risk_score": (sum(scores) / len(scores)) if scores else 0.0,
                }
            )

        batch = build_batch_summary_model(
            per_image_rows=per_rows,
            aggregated_per_class=agg,
            risk_distribution=rd,
            params=params,
        )
        validate_batch_summary_v1(batch)
        _ = write_batch_summary_bundle(batch_summary=batch)

    # Negative bbox validation should fail.
    from backend.report_schema import validate_image_report_v1 as v

    bad = dict(per_reports[0][0])
    bad["image"] = dict(bad["image"], width_px=10, height_px=10)
    try:
        v(bad)
        raise AssertionError("Expected bbox validation failure")
    except ReportValidationError as e:
        _assert(e.code in ("E_BBOX_OOB", "E_BBOX_ORDER"), f"Unexpected code: {e.code}")


def main() -> int:
    tests = [
        ("imports", test_imports),
        ("syntax", test_syntax_compile),
        ("e2e", test_end_to_end_reports),
    ]

    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {e}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
