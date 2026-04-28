#!/usr/bin/env python3
"""
GPR B-scan Analysis - Simple CLI Version
Run directly without Gradio for testing
"""

import sys
import os
sys.path.insert(0, '/Users/pritamwani')

import argparse
import base64
import time
from pathlib import Path

import cv2
import ollama
from ultralytics import YOLO


# Configuration
OLLAMA_MODEL = "gemma4:latest"
YOLO_MODEL = "runs/detect/output/yolo/fine_k8_more30/weights/best.pt"


def denoise(input_path):
    """Denoise GPR image"""
    output_path = f".tmp/denoised/{Path(input_path).stem}.denoised.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    cv2.imwrite(output_path, img)
    return output_path


def yolo_detect(image_path):
    """Run YOLO detection"""
    model = YOLO(YOLO_MODEL)
    results = model.predict(source=image_path, conf=0.25, iou=0.7, verbose=False)
    r = results[0]
    names = r.names if hasattr(r, 'names') else {}
    
    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            detections.append({
                "label": names.get(int(box.cls), "unknown"),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })
    return {"detections": detections}


def gemma_analyze(image_path, detections):
    """Generate Gemma analysis"""
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    det_text = "\n".join([f"- {d['label']}: {d['confidence']:.1%}" 
                       for d in detections.get('detections', [])]) or "No objects detected"
    
    prompt = f"""Analyze this GPR B-scan image.

Detected objects:
{det_text}

Provide a technical report with:
1. Subsurface features found
2. Estimated depth/position
3. Signal quality
4. Risk assessment
5. Recommendations"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        options={"temperature": 0.3}
    )
    return response['message']['content']


def main():
    parser = argparse.ArgumentParser(description='GPR B-scan Analyzer')
    parser.add_argument('image', help='Path to GPR image')
    args = parser.parse_args()
    
    img_path = args.image
    if not Path(img_path).exists():
        print(f"❌ File not found: {img_path}")
        return 1
    
    print(f"🚀 Analyzing: {img_path}\n")
    
    # Step 1: Denoise
    print("[1/3] Denoising...")
    t0 = time.time()
    denoised = denoise(img_path)
    print(f"    ✅ {denoised} ({time.time()-t0:.1f}s)")
    
    # Step 2: Detect
    print("[2/3] YOLO detection...")
    t0 = time.time()
    detections = yolo_detect(denoised)
    print(f"    ✅ {len(detections['detections'])} found ({time.time()-t0:.1f}s)")
    for d in detections['detections']:
        print(f"       - {d['label']}: {d['confidence']:.1%}")
    
    # Step 3: Analyze
    print("[3/3] Gemma analysis...")
    t0 = time.time()
    analysis = gemma_analyze(denoised, detections)
    print(f"    ✅ Done ({time.time()-t0:.1f}s)\n")
    
    # Print Report
    print("=" * 50)
    print("GPR B-SCAN ANALYSIS REPORT")
    print("=" * 50)
    print(analysis)
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())