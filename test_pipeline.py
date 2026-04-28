#!/usr/bin/env python3
"""
Standalone test of the GPR app pipeline with Gradio-like inputs
"""

import sys
import traceback
from pathlib import Path
import base64
import time

# Add project to path
sys.path.insert(0, '/Users/pritamwani')

# Import app functions
import gpr_app
import cv2
import ollama
from ultralytics import YOLO


def test_pipeline(test_image_path):
    """Test the full pipeline exactly like Gradio would"""
    results = {
        "image_path": test_image_path,
        "denoised": None,
        "yolo": None,
        "gemma": None,
        "full": None,
        "errors": []
    }
    
    # Make path absolute
    if not Path(test_image_path).is_absolute():
        test_image_path = str(Path("/Users/pritamwani").resolve() / test_image_path)
    
    print(f"\n{'='*50}")
    print(f"Testing with: {test_image_path}")
    print(f"Exists: {Path(test_image_path).exists()}")
    print(f"{'='*50}\n")
    
    try:
        # Step 1: Denoise
        print("[1/4] Denoising...")
        start = time.time()
        denoised = gpr_app.denoise_image(test_image_path)
        results["denoised"] = denoised
        print(f"    ✅ Done in {time.time()-start:.1f}s: {Path(denoised).name}")
    except Exception as e:
        results["errors"].append(f"Denoise: {e}")
        print(f"    ❌ FAILED: {e}")
        return results
    
    try:
        # Step 2: YOLO
        print("[2/4] Running YOLO...")
        start = time.time()
        dets = gpr_app.run_yolo_detection(denoised)
        results["yolo"] = dets
        print(f"    ✅ Done in {time.time()-start:.1f}s: {len(dets['detections'])} detections")
        for d in dets.get("detections", []):
            print(f"        - {d['label']}: {d['confidence']:.1%}")
    except Exception as e:
        results["errors"].append(f"YOLO: {e}")
        print(f"    ❌ FAILED: {e}")
        traceback.print_exc()
    
    try:
        # Step 3: Gemma (short test)
        print("[3/4] Running Gemma 4...")
        start = time.time()
        with open(denoised, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = ollama.chat(
            model='gemma4:latest',
            messages=[{
                'role': 'user',
                'content': 'Describe this GPR B-scan briefly (2-3 sentences). Focus on what subsurface features you see.',
                'images': [img_b64],
            }],
            options={'temperature': 0.3, 'num_ctx': 2048},
        )
        results["gemma"] = response['message']['content']
        print(f"    ✅ Done in {time.time()-start:.1f}s")
        print(f"    Preview: {results['gemma'][:150]}...")
    except Exception as e:
        results["errors"].append(f"Gemma: {e}")
        print(f"    ❌ FAILED: {e}")
        traceback.print_exc()
    
    try:
        # Step 4: Full analysis
        print("[4/4] Full analysis pipeline...")
        start = time.time()
        report, img_out = gpr_app.analyze_gpr_scan(test_image_path)
        results["full"] = report
        print(f"    ✅ Done in {time.time()-start:.1f}s: {len(report)} chars")
    except Exception as e:
        results["errors"].append(f"Full: {e}")
        print(f"    ❌ FAILED: {e}")
        traceback.print_exc()
    
    return results


def test_yolo_accuracy():
    """Test YOLO accuracy on validation set"""
    print(f"\n{'='*50}")
    print("Testing YOLO accuracy on validation set")
    print(f"{'='*50}\n")
    
    val_dir = Path("/Users/pritamwani/.tmp/yolo_dataset_fine/images/val")
    if not val_dir.exists():
        print(f"❌ Val directory not found: {val_dir}")
        return
    
    # Load YOLO
    model = YOLO("/Users/pritamwani/runs/detect/output/yolo/fine_k8_more30/weights/best.pt")
    
    # Get class names
    class_names = ['cavities', 'clear_point_reflector', 'cluttered_multi_target', 
               'disturbed_zone', 'elongated_linear_target', 
               'intersecting_linear_and_point_reflector', 
               'low_snr_point_reflector', 'multiple_point_reflectors', 
               'strong_high_contrast_reflector']
    
    results = []
    val_images = list(val_dir.glob("*.jpg"))[:20]  # Test on 20 images
    
    print(f"Testing on {len(val_images)} images...\n")
    
    tp = fp = fn = 0
    for img_path in val_images:
        try:
            preds = model.predict(source=str(img_path), conf=0.25, iou=0.7, verbose=False)
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                for cls in boxes.cls.cpu().numpy():
                    results.append(class_names[int(cls)])
                    tp += 1
            else:
                fn += 1
        except Exception as e:
            print(f"Error on {img_path.name}: {e}")
    
    print(f"\nResults on {len(val_images)} images:")
    print(f"  Predictions: {tp}")
    print(f"  No prediction: {fn}")
    print(f"  Unique predicted: {set(results)}")


if __name__ == "__main__":
    # Test pipeline
    test_img = ".tmp/yolo_dataset_fine/images/train/001_aug_2.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    
    print("🚀 GPR App Standalone Test\n")
    
    # Test pipeline
    results = test_pipeline(test_img)
    
    if results["errors"]:
        print(f"\n❌ ERRORS FOUND:")
        for err in results["errors"]:
            print(f"  - {err}")
    else:
        print(f"\n✅ ALL STEPS COMPLETED!")
    
    # Test YOLO accuracy
    test_yolo_accuracy()