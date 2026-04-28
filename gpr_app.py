#!/usr/bin/env python3
"""
GPR B-scan Analysis App with Gradio UI
Uses YOLO for detection + Gemma 4 (via Ollama) for detailed report generation
"""

import base64
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import ollama
import requests
import torch
from ultralytics import YOLO


# Configuration
OLLAMA_MODEL = "gemma4:latest"
YOLO_MODEL_PATH = "runs/detect/output/yolo/fine_k8_more30/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25


# Initialize YOLO model (lazy load)
_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for Ollama API"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def denoise_image(input_path: str, output_dir: str = ".tmp/denoised") -> str:
    """Denoise a GPR B-scan image"""
    output_path = Path(output_dir) / f"{Path(input_path).stem}.denoised.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    
    # Median blur
    img = cv2.medianBlur(img, 3)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    cv2.imwrite(str(output_path), img)
    return str(output_path)


def run_yolo_detection(image_path: str) -> dict:
    """Run YOLO detection on image"""
    model = get_yolo_model()
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, iou=0.7, verbose=False)
    r = results[0]
    names = getattr(r, "names", {}) or {}
    
    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy().tolist()
        confs = b.conf.cpu().numpy().tolist()
        clss = b.cls.cpu().numpy().tolist()
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            cls_i = int(cls)
            detections.append({
                "label_id": cls_i,
                "label": str(names.get(cls_i, cls_i)),
                "confidence": float(c),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return {"detections": detections}


def generate_gemma_analysis(image_path: str, detections: dict) -> str:
    """Use Gemma 4 to generate detailed analysis of the GPR scan"""
    image_b64 = encode_image_base64(image_path)
    
    # Build detection summary
    det_list = detections.get("detections", [])
    if not det_list:
        detection_summary = "No objects were detected in this GPR scan."
    else:
        summary_parts = []
        for d in det_list:
            label = d.get("label", "unknown")
            conf = d.get("confidence", 0)
            bbox = d.get("bbox_xyxy", [0, 0, 0, 0])
            summary_parts.append(
                f"- {label} (confidence: {conf:.2%}, bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}])"
            )
        detection_summary = "Objects detected:\n" + "\n".join(summary_parts)

    prompt = f"""You are a Ground Penetrating Radar (GPR) expert analyst. Analyze this GPR B-scan image and provide a detailed report.

{detection_summary}

Please provide a comprehensive analysis including:
1. **Object Classification**: What type of subsurface features are present? (e.g., rebar, pipeline, cavity, utilities, etc.)
2. **Location & Depth**: Estimated positions and depths based on hyperbola positions in the scan
3. **Signal Quality**: Assessment of the GPR signal (strength, clarity, noise level)
4. **Anomaly Description**: Detailed description of any detected reflectors, diffractions, or hyperbolas
5. **Risk Assessment**: Based on findings, what are the implications for construction or survey work?
6. **Recommendations**: Any follow-up actions or additional survey recommendations?

Provide your response as a well-structured technical report with clear sections."""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            options={"temperature": 0.3, "num_ctx": 4096},
        )
        return response["message"]["content"]
    except Exception as e:
        # Fallback to generate API if chat fails
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],
                        }
                    ],
                    "temperature": 0.3,
                },
                timeout=180,
            )
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
        except Exception as e2:
            return f"Error generating analysis: {e}\nFallback error: {e2}"


def analyze_gpr_scan(image_path: str) -> tuple[str, str]:
    """
    Main analysis pipeline:
    1. Denoise the GPR image
    2. Run YOLO detection
    3. Generate Gemma 4 analysis
    Returns: (result_text, overlay_image_path)
    """
    # Step 1: Denoise
    denoised_path = denoise_image(image_path)
    print(f"✅ Denoised: {denoised_path}")
    
    # Step 2: Detect
    detection_result = run_yolo_detection(denoised_path)
    detections = detection_result.get("detections", [])
    print(f"✅ YOLO detection: {len(detections)} objects found")
    
    # Step 3: Generate Gemma analysis
    print(f"🤖 Running Gemma 4 analysis...")
    analysis = generate_gemma_analysis(denoised_path, detection_result)
    print(f"✅ Gemma analysis complete")
    
    # Build result summary
    if detections:
        summary = f"**Detections ({len(detections)} found):**\n\n"
        for d in detections:
            label = d.get("label", "unknown")
            conf = d.get("confidence", 0)
            summary += f"- **{label}**: {conf:.1%} confidence\n"
    else:
        summary = "**No objects detected** in this scan.\n\n"
    
    result = f"## GPR B-scan Analysis Report\n\n"
    result += f"### Detection Summary\n{summary}\n"
    result += f"### Detailed AI Analysis\n\n{analysis}\n\n"
    result += f"---\n*Generated by YOLO (fine_k8_more30) + Gemma 4 (local inference)*\n"
    
    return result, denoised_path


def gr_interface(image, progress=gr.Progress()):
    """
    Gradio interface function.
    Handles PIL Image objects directly.
    """
    import traceback
    import numpy as np
    import cv2
    
    print(f"[DEBUG] gr_interface called with image type: {type(image)}")
    
    if image is None:
        return "⚠️ Please upload a GPR B-scan image.", None
    
    try:
        # Step 1: Get the image path - Gradio PIL mode passes numpy array directly!
        img_path = None
        
        # Case 1: numpy array (PIL Image as numpy)
        if isinstance(image, np.ndarray):
            # Save the array to a temp file
            temp_dir = Path("/tmp/gradio")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / "upload.jpg"
            cv2.imwrite(str(temp_path), image)
            img_path = str(temp_path)
            print(f"[DEBUG] Saved numpy array to: {img_path}")
        
        # Case 2: Already a string path that exists
        elif isinstance(image, str):
            if Path(image).exists():
                img_path = str(Path(image).resolve())
            elif (temp_dir / image).exists():
                img_path = str((temp_dir / image).resolve())
        
        # Case 3: Path-like object (has .name)
        elif hasattr(image, 'name'):
            img_path = str(Path(image.name).resolve())
        
        # Case 4: Dict with file info
        elif isinstance(image, dict):
            img_path = image.get('name') or image.get('path')
            if img_path:
                img_path = str(Path(img_path).resolve())
        
        # Verify we have a valid path
        if not img_path or not Path(img_path).exists():
            debug_info = f"Could not resolve uploaded image.\nType: {type(image)}"
            print(f"[DEBUG] {debug_info}")
            return f"⚠️ {debug_info}", None
        
        print(f"[DEBUG] Using image path: {img_path}")
        
        progress(0.1, desc="Denoising image...")
        denoised = denoise_image(img_path)
        print(f"[DEBUG] Denoised: {denoised}")
        
        progress(0.4, desc="Running YOLO detection...")
        detection_result = run_yolo_detection(denoised)
        print(f"[DEBUG] YOLO: {len(detection_result.get('detections', []))} detections")
        
        progress(0.6, desc="Running Gemma 4 analysis... (this may take a moment)")
        analysis = generate_gemma_analysis(denoised, detection_result)
        print(f"[DEBUG] Gemma: {len(analysis)} chars")
        
        # Build result summary
        detections = detection_result.get("detections", [])
        if detections:
            summary = f"**Detections ({len(detections)} found):**\n\n"
            for d in detections:
                label = d.get("label", "unknown")
                conf = d.get("confidence", 0)
                summary += f"- **{label}**: {conf:.1%} confidence\n"
        else:
            summary = "**No objects detected** in this scan.\n\n"
        
        result = f"## 📊 GPR B-scan Analysis Report\n\n"
        result += f"### Detection Summary\n{summary}\n"
        result += f"### 🤖 Detailed AI Analysis\n\n{analysis}\n\n"
        result += f"---\n*Generated by YOLO (fine_k8_more30) + Gemma 4 (local inference)*\n"
        
        progress(1.0, desc="Complete!")
        return result, denoised
        
    except Exception as e:
        err_detail = traceback.format_exc()
        print(f"[ERROR] gr_interface: {e}\n{err_detail}")
        err_msg = f"⚠️ Error:\n\n**{type(e).__name__}**: {str(e)}\n\n```\n{err_detail[:800]}\n```"
        return err_msg, None


# Build Gradio UI
with gr.Blocks(title="GPR B-scan Analyzer", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """# 🔬 GPR B-scan AI Analyzer
    
    **Pipeline:** Upload Image → Denoise → YOLO Detection → Gemma 4 Analysis
    
    ---
    """,
        elem_id="header"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Upload GPR Scan", 
                type="pil",
                height=350,
                elem_id="upload_input"
            )
            analyze_btn = gr.Button("🔍 Analyze GPR Scan", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output_text = gr.Markdown(
                """## 📊 Analysis Results

                Upload a GPR B-scan image and click **Analyze** to begin.
                
                Processing includes:
                1. Denoising (median + CLAHE)
                2. Object detection (fine_k8_more30 model)
                3. AI analysis (Gemma 4)
                """,
                label="Analysis Report"
            )
    
    with gr.Row():
        with gr.Column():
            denoised_output = gr.Image(
                label="🖼️ Denoised Scan",
                type="filepath",
                height=280
            )
    
    # Status indicator
    status_text = gr.Textbox(label="Status", visible=False)
    
    analyze_btn.click(
        fn=gr_interface,
        inputs=[input_image],
        outputs=[output_text, denoised_output],
    )
    
    gr.Markdown(
        "---\n"
        "### About\n"
        "- **YOLO Model**: fine_k8_more30 (30-epoch trained on MPS)\n"
        "- **LLM**: Gemma 4:latest (via Ollama, local inference)\n"
        "- **Processing**: 100% local - no data leaves your machine"
    )

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting GPR B-scan Analyzer...")
    print(f"   - YOLO model: {YOLO_MODEL_PATH}")
    print(f"   - LLM: {OLLAMA_MODEL}")
    print(f"   - Ollama API: http://localhost:11434")
    
    # Check YOLO model exists
    if not Path(YOLO_MODEL_PATH).exists():
        print(f"⚠️  Warning: YOLO model not found at {YOLO_MODEL_PATH}")
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"✅ Ollama available: {model_names}")
    except Exception as e:
        print(f"⚠️  Cannot connect to Ollama: {e}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )