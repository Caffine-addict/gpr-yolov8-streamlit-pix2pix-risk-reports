#!/usr/bin/env python3
"""
GPR B-scan Analyzer - Streamlit Version
Simple and reliable UI for analyzing GPR scans
"""

import streamlit as st
import cv2
import base64
import ollama
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuration
OLLAMA_MODEL = "gemma4:latest"
YOLO_MODEL_PATH = "runs/detect/output/yolo/fine_k8_more30/weights/best.pt"

st.set_page_config(
    page_title="GPR B-scan Analyzer",
    page_icon="🔬",
    layout="wide"
)


def denoise_image(input_path: str) -> str:
    """Denoise GPR image"""
    output_path = Path(".tmp/denoised") / f"{Path(input_path).stem}.denoised.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    cv2.imwrite(str(output_path), img)
    return str(output_path)


def run_yolo_detection(image_path: str) -> dict:
    """Run YOLO detection"""
    model = YOLO(YOLO_MODEL_PATH)
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


def generate_gemma_analysis(image_path: str, detections: dict) -> str:
    """Generate Gemma analysis"""
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    det_list = detections.get("detections", [])
    if not det_list:
        detection_summary = "No objects were detected in this GPR scan."
    else:
        summary_parts = []
        for d in det_list:
            label = d.get("label", "unknown")
            conf = d.get("confidence", 0)
            summary_parts.append(f"- {label} ({conf:.1%} confidence)")
        detection_summary = "Objects detected:\n" + "\n".join(summary_parts)
    
    prompt = f"""You are a GPR (Ground Penetrating Radar) expert. Analyze this B-scan image and provide a detailed technical report.

{detection_summary}

Provide a report with:
1. Subsurface features found
2. Location and depth estimates
3. Signal quality assessment
4. Risk implications
5. Recommendations for further survey"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"temperature": 0.3}
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit UI
st.title("🔬 GPR B-scan AI Analyzer")
st.markdown("---")

# Sidebar with info
with st.sidebar:
    st.header("Configuration")
    st.info(f"**YOLO Model:** {YOLO_MODEL_PATH}")
    st.info(f"**LLM:** {OLLAMA_MODEL}")
    st.info("**Processing:** 100% local")
    
    st.markdown("---")
    st.markdown("### Classes Detected:")
    classes = [
        "cavities", "clear_point_reflector", "cluttered_multi_target",
        "disturbed_zone", "elongated_linear_target", 
        "intersecting_linear_and_point_reflector", "low_snr_point_reflector",
        "multiple_point_reflectors", "strong_high_contrast_reflector"
    ]
    for c in classes:
        st.text(f"• {c}")


# Main area
st.title("🔬 GPR B-scan AI Analyzer")
st.markdown("---")

# Initialize session state
if 'analyze_triggered' not in st.session_state:
    st.session_state.analyze_triggered = False

# Sidebar with info
with st.sidebar:
    st.header("Configuration")
    st.info(f"**YOLO Model:** {YOLO_MODEL_PATH}")
    st.info(f"**LLM:** {OLLAMA_MODEL}")
    st.info("**Processing:** 100% local")
    
    st.markdown("---")
    st.markdown("### Classes Detected:")
    classes = [
        "cavities", "clear_point_reflector", "cluttered_multi_target",
        "disturbed_zone", "elongated_linear_target", 
        "intersecting_linear_and_point_reflector", "low_snr_point_reflector",
        "multiple_point_reflectors", "strong_high_contrast_reflector"
    ]
    for c in classes:
        st.text(f"• {c}")


col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload GPR Scan")
    uploaded_file = st.file_uploader("Choose a GPR image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Save uploaded file
        temp_path = Path("/tmp/gradio/upload.jpg")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_bytes = uploaded_file.read()
        temp_path.write_bytes(file_bytes)
        
        st.image(file_bytes, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze GPR Scan", type="primary"):
            st.session_state.analyze_triggered = True

with col2:
    st.subheader("📊 Analysis Results")
    
    if st.session_state.analyze_triggered and uploaded_file:
        # Clear the trigger
        st.session_state.analyze_triggered = False
        
        with st.spinner("Processing..."):
            # Step 1: Denoise
            with st.spinner("1/3 Denoising..."):
                denoised_path = denoise_image(str(temp_path))
                st.success("✅ Denoised")
            
            # Step 2: Detect
            with st.spinner("2/3 Running YOLO detection..."):
                detections = run_yolo_detection(denoised_path)
                n_detections = len(detections.get("detections", []))
                st.success(f"✅ YOLO: {n_detections} objects found")
            
            # Step 3: Analyze
            with st.spinner("3/3 Running Gemma 4 analysis..."):
                analysis = generate_gemma_analysis(denoised_path, detections)
                st.success("✅ Analysis complete")
            
            # Display results
            st.markdown("### Detection Summary")
            if detections.get("detections"):
                for d in detections["detections"]:
                    st.markdown(f"- **{d['label']}**: {d['confidence']:.1%}")
            else:
                st.warning("No objects detected")
            
            st.markdown("---")
            st.markdown("### 🤖 AI Analysis")
            st.markdown(analysis)
            
            # Show denoised image
            st.markdown("---")
            st.markdown("### 🖼️ Denoised Scan")
            st.image(denoised_path, caption="Denoised", use_container_width=True)
    
    else:
        st.info("👆 Upload an image and click Analyze to begin.")
        st.markdown("""
        **Pipeline:**
        1. Upload GPR B-scan image
        2. Denoise (median blur + CLAHE)
        3. Object detection (YOLO fine_k8_more30)
        4. AI analysis (Gemma 4)
        """)

st.markdown("---")
st.caption("Generated by YOLO + Gemma 4 (local inference)")