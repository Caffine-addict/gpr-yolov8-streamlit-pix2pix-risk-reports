# GPR B-Scan Analysis Pipeline

Automated Ground Penetrating Radar (GPR) B-Scan analysis system combining YOLOv8 object detection, pseudo-labeling for fine-grained classification, and Large Language Models (Gemma 4) for technical report generation.

## Overview

This project presents a complete end-to-end pipeline for GPR B-scan analysis achieving **82.2% mAP50** on 9-class fine-grained classification. The system operates entirely locally ensuring data privacy while providing real-time inference capability.

## Technology Stack

### Core Technologies
- **YOLOv8n**: Nano-scale object detection model (3.2M parameters, 6.2MB)
- **OpenCV**: Image preprocessing (median filtering, CLAHE enhancement)
- **Streamlit**: Web interface for interactive analysis
- **Ollama**: Local LLM runtime for Gemma 4 (9.6GB model)
- **PyTorch**: Deep learning framework with MPS backend for Apple Silicon
- **Ultralytics**: YOLOv8 implementation
- **scikit-learn**: K-Means clustering for pseudo-labeling
- **ResNet18**: Feature extraction for pseudo-labeling

### Development Environment
- **Platform**: macOS (Apple M4 chip with MPS acceleration)
- **Language**: Python 3.8+
- **Package Management**: Virtual environment (venv)

## Project Structure

```
gpr-analysis/
├── gpr_streamlit.py          # Main Streamlit web application
├── gpr_app.py               # Alternative Gradio interface (deprecated)
├── tools/
│   ├── voc_to_yolo.py       # Convert Pascal VOC annotations to YOLO format
│   ├── denoise_image.py      # Image preprocessing (median + CLAHE)
│   ├── predict_yolo.py       # YOLO inference script
│   ├── pseudo_label_utilities.py  # Pseudo-labeling pipeline
│   ├── materialize_pseudo_labels_yolo.py  # Apply pseudo-labels to dataset
│   └── generate_report.py    # LLM report generation
├── generate_training_plot.py   # Training metrics visualization
├── generate_loss_plot.py      # Loss curves visualization
├── generate_detection_examples.py  # Detection examples for papers
├── generate_comparison_chart.py    # Comparative analysis charts
├── generate_architecture_diagram.py  # System architecture diagrams
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Apple M-series chip (for MPS acceleration) or CUDA-compatible GPU
- Ollama installed locally

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gpr-analysis.git
cd gpr-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull Gemma 4
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:latest
```

## Workflow

### 1. Data Preparation
```bash
# Convert VOC annotations to YOLO format
python tools/voc_to_yolo.py --input .tmp/gpr_data/GPR_data/ \\
                                    --output .tmp/yolo_dataset/

# Apply pseudo-labels for fine-grained classification
python tools/materialize_pseudo_labels_yolo.py \\
       --cluster-mapping output/pseudo_labels/cluster_mapping.json \\
       --input .tmp/yolo_dataset/ \\
       --output .tmp/yolo_dataset_fine/
```

### 2. Training
```bash
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='.tmp/yolo_dataset_fine/gpr_fine.yaml',
    epochs=30,
    imgsz=224,
    batch=32,
    device='mps',  # or 'cuda', 'cpu'
    optimizer='Adam',
    lr0=0.01,
    conf=0.25,
    iou=0.7
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
```

### 3. Inference
```bash
# Run detection on single image
python tools/predict_yolo.py \\
    --model runs/detect/output/yolo/fine_k8_more30/weights/best.pt \\
    --image .tmp/gpr_data/GPR_data/Utilities/008.jpg \\
    --conf 0.25

# Run Streamlit web interface
streamlit run gpr_streamlit.py --server.port 7860
```

### 4. Report Generation
The system uses Gemma 4 via Ollama to generate detailed technical reports:

```python
import ollama

response = ollama.chat(
    model='gemma4:latest',
    messages=[{
        'role': 'user',
        'content': """Analyze this GPR B-scan image with detected objects:
        - Image: [base64 encoded denoised image]
        - Detections: [JSON with bounding boxes, labels, confidence]
        
        Generate a technical report including:
        1. Subsurface feature classification
        2. Depth and position estimates
        3. Signal quality assessment
        4. Risk implications
        5. Recommendations"""
    }]
)
print(response['message']['content'])
```

## Pseudo-Labeling Pipeline

Fine-grained classification is achieved through unsupervised clustering:

1. **Crop Utility Regions**: Extract bounding boxes from coarse "Utility" labels
2. **Feature Extraction**: ResNet18 embeddings (512-dimensional vectors)
3. **Dimensionality Reduction**: PCA to 50 components
4. **Clustering**: K-Means with k=8
5. **Manual Labeling**: Domain experts assign semantic labels to clusters
6. **Label Regeneration**: Update YOLO labels with 9-class taxonomy

Resulting classes:
- cavities
- clear_point_reflector
- strong_high_contrast_reflector
- multiple_point_reflectors
- elongated_linear_target
- intersecting_linear_and_point_reflector
- disturbed_zone
- cluttered_multi_target
- low_snr_point_reflector

## Performance Metrics

### Best Model (Epoch 25)
| Metric | Value |
|--------|-------|
| Precision | 76.5% |
| Recall | 76.1% |
| mAP50 | **82.2%** |
| mAP50-95 | 56.9% |
| Inference Time | 0.12s/image |
| Model Size | 6.2 MB |

### Comparative Analysis
| Method | mAP50 | Inference (s) | Params (M) |
|--------|-------|--------------|------------|
| **Proposed (YOLOv8n)** | **82.2%** | **0.12** | **3.2** |
| Faster R-CNN | 77.8% | 0.35 | 41.3 |
| YOLOv3 | 71.4% | 0.22 | 61.5 |
| SSD MobileNet | 68.2% | 0.08 | 6.8 |
| RetinaNet | 75.6% | 0.28 | 36.2 |
| Custom CNN | 68.2% | 0.45 | 11.2 |

## Key Features

- **Privacy-Preserving**: All processing local, no data leaves user's machine
- **Real-Time Inference**: 0.12s per image for detection
- **Fine-Grained Classification**: 9-class taxonomy via pseudo-labeling
- **Automated Reporting**: LLM-generated technical reports
- **Interactive UI**: Streamlit web interface with drag-and-drop
- **Cross-Platform**: Runs on macOS, Linux, Windows

## Dataset

- **Original**: 285 GPR B-scan images (JPEG, 224x224)
- **Augmented**: 2,239 images using RandAugment
- **Total**: 2,524 images (80/20 train/val split)
- **Classes**: 9 fine-grained categories
- **Format**: Pascal VOC XML → YOLO txt conversion

## Training Configuration

```yaml
# YOLOv8 training config
model: yolov8n.pt
data: gpr_fine.yaml
epochs: 30
imgsz: 224
batch: 32
optimizer: Adam
lr0: 0.01
lrf: 0.0001
conf: 0.25
iou: 0.7
device: mps  # Apple Silicon GPU
```

## Results Visualization

Training metrics and detection examples are available in the `paper_images/` directory (not included in this repo - see paper for details).

## Applications

- Construction site safety assessment
- Utility mapping and buried services detection
- Archaeological investigation
- Road and bridge infrastructure inspection
- Environmental remediation

## Contributors

- **Pritam Wani** - Lead Developer
- **Sucheta Rout** - Co-Developer  
- **Prof. Sahil Pocker** - Supervisor

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work, please cite:

```bibtex
@techreport{GPR2025,
  author = {Wani, Pritam and Rout, Sucheta and Pocker, Sahil},
  title = {Automated Ground Penetrating Radar B-Scan Analysis Using Deep Learning and Large Language Models},
  institution = {Dayananda Sagar University},
  year = {2025},
  type = {B.Tech Project Report}
}
```

## Acknowledgment

We thank Dayananda Sagar University, School of Engineering for providing the computational resources and guidance throughout this project.