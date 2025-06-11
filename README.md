TrueSight_AI/
│
├── assets/
│   ├── logo.png
│   ├── background.jpg
│   └── sample_images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── dataset/
│   ├── real/
│   │   ├── real1.jpg
│   │   ├── ...
│   └── fake/
│       ├── fake1.jpg
│       ├── ...
│
├── models/
│   └── ai_detector_state_dict.pth
│
├── outputs/
│   ├── confidence_gauge.png
│   ├── heatmap.png
│   ├── temp_uploaded_image.jpg
│   ├── report_sample.pdf
│   └── videoplayback.mp4 (uploaded videos)
│
├── pages/
│   ├── 1_Try_Demo.py
│   ├── 2_Image_Detection.py
│   ├── 3_Video_Detection.py
│   ├── 4_Detection_History.py
│   └── 5_Report_Generator.py
│
├── utils/
│   ├── model_utils.py
│   ├── report_utils.py
│   └── history_utils.py
│
├── detection_history.json
├── train_model.py
├── README.md
├── requirements.txt
└── Home.py


## 📦 Installation

```bash
git clone https://github.com/yourusername/truesight-ai.git
cd truesight-ai
pip install -r requirements.txt
streamlit run main.py
[11-06-2025 08:37 PM] Nandyyy: # 🔍 TrueSight AI - Deepfake Detection Web App

*TrueSight AI* is a powerful and user-friendly web application built to detect *AI-generated* or *manipulated images and videos* using deep learning. It combines the accuracy of ResNet18 with intuitive visual explanations to help users validate digital content authenticity.

---

## 🚀 Features

- ✅ Upload image or video to detect authenticity
- ✅ Confidence Gauge (0–100%) for result certainty
- ✅ Grad-CAM Heatmap highlighting decision regions
- ✅ Detection History Log
- ✅ Downloadable PDF report with visuals
- ✅ Clean and responsive Streamlit UI

---

## 🧠 Model Overview

- *Model Used*: ResNet18 (pretrained on ImageNet, fine-tuned for binary classification)
- *Trained On*: Real vs. AI-generated images
- *Prediction*: Classifies input as Authentic Content or AI-Generated Content

---

## 🛠 Tech Stack

| Layer       | Technology                  |
|-------------|-----------------------------|
| Frontend    | Streamlit, HTML, CSS        |
| Backend     | Python (Torch, OpenCV)      |
| ML Model    | ResNet18 (PyTorch)          |
| Visualization | Plotly, Grad-CAM (CV2)    |
| Logging     | JSON-based history storage  |
| Output      | PDF Report (via FPDF)       |

---

## 📂 Folder Structure
