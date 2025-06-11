import streamlit as st
import os
from PIL import Image
from utils.model_utils import load_model, predict_image, confidence_gauge, generate_heatmap
from utils.history_utils import log_detection
from numpy import float32

st.title("🔍 Try a Demo")

# === Load Sample Images ===
sample_dir = "assets/sample_images"
sample_images = os.listdir(sample_dir)

# === Image Selection ===
choice = st.selectbox("Choose a sample image", sample_images)
img_path = os.path.join(sample_dir, choice)
image = Image.open(img_path)

# === Load Model and Predict ===
model = load_model("models/ai_detector_state_dict.pth")
result, confidence = predict_image(image, model)

# === Confidence-based Quote Function ===
def get_confidence_quote(confidence):
    if confidence <= 24.5:
        return "The model is highly uncertain about this prediction. Consider re-evaluating the content."
    elif 24.6 <= confidence <= 49.5:
        return "The confidence is low. Additional verification is recommended before trusting this result."
    elif 49.6 <= confidence <= 74.5:
        return "The model shows moderate confidence. This result is likely correct but not definitive."
    elif 74.6 <= confidence <= 100:
        return "The model is highly confident. This result can be trusted with strong assurance."

quote = get_confidence_quote(confidence)

# === Layout: Image (Left) | Prediction (Right) ===
top_col1, top_col2 = st.columns(2)

with top_col1:
    st.image(image, caption="Selected Image", use_container_width=True)

with top_col2:
    color = "#4CAF50" if result == "Authentic Content" else "#FF5722"
    st.markdown(f"""
        <div style='text-align:center; padding: 20px 0;'>
            <span style='font-size: 36px; font-weight: bold;'>Prediction:</span><br>
            <span style='font-size: 48px; font-weight: bold; color: {color};'>{result}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:18px; color:gray;'>{quote}</p>", unsafe_allow_html=True)

# === Layout: Gauge (Left) | Heatmap (Right) ===
bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    confidence_gauge(confidence)

with bottom_col2:
    heatmap_image = generate_heatmap(model, image)
    st.image(heatmap_image, caption="Heatmap Visualization", use_container_width=True)

# === Log Detection for Report Generator ===
log_detection(choice, result, float(confidence))
