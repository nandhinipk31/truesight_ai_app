import streamlit as st
import json
import os
from PIL import Image, UnidentifiedImageError
from utils.model_utils import load_model, predict_image, confidence_gauge, generate_heatmap
from utils.report_utils import generate_pdf_with_visuals

st.title("📄 Report Generator")

# === USER INPUTS ===
report_name = st.text_input("Enter Report Name (no spaces)")
font_family = st.selectbox("Select Font", ["Arial", "Times", "Courier"])

# === LOAD DETECTION HISTORY ===
try:
    with open("detection_history.json", "r") as f:
        history = json.load(f)

        # ✅ Only show image-type detections (not videos)
        image_history = [
            h for h in history
            if h["filename"].lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_history:
            st.warning("No image entries available in detection history.")
            st.stop()

        file_options = [
            f"{item['filename']} - {item['result']} ({item['confidence']}%)"
            for item in image_history
        ]
        selected = st.selectbox("Choose from Detection History", file_options)
        selected_entry = image_history[file_options.index(selected)]

except Exception as e:
    st.error("No detection history available.")
    st.stop()

# === RESOLVE IMAGE PATH ===
filename = selected_entry["filename"]
possible_paths = [
    os.path.join("assets", "sample_images", filename),
    os.path.join("outputs", filename),
    filename
]

image_path = None
for path in possible_paths:
    if os.path.exists(path):
        image_path = path
        break

if not image_path:
    st.error(f"Image file not found: {filename}")
    st.stop()

# === LOAD & SHOW IMAGE ===
try:
    image = Image.open(image_path)
    st.image(image, caption="Selected Image", use_container_width=True)
except UnidentifiedImageError:
    st.error(f"The selected file is not a valid image: {filename}")
    st.stop()

# === LOAD MODEL AND PREDICT ===
model = load_model("models/ai_detector_state_dict.pth")
result, confidence = predict_image(image, model)

# === SAVE VISUALS FOR PDF ===
confidence_gauge(confidence, save_path="outputs/confidence_gauge.png")
generate_heatmap(model, image, save_path="outputs/heatmap.png")

# === GENERATE REPORT BUTTON ===
if st.button("Generate Report"):
    if not report_name.strip():
        st.warning("⚠ Please enter a report name before generating.")
        st.stop()

    pdf_path = generate_pdf_with_visuals(
        report_name=report_name.strip(),
        result=result,
        confidence=confidence,
        image=image,
        font_family=font_family
    )

    if pdf_path and os.path.exists(pdf_path):
        st.success("✅ PDF Report Generated Successfully!")
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Report",
                f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
    else:
        st.error("❌ Failed to generate the PDF. Please try again.")
