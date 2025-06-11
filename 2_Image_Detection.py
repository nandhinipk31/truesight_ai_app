import streamlit as st
import os
from PIL import Image
from utils.model_utils import load_model, predict_image, confidence_gauge, generate_heatmap
from utils.history_utils import log_detection

st.title("🖼️ Image Detection")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)

    # ✅ Convert grayscale to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ✅ Save to outputs folder for PDF generation
    save_path = os.path.join("outputs", uploaded_file.name)
    image.save(save_path)

    # ✅ Load model and predict
    model = load_model("models/ai_detector_state_dict.pth")
    result, confidence = predict_image(image, model)

    # === Layout: Top Row (Image | Prediction)
    top_col1, top_col2 = st.columns(2)

    with top_col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with top_col2:
        color = "#4CAF50" if result == "Authentic Content" else "#FF5722"
        st.markdown(f"""
            <div style='text-align:center; padding: 20px 0;'>
                <span style='font-size: 36px; font-weight: bold;'>Prediction:</span><br>
                <span style='font-size: 48px; font-weight: bold; color: {color};'>{result}</span>
            </div>
        """, unsafe_allow_html=True)

        # ✅ Add quote below prediction
        if confidence <= 24.5:
            quote = "The model is highly uncertain about this prediction. Consider re-evaluating the content."
        elif 24.6 <= confidence <= 49.5:
            quote = "The confidence is low. Additional verification is recommended before trusting this result."
        elif 49.6 <= confidence <= 74.5:
            quote = "The model shows moderate confidence. This result is likely correct but not definitive."
        else:
            quote = "The model is highly confident. This result can be trusted with strong assurance."

        st.markdown(f"<p style='text-align:center; font-size:16px; font-style:italic;'>{quote}</p>", unsafe_allow_html=True)

    # === Layout: Bottom Row (Gauge | Heatmap)
    bottom_col1, bottom_col2 = st.columns(2)

    with bottom_col1:
        confidence_gauge(confidence)

    with bottom_col2:
        heatmap_image = generate_heatmap(model, image)
        st.image(heatmap_image, caption="Heatmap Visualization", use_container_width=True)

    # ✅ Log the detection
    from numpy import float32
    log_detection(uploaded_file.name, result, float(confidence))
