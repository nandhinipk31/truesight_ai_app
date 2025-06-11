import streamlit as st
from utils.model_utils import load_model, predict_video, confidence_gauge
from utils.history_utils import log_detection
import os

st.title("🎥 Video Detection")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_video:
    # Save video locally
    video_path = os.path.join("outputs", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Load model and predict
    model = load_model("models/ai_detector_state_dict.pth")
    result, confidence = predict_video(video_path, model)

    # === Layout: First Row (Video on Left, Prediction Result on Right)
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.video(video_path)

    with top_col2:
        color = "#4CAF50" if result == "Authentic Content" else "#FF5722"
        st.markdown(f"""
            <div style='text-align:center; padding: 20px 0;'>
                <span style='font-size: 36px; font-weight: bold;'>Prediction:</span><br>
                <span style='font-size: 48px; font-weight: bold; color: {color};'>{result}</span>
            </div>
        """, unsafe_allow_html=True)

        # === Add quote based on confidence level
        if confidence <= 24.5:
            quote = "The model is highly uncertain about this prediction. Consider re-evaluating the content."
        elif 24.6 <= confidence <= 49.5:
            quote = "The confidence is low. Additional verification is recommended before trusting this result."
        elif 49.6 <= confidence <= 74.5:
            quote = "The model shows moderate confidence. This result is likely correct but not definitive."
        else:
            quote = "The model is highly confident. This result can be trusted with strong assurance."

        st.markdown(f"<p style='text-align:center; font-size:16px; font-style:italic;'>{quote}</p>", unsafe_allow_html=True)

    # === Layout: Second Row (Confidence Gauge on Left, Placeholder for Graph on Right)
    bottom_col1, bottom_col2 = st.columns(2)
    with bottom_col1:
        confidence_gauge(confidence)

    with bottom_col2:
        st.info("🔍 Heatmap not available for video yet.")  # Placeholder text for future video heatmap feature

    # ✅ Log Detection
    from numpy import float32
    log_detection(uploaded_video.name, result, float(confidence))