import streamlit as st

# Set page title and layout
st.set_page_config(page_title="TrueSight AI", layout="wide")

# Inject CSS to set background image and make content background transparent
st.markdown("""
    <style>
    /* Background image for main app container */
    [data-testid="stAppViewContainer"] {
        background-image: url("assets/background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Make main content container transparent */
    [data-testid="stAppViewContainer"] > div:first-child {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("TrueSight AI")
st.sidebar.markdown("""
### Instructions:
- Choose a page from the sidebar.
- Upload or select an image/video.
- View detection results and download report.

### Contact:
📧 support@truesightai.com  
📞 +91-9876543210
""")

# Main page content
st.image("assets/logo.png", width=200)

st.title("👁️ Welcome to TrueSight AI")

st.header("⚙️ How It Works")
st.markdown("""
1. **Upload** an image or video from your system.  
2. Our advanced **AI model** classifies it as **Authentic** or **AI-Generated** content.  
3. You will receive:
    - A detailed **Prediction Result**.
    - A **Confidence Gauge** showing certainty level.
    - An **Explainable Heatmap** showing AI's focus.
    - Option to **Download a Visual PDF Report**.
""")

st.header("ℹ️ About Us")
st.markdown("""
**TrueSight AI** is built to detect manipulated or synthetic content in the digital world.

- ✅ **Trained on thousands** of real vs AI-generated examples.  
- 🧠 Powered by **deep learning (ResNet18)** and Grad-CAM.  
- 🧪 Designed for **researchers, educators, and journalists** to ensure media authenticity.

We aim to bring **transparency and trust** to AI-driven content by offering explainable and visual results.
""")
