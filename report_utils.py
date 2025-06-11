from fpdf import FPDF
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio

# Step 1: Create the confidence gauge chart
def create_confidence_gauge_image(confidence, save_path="outputs/confidence_gauge.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence Meter"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lime"},
                {'range': [80, 100], 'color': "green"},
            ],
        }
    ))

    try:
        fig.write_image(save_path, format="png", width=800, height=400, scale=2, engine="kaleido")
        print(f"✅ Gauge saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to save gauge: {e}")

# Step 2: Get quote based on confidence
def get_confidence_quote(confidence):
    if confidence <= 24.5:
        return "The model is highly uncertain about this prediction. Consider re-evaluating the content."
    elif 24.6 <= confidence <= 49.5:
        return "The confidence is low. Additional verification is recommended before trusting this result."
    elif 49.6 <= confidence <= 74.5:
        return "The model shows moderate confidence. This result is likely correct but not definitive."
    elif 74.6 <= confidence <= 100:
        return "The model is highly confident. This result can be trusted with strong assurance."
    return ""

# Step 3: Generate the PDF report
def generate_pdf_with_visuals(report_name, result, confidence, image, font_family="Arial"):
    os.makedirs("outputs", exist_ok=True)

    # Save visuals
    create_confidence_gauge_image(confidence)

    gauge_path = "outputs/confidence_gauge.png"
    heatmap_path = "outputs/heatmap.png"
    uploaded_image_path = "outputs/temp_uploaded_image.jpg"
    image.save(uploaded_image_path)

    quote = get_confidence_quote(confidence)

    pdf = FPDF()
    pdf.add_page()

    # === LOGO ===
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=10, w=30)

    # === TITLE ===
    pdf.set_font(font_family, 'B', 22)
    pdf.set_xy(0, 15)
    pdf.cell(210, 10, "TrueSight AI Detection Report", align="C", ln=True)
    pdf.ln(20)

    # === UPLOADED IMAGE ===
    pdf.set_font(font_family, 'B', 14)
    pdf.set_xy(15, 45)
    pdf.cell(80, 10, txt="Uploaded Image", ln=True)
    pdf.image(uploaded_image_path, x=15, y=55, w=80, h=80)

    # === PREDICTION RESULT ===
    pdf.set_xy(110, 45)
    pdf.set_font(font_family, 'B', 14)
    pdf.cell(80, 10, txt="Prediction Result", ln=True)

    pdf.set_xy(110, 60)
    pdf.set_font(font_family, '', 12)
    pdf.multi_cell(80, 10, txt=f"{result}\n\nConfidence: {confidence:.2f}%", align='L')

    # === QUOTE ===
    pdf.set_xy(110, 100)
    pdf.set_font(font_family, 'I', 11)
    pdf.multi_cell(80, 6, txt=f"Note: {quote}", align='L')

    # === CONFIDENCE GAUGE ===
    if os.path.exists(gauge_path):
        pdf.set_xy(15, 145)
        pdf.set_font(font_family, 'B', 13)
        pdf.cell(80, 10, txt="Confidence Gauge", ln=True)
        pdf.image(gauge_path, x=15, y=155, w=80, h=60)
    else:
        print("❌ Gauge image missing:", gauge_path)

    # === HEATMAP ===
    if os.path.exists(heatmap_path):
        pdf.set_xy(110, 145)
        pdf.set_font(font_family, 'B', 13)
        pdf.cell(80, 10, txt="Heatmap Visualization", ln=True)
        pdf.image(heatmap_path, x=110, y=155, w=80, h=60)
    else:
        print("⚠ Heatmap image missing:", heatmap_path)

    # === SAVE PDF ===
    pdf_path = f"report_{report_name}.pdf"
    pdf.output(pdf_path)
    print(f"✅ Report saved: {pdf_path}")

    return pdf_path
