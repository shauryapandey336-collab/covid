import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- SESSION STATE INIT ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "reports" not in st.session_state:
    st.session_state.reports = []

if "latest_pdf" not in st.session_state:
    st.session_state.latest_pdf = None

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="COVID-19 X-Ray Detection System",
    page_icon="🦠",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.title { text-align:center; font-size:40px; font-weight:800; }
.subtitle { text-align:center; color:#6b7280; margin-bottom:20px; }
.card {
    background:white;
    padding:25px;
    border-radius:14px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.08);
}
.footer { text-align:center; color:#6b7280; margin-top:40px; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = load_model("covid_model.h5")

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🦠 COVID-19 X-Ray Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning based Emergency Report Generator</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧾 Patient Information")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Age", 1, 120, 25)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
st.sidebar.markdown(f"📊 **Total Reports:** {len(st.session_state.reports)}")

# ---------------- MAIN CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray (JPG / PNG / JPEG)",
    type=["jpg", "png", "jpeg"]
)

# ---------------- ANALYZE IMAGE ----------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Chest X-Ray", use_container_width=True)

    img = img.resize((224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Analyze X-Ray"):
        with st.spinner("Analyzing using CNN model..."):
            pred = model.predict(img)[0][0]

        if pred > 0.5:
            st.session_state.prediction = "COVID NEGATIVE"
            st.session_state.confidence = pred * 100
        else:
            st.session_state.prediction = "COVID POSITIVE"
            st.session_state.confidence = (1 - pred) * 100

# ---------------- SHOW RESULT ----------------
if st.session_state.prediction:
    st.divider()

    if st.session_state.prediction == "COVID NEGATIVE":
        st.success("✅ COVID NEGATIVE")
    else:
        st.error("🚨 COVID POSITIVE – EMERGENCY ALERT")
        st.markdown("""
        <div style="background:#fee2e2;padding:15px;border-radius:10px;border-left:6px solid #b91c1c;">
        <b>🏥 Immediate Hospital Admission Recommended</b><br>
        • Continuous medical supervision<br>
        • Oxygen monitoring required<br>
        • Contact nearest hospital immediately
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(st.session_state.confidence))
    st.write(f"**Confidence:** {st.session_state.confidence:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- GENERATE PDF ----------------
if st.session_state.prediction and patient_name:

    if st.button("📄 Generate Medical PDF Report"):
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
        filename = f"Covid_Report_{patient_name.replace(' ','_')}_{len(st.session_state.reports)+1}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("COVID-19 X-Ray Detection Report", styles["Title"]))
        content.append(Paragraph(f"Date & Time: {timestamp}", styles["Normal"]))
        content.append(Paragraph("<br/>", styles["Normal"]))
        content.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
        content.append(Paragraph(f"Age: {patient_age}", styles["Normal"]))
        content.append(Paragraph(f"Gender: {patient_gender}", styles["Normal"]))
        content.append(Paragraph("<br/>", styles["Normal"]))
        content.append(Paragraph(f"Result: {st.session_state.prediction}", styles["Normal"]))
        content.append(Paragraph(f"Confidence: {st.session_state.confidence:.2f}%", styles["Normal"]))
        content.append(Paragraph("<br/>", styles["Normal"]))

        if st.session_state.prediction == "COVID POSITIVE":
            content.append(Paragraph(
                "Emergency Alert: Immediate hospital admission recommended.",
                styles["Normal"]
            ))

        content.append(Paragraph("<br/>Educational purpose only.", styles["Italic"]))
        content.append(Paragraph("Made by Shaurya Pandey", styles["Normal"]))

        doc.build(content)

        st.session_state.latest_pdf = filename
        st.session_state.reports.append({
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "result": st.session_state.prediction,
            "confidence": f"{st.session_state.confidence:.2f}%",
            "time": timestamp,
            "file": filename
        })

        st.success("📄 PDF Report Generated")

# ---------------- DOWNLOAD LATEST ----------------
if st.session_state.latest_pdf:
    with open(st.session_state.latest_pdf, "rb") as f:
        st.download_button(
            "⬇️ Download Latest Report",
            f,
            file_name=st.session_state.latest_pdf,
            mime="application/pdf"
        )

# ---------------- REPORT HISTORY ----------------
if st.session_state.reports:
    st.divider()
    st.subheader("📑 Report History")

    for i, r in enumerate(st.session_state.reports, start=1):
        with st.expander(f"Report {i} – {r['name']} ({r['time']})"):
            st.write(f"Name: {r['name']}")
            st.write(f"Age: {r['age']}")
            st.write(f"Gender: {r['gender']}")
            st.write(f"Result: {r['result']}")
            st.write(f"Confidence: {r['confidence']}")
            with open(r["file"], "rb") as f:
                st.download_button(
                    "⬇️ Download This Report",
                    f,
                    file_name=r["file"],
                    mime="application/pdf",
                    key=f"hist_{i}"
                )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
⚠️ Educational & academic use only<br>
<b>Made by Shaurya Pandey</b>
</div>
""", unsafe_allow_html=True)
