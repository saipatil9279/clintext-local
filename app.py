import os
import json
import base64
import streamlit as st
import PyPDF2
from PIL import Image
from agent import load_clinical_data, extract_clinical_data

# -----------------------------------------
# CACHING & SETUP
# -----------------------------------------
st.set_page_config(
    page_title="ClinText | Clinical Extraction",
    page_icon="🩺",
    layout="wide"
)

df_samples = load_clinical_data()

# Detect execution environment (Cloud vs Local Edge)
is_cloud = False
try:
    if "GEMINI_API_KEY" in st.secrets or os.environ.get("GEMINI_API_KEY"):
        is_cloud = True
except Exception:
    is_cloud = bool(os.environ.get("GEMINI_API_KEY"))

if is_cloud:
    model_label = "cloud-hosted Gemini 1.5 Flash via Google AI Studio"
    status_badge = "☁️ **Active Architecture:** Cloud Inference (`gemini-1.5-flash`)"
else:
    model_label = "local Llama 3.2 Vision via Ollama"
    status_badge = "💻 **Active Architecture:** Local Edge Inference (`llama3.2-vision` via Ollama)"

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("🩺 ClinText: AI Medical Scribe")
st.markdown(f"Transform unstructured clinical documents and handwritten prescriptions into structured insights using **{model_label}**.")
st.caption(status_badge)
st.markdown("---")

# -----------------------------------------
# LAYOUT: LEFT COLUMN (Input) / RIGHT COLUMN (Dashboard)
# -----------------------------------------
col_input, col_dashboard = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📥 Input Clinical Record")
    
    tab_sample, tab_upload = st.tabs(["📄 Sample Records", "📂 File Upload (PDF/Image)"])
    
    input_text = ""
    input_image_base64 = None
    
    with tab_sample:
        selected_index = st.selectbox(
            "Select a Hugging Face Sample:",
            range(len(df_samples)),
            format_func=lambda x: f"Patient Record #{x+1}"
        )
        sample_text = st.text_area(
            "Review/Edit Text:",
            value=df_samples.iloc[selected_index]['text'],
            height=300,
            key="sample_text"
        )
        input_text = sample_text
        
    with tab_upload:
        uploaded_file = st.file_uploader("Upload Medical Record (PDF, JPG, PNG)", type=["pdf", "png", "jpg", "jpeg"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pdf_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                input_text = st.text_area(
                    "Extracted PDF Text:",
                    value=pdf_text,
                    height=300,
                    key="pdf_text"
                )
            else:
                # Handle Image Uploads (Handwritten Prescriptions)
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Prescription", use_container_width=True)
                input_image_base64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    run_extraction = st.button("⚡ Extract Clinical Insights", type="primary", use_container_width=True)

# -----------------------------------------
# RIGHT COLUMN (The Dashboard)
# -----------------------------------------
with col_dashboard:
    st.subheader("🏥 Patient Summary Dashboard")
    
    if run_extraction and (input_text.strip() or input_image_base64):
        with st.spinner(f"Analyzing record with {model_label}..."):
            raw_output = extract_clinical_data(text=input_text, image_base64=input_image_base64)
            
            try:
                clean_output = raw_output.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_output)
                
                st.info(f"**Primary Diagnosis:** {data.get('primary_diagnosis', 'Not specified')}")
                
                alerts = data.get("potential_risks_or_alerts", [])
                if alerts and str(alerts[0]).strip().lower() != "none":
                    for alert in alerts:
                        st.error(f"⚠️ **CRITICAL ALERT:** {alert}")
                
                st.markdown("### 💊 Key Medications")
                meds = data.get("key_medications", [])
                if meds:
                    for med in meds:
                        st.markdown(f"- {med}")
                else:
                    st.write("No specific medications extracted.")
                    
                st.markdown("### 📅 Follow-up Plan")
                follow_ups = data.get("follow_up_actions", [])
                if follow_ups:
                    for action in follow_ups:
                        st.markdown(f"- {action}")
                else:
                    st.write("No follow-up actions specified.")
                
                st.download_button(
                    label="📥 Download Structured JSON",
                    data=json.dumps(data, indent=2),
                    file_name="clinical_summary.json",
                    mime="application/json",
                    use_container_width=True
                )
                    
                with st.expander("🛠️ View Raw JSON (Developer)"):
                    st.json(data)
                    
            except json.JSONDecodeError:
                st.error("Failed to parse the LLM output into a valid dashboard. The model may have hallucinated or couldn't read the handwriting.")
                st.write("**Raw Output:**")
                st.text(raw_output)
    
    elif not run_extraction:
        st.caption("Awaiting input... Upload a file and click 'Extract Clinical Insights'.")