import streamlit as st
import json
import PyPDF2
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

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("🩺 ClinText: AI Medical Scribe")
st.markdown("Transform unstructured clinical PDFs into structured insights using local Llama 3.2.")
st.markdown("---")

# -----------------------------------------
# LAYOUT: LEFT COLUMN (Input) / RIGHT COLUMN (Dashboard)
# -----------------------------------------
col_input, col_dashboard = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📥 Input Clinical Record")
    
    tab_sample, tab_pdf = st.tabs(["📄 Sample Records", "📑 PDF Upload"])
    
    input_text = ""
    
    with tab_sample:
        selected_index = st.selectbox("Select a Hugging Face Sample:", range(len(df_samples)), format_func=lambda x: f"Patient Record #{x+1}")
        input_text = st.text_area("Review/Edit Text:", value=df_samples.iloc[selected_index]['text'], height=300, key="sample_text")
        
    with tab_pdf:
        uploaded_file = st.file_uploader("Upload Medical Record (PDF)", type=["pdf"])
        if uploaded_file:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
            input_text = st.text_area("Extracted PDF Text:", value=pdf_text, height=300, key="pdf_text")

    run_extraction = st.button("⚡ Extract Clinical Insights", type="primary", use_container_width=True)

# -----------------------------------------
# RIGHT COLUMN (The Dashboard)
# -----------------------------------------
with col_dashboard:
    st.subheader("🏥 Patient Summary Dashboard")
    
    if run_extraction and input_text.strip():
        with st.spinner("Analyzing record with LLM..."):
            raw_output = extract_clinical_data(input_text)
            
            try:
                clean_output = raw_output.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_output)
                
                st.info(f"**Primary Diagnosis:** {data.get('primary_diagnosis', 'Not specified')}")
                
                alerts = data.get('potential_risks_or_alerts', [])
                if alerts and alerts[0].lower() != "none":
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
                    
                with st.expander("🛠️ View Raw JSON (Developer)"):
                    st.json(data)
                    
            except json.JSONDecodeError:
                st.error("Failed to parse the LLM output into a valid dashboard. The model may have hallucinated.")
                st.write("**Raw Output:**")
                st.text(raw_output)
    
    elif not run_extraction:
        st.caption("Awaiting input... Select a record on the left and click 'Extract Clinical Insights'.")