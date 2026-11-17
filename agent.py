import os
import base64
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from datasets import load_dataset
import pandas as pd

# -----------------------------------------
# DUAL ARCHITECTURE ROUTING (Multimodal)
# -----------------------------------------
# Safely check for Gemini API key
gemini_api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    pass 

if not gemini_api_key:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Route the client connection based on the key
if gemini_api_key:
    # CLOUD MODE: Gemini 1.5 Pro
    genai.configure(api_key=gemini_api_key)
    MODEL_NAME = "gemini-1.5-pro"
    print("Agent initialized in CLOUD MODE (Gemini Vision)")
else:
    # LOCAL MODE: Ollama Vision
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3.2-vision"
    print("Agent initialized in LOCAL MODE (Ollama Vision)")

# -----------------------------------------
# DATA PROCESSING & EXTRACTION
# -----------------------------------------
@st.cache_data
def load_clinical_data():
    """Loads open-source synthetic clinical notes from Hugging Face."""
    dataset = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train[:20]")
    df = pd.DataFrame(dataset)
    
    if 'text' not in df.columns:
        if 'input' in df.columns:
            df['text'] = df['input']
        elif 'note' in df.columns:
            df['text'] = df['note']
        else:
            string_cols = df.select_dtypes(include=['object']).columns
            df['text'] = df[string_cols[0]] if len(string_cols) > 0 else "Error: No text column found."
            
    return df

def extract_clinical_data(text: str = "", image_base64: str = None):
    """Sends unstructured text or base64 images to the active Vision LLM."""
    
    system_instructions = """
    You are an expert clinical data extraction assistant. Parse the provided clinical record 
    (which may be text or a handwritten prescription image) and output a strict JSON object containing:
    1. "primary_diagnosis": (string)
    2. "key_medications": (array of strings with dosages if available)
    3. "potential_risks_or_alerts": (array of strings)
    4. "follow_up_actions": (array of strings)

    CRITICAL PHARMACY RULE: If any handwritten text (especially medications) is illegible, 
    do not guess. Output "UNREADABLE" for that specific item.

    Return valid JSON only, with no markdown code block formatting or extra conversational text.
    """

    if gemini_api_key:
        # --- GEMINI EXECUTION (CLOUD) ---
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instructions
        )
        
        contents = []
        if text.strip():
            contents.append(f"Clinical Note Text:\n{text[:3000]}")
            
        if image_base64:
            contents.append({
                "mime_type": "image/jpeg",
                "data": base64.b64decode(image_base64)
            })
            
        try:
            # Force strictly structured JSON using Gemini's native MIME type control
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return response.text
        except Exception as e:
            return f'{{"error": "Gemini API Failed. Details: {str(e)}"}}'

    else:
        # --- OLLAMA EXECUTION (LOCAL EDGE) ---
        content_payload = [{"type": "text", "text": system_instructions}]
        
        if text.strip():
            content_payload.append({
                "type": "text", 
                "text": f"Clinical Note Text:\n{text[:3000]}"
            })
            
        if image_base64:
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'{{"error": "Ollama API Connection Failed. Details: {str(e)}"}}'