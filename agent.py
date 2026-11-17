import os
import streamlit as st
from openai import OpenAI
from datasets import load_dataset
import pandas as pd

# -----------------------------------------
# DUAL ARCHITECTURE ROUTING
# -----------------------------------------
# Safely check for Groq API key in Streamlit Secrets or OS Environment
groq_api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass # No secrets file found locally

if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY")

# Route the client connection based on the key
if groq_api_key:
    # CLOUD MODE: Groq API
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)
    MODEL_NAME = "openai/gpt-oss-120b"
    print("Agent initialized in CLOUD MODE (Groq)")
else:
    # LOCAL MODE: Ollama
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3.2:3b"
    print("Agent initialized in LOCAL MODE (Ollama)")

# -----------------------------------------
# DATA PROCESSING & EXTRACTION
# -----------------------------------------
@st.cache_data
def load_clinical_data():
    """Loads open-source synthetic clinical notes from Hugging Face."""
    dataset = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train[:20]")
    df = pd.DataFrame(dataset)
    
    # Handle schema variations dynamically
    if 'text' not in df.columns:
        if 'input' in df.columns:
            df['text'] = df['input']
        elif 'note' in df.columns:
            df['text'] = df['note']
        else:
            string_cols = df.select_dtypes(include=['object']).columns
            df['text'] = df[string_cols[0]] if len(string_cols) > 0 else "Error: No text column found."
            
    return df

def extract_clinical_data(note_text: str):
    """Sends unstructured text to the active LLM (Groq or Ollama) for JSON extraction."""
    prompt = f"""
    You are an expert clinical data extraction assistant. Parse the following unstructured discharge summary 
    and output a strict JSON object containing:
    1. "primary_diagnosis": (string)
    2. "key_medications": (array of strings with dosages if available)
    3. "potential_risks_or_alerts": (array of strings)
    4. "follow_up_actions": (array of strings)

    Clinical Note:
    {note_text[:3000]}

    Return valid JSON only, with no markdown code block formatting or extra conversational text.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "API Connection Failed. Using Model: {MODEL_NAME}. Details: {str(e)}"}}'