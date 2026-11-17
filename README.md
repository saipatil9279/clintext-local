# 🩺 ClinText: Multimodal Edge-to-Cloud Clinical AI Engine

[![Python](https://img.shields.io/badge/Python-3.15-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Cloud-orange.svg)](https://groq.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local-green.svg)](https://ollama.com)

Most healthcare data is trapped in messy, unstructured PDFs or illegible handwritten prescriptions. **ClinText** is a multimodal extraction pipeline that rips data out of text and images, structuring it into strictly typed JSON (diagnoses, critical alerts, meds, follow-ups).

Instead of hardcoding a single provider, this engine is built with a **dual-execution architecture**. It sniffs out its environment at runtime and routes inference accordingly—giving you HIPAA-compliant edge execution locally, and face-melting API speed in the cloud.

### 🚀 [Try the Live Cloud Demo](https://clintext-local.streamlit.app/)

---

## 🧠 Dual-Routing Multimodal Pipeline

Write once, run anywhere. The `agent.py` controller checks for cloud secrets at startup and routes both text payloads and Base64-encoded image streams dynamically.

```text
               [ Unstructured Clinical PDF / Handwritten Image ]
                                  │
                                  ▼
                         { agent.py Controller }
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ 💻 Edge Mode (Local) ]                        [ ☁️ Production Mode (Cloud) ]
  • Backend: Ollama (localhost:11434)             • Backend: Groq Vision API 
  • Model: Llama 3.2 Vision (11B)                 • Model: Qwen 3.8 (27B)
  • Vibe: 100% private, offline OCR               • Vibe: High-speed web deployment
```

- **Running Locally?** It silently defaults to the `llama3.2-vision` model running on your local Ollama port. Zero data leaves your machine. 
- **Pushed to Streamlit Cloud?** It detects the `GROQ_API_KEY` secret and instantly hot-swaps to Groq's high-speed inference layer for seamless web performance.

---

## ⚡ Core Capabilities

- **Digital PDF to Strict JSON:** Ingests raw discharge summaries and forces the LLM to output clean schemas. No chatty text, just data.
- **Image OCR & The "Pharmacy Rule":** Upload handwritten prescriptions (JPG/PNG). The system prompt enforces strict anti-hallucination rules—if the cursive is illegible, the model outputs "UNREADABLE" instead of guessing fatal medication doses.
- **Synthetic Data Testing:** Integrated directly with Hugging Face (`starmpcc/Asclepius-Synthetic-Clinical-Notes`) so you can test the pipeline without handling real PHI.
- **Real-Time Clinical Dashboard:** Streamlit UI automatically renders the JSON into a clean medical dashboard with red-flag critical alerts.

---

## 🛠️ The Stack

- **UI & Routing:** Streamlit, Python 3.15
- **Data Pipeline:** PyPDF2, Pillow, Pandas, Hugging Face `datasets`
- **Inference Engines:** Ollama (Edge), Groq API (Cloud)

---

## 💻 Run it Locally (Edge Mode)

Want to run the multimodal pipeline entirely on your own hardware for maximum privacy?

```bash
# 1. Clone & enter
git clone [https://github.com/saipatil9279/clintext-local.git](https://github.com/saipatil9279/clintext-local.git)
cd clintext-local

# 2. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Spin up the local LLM
# (Requires Ollama installed: [https://ollama.com](https://ollama.com))
ollama pull llama3.2-vision

# 4. Launch the dashboard
streamlit run app.py
```

---

## ☁️ Cloud Deployment Setup

To push this to Streamlit Community Cloud and activate the high-speed Groq pipeline, simply add your API key to the Streamlit **Settings > Secrets** dashboard:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```
*The app will automatically detect this key and bypass the local Ollama server.*

---