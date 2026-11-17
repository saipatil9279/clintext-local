# 🩺 ClinText: Environment-Aware Clinical NLP Engine

[![Python](https://img.shields.io/badge/Python-3.15-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Cloud-orange.svg)](https://groq.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local-green.svg)](https://ollama.com)

Most healthcare data is trapped in messy, unstructured PDFs. **ClinText** rips that data out and structures it into strictly typed JSON (diagnoses, critical alerts, meds, follow-ups) using a dynamic LLM pipeline.

Instead of hardcoding a single provider, this engine is built with a **dual-execution architecture**. It sniffs out its environment at runtime and routes inference accordingly—giving you HIPAA-compliant edge execution locally, and face-melting API speed in the cloud.

### 🚀 [Try the Live Cloud Demo](https://clintext-local.streamlit.app/)

---

## Dual-Routing Architecture

Write once, run anywhere. The `agent.py` controller checks for cloud secrets at startup. No code changes required to deploy.

```text
                     [ Unstructured Clinical PDF / Text ]
                                  │
                                  ▼
                         { agent.py Controller }
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ 💻 Edge Mode (Local) ]                        [ ☁️ Production Mode (Cloud) ]
  • Backend: Ollama (localhost:11434)             • Backend: Groq API 
  • Model: Llama 3.2 (3B)                         • Model: GPT-OSS-120B
  • Vibe: 100% private, zero-latency              • Vibe: High-speed web deployment
```

- **Running Locally?** It silently defaults to the `llama3.2:3b` model running on your local Ollama port. Zero data leaves your machine. 
- **Pushed to Streamlit Cloud?** It detects the `GROQ_API_KEY` secret and instantly hot-swaps to Groq's high-speed inference layer for seamless web performance.

---

## ⚡ Core Capabilities

- **Messy PDF to Strict JSON:** Ingests raw discharge summaries and forces the LLM to output clean schemas. No chatty text, just data.
- **Synthetic Data Testing:** Integrated directly with Hugging Face (`starmpcc/Asclepius-Synthetic-Clinical-Notes`) so you can test the pipeline without handling real PHI.
- **Real-Time Clinical Dashboard:** Streamlit UI automatically renders the JSON into a clean medical dashboard with red-flag critical alerts.
- **Dev-Ready Export:** One-click download for the structured JSON payload to pipe into downstream databases.

---

## 🛠️ The Stack

- **UI & Routing:** Streamlit, Python 3.15
- **Data Pipeline:** PyPDF2, Pandas, Hugging Face `datasets`
- **Inference Engines:** Ollama (Edge), Groq API (Cloud)

---

## 💻 Run it Locally (Edge Mode)

Want to run the Llama 3.2 pipeline entirely on your own hardware for maximum privacy?

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
ollama pull llama3.2

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