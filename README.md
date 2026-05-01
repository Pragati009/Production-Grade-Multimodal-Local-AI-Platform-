# Local AI Chat — Production-Grade RAG + LoRA Platform

A full-stack AI system built entirely from scratch — RAG pipeline, LoRA fine-tuning, voice input via Whisper, real-time monitoring with Prometheus & Grafana, and a live model-toggle API. No cloud, no API keys, runs on your own machine.

---

## What It Does

Ask questions about any PDF document by typing or speaking. The AI retrieves the most relevant sections and answers using a local LLM. Switch between two AI modes at runtime without restarting anything.

**Two AI modes, swappable live:**
- **Base** — Gemma 2b (via Ollama), RAG-grounded, answers only from the PDF
- **Fine-Tuned** — GPT-2 + LoRA adapter trained on 181 custom ML Q&A pairs, **8x faster**

**Three ways to interact:**
- Streamlit chat UI (dark theme, citations, voice tab)
- REST API (`curl`, any HTTP client)
- CLI voice client (`voice_chat.py` — mic → Whisper → AI → optional TTS)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit UI  (port 4000)                  │
│  Dark theme · Chat history · Voice tab · Source citations    │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI Server  (port 8000)                │
│                                                              │
│  GET  /chat   ──► RAG Pipeline ──► Base model (Gemma 2b)    │
│             └──► Fine-tuned GPT-2 + LoRA adapter            │
│  POST /voice  ──► Whisper tiny (STT) ──► /chat              │
│  POST /model  ──► Toggle base / finetuned at runtime        │
│  GET  /metrics ──► Prometheus scrape endpoint               │
└─────────┬───────────────────────┬───────────────────────────┘
          │                       │
┌─────────▼─────────┐   ┌────────▼─────────┐
│   RAG Pipeline    │   │   SQLite Memory   │
│ PDF → 200-word    │   │ Stores all Q&A    │
│ chunks → MiniLM   │   │ per model with    │
│ embeddings →      │   │ schema migration  │
│ FAISS L2 index    │   └──────────────────┘
└─────────┬─────────┘
          │
┌─────────▼──────────┐     ┌──────────────────┐
│   Prometheus       │────►│  Grafana          │
│   port 9090        │     │  port 3000        │
│   Scrapes every 5s │     │  12-panel dashboard│
└────────────────────┘     └──────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| RAG Pipeline | PDF → chunked → MiniLM embeddings → FAISS L2 search → LLM |
| Local LLM | Gemma 2b via Ollama — no cloud, no API key |
| LoRA Fine-Tuning | GPT-2 fine-tuned on 181 custom ML Q&A pairs (0.24% of params trained) |
| Model Toggle API | Switch models live at runtime via `POST /model` |
| Voice Input | Whisper tiny transcribes mic audio — no ffmpeg needed |
| Optional TTS | pyttsx3 reads AI answers aloud (`--tts` flag) |
| Streamlit UI | Dark theme, model toggle, voice tab, expandable source citations |
| Prometheus Metrics | Latency histogram, hallucination rate, CPU/RAM gauges, retrieval distance |
| Grafana Dashboard | 12-panel live dashboard, auto-provisioned via JSON |
| Retrieval Quality | FAISS L2 distance tracked as a real-time Gauge metric |
| Conversation Memory | SQLite stores all Q&A history per model |
| Docker Compose | One-command Prometheus + Grafana setup |

---

## Tech Stack

**AI / ML**
- `sentence-transformers` — MiniLM-L6-v2 for document embeddings
- `faiss-cpu` — vector similarity search
- `Ollama` — local LLM runtime (Gemma 2b)
- `transformers` + `peft` + `trl` — LoRA fine-tuning pipeline
- `openai-whisper` — speech-to-text (float32 numpy, no ffmpeg)
- `torch` — model inference

**Backend**
- `FastAPI` — REST API with async endpoints
- `SQLite` — conversation memory with schema migrations
- `pypdf` — PDF text extraction
- `httpx` — async internal HTTP for voice → chat routing

**Frontend**
- `Streamlit` — dark-themed chat UI with voice tab and source citations
- `pyttsx3` — text-to-speech (optional CLI flag)
- `sounddevice` — microphone recording for CLI voice client

**MLOps / Monitoring**
- `Prometheus` — custom counters, histograms, and gauges
- `Grafana` — dashboard with auto-provisioned datasource + panels
- `Docker Compose` — container orchestration
- `psutil` — CPU and RAM monitoring

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### 1. Pull the base LLM
```bash
ollama pull gemma:2b
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your PDF
Place a PDF in the project root and update the filename in `main.py`:
```python
reader = PdfReader("your-document.pdf")
```

### 4. Start Prometheus + Grafana
```bash
docker-compose up -d
```

### 5. Start the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Start the Streamlit UI
```bash
streamlit run app.py
```

Open **http://localhost:8501** to chat with your document.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/chat?prompt=your question` | Ask the AI (text) |
| `POST` | `/voice` | Upload WAV/MP3, get transcription + answer |
| `GET` | `/model` | Which model is active |
| `POST` | `/model?name=base` | Switch to Gemma 2b (RAG mode) |
| `POST` | `/model?name=finetuned` | Switch to GPT-2 + LoRA |
| `GET` | `/metrics` | Prometheus scrape endpoint |

**Examples:**
```bash
# Ask a question
curl "http://localhost:8000/chat?prompt=What is Naive Bayes?"

# Switch to fine-tuned model
curl -X POST "http://localhost:8000/model?name=finetuned"

# Voice input
curl -X POST "http://localhost:8000/voice" -F "file=@question.wav"

# Response format
{
  "model": "base",
  "response": "Naive Bayes is a probabilistic classifier...",
  "latency": 47.3,
  "sources": [
    { "chunk_id": 12, "text": "...", "distance": 84.2 }
  ]
}
```

---

## Voice Chat (CLI)

```bash
# Basic — text responses
python voice_chat.py

# With text-to-speech
python voice_chat.py --tts

# Use fine-tuned model, 8-second recording window
python voice_chat.py --model finetuned --seconds 8
```

Press Enter to record, speak, then the AI answers. No ffmpeg required.

---

## Monitoring Dashboard

Open **http://localhost:3000** (login: `admin` / `admin`)

The **"Local AI Chat Monitor"** dashboard has 12 panels:

| Panel | What It Shows |
|---|---|
| Response Latency (Avg + P95) | How fast the AI answers over time |
| Hallucination vs Success Rate | Requests answered vs. "could not find" |
| CPU Usage (gauge) | Live %, red above 85% |
| RAM Usage (gauge) | Live %, red above 90% |
| Retrieval Quality — FAISS Distance | Lower = AI found more relevant context |
| Active Model | Shows Base or Fine-Tuned |
| Retrieval Quality Over Time | FAISS distance trend |
| Answer Success Rate % | Success vs hallucination as percentages |
| Total Requests | Lifetime counter |
| Successful Answers | Lifetime counter |
| Hallucinations | Lifetime counter |
| Avg Response Time | Current moving average |

---

## LoRA Fine-Tuning

A custom ML dataset of **181 Q&A pairs** (covering Naive Bayes, Logistic Regression, neural networks, regularization, clustering, and more) was created and used to fine-tune GPT-2:

```bash
# Run fine-tuning (~10-20 min on CPU)
python finetune.py

# Compare base vs fine-tuned on 30 questions
python compare_models.py
```

**Training results:**
- Trainable parameters: **294,912** (0.24% of GPT-2's 124M)
- Training time: ~5 minutes on CPU
- Training loss: 4.15 → 3.71 over 3 epochs
- Fine-tuned model is **8.1x faster** than Gemma 2b

The LoRA adapter is saved in `ml_adapter/` — only **1.2 MB** vs. 500 MB for a full model copy.

---

## Project Structure

```
├── main.py                  # FastAPI app — RAG, model toggle, metrics, voice
├── app.py                   # Streamlit UI — dark theme, voice tab, citations
├── finetune.py              # LoRA fine-tuning pipeline (GPT-2 + PEFT + SFTTrainer)
├── compare_models.py        # Base vs fine-tuned evaluation (30 questions)
├── voice_chat.py            # CLI voice client — mic → Whisper → AI → TTS
├── requirements.txt         # All Python dependencies (pinned)
├── docker-compose.yml       # Prometheus + Grafana setup
├── prometheus.yml           # Prometheus scrape config (5s interval)
├── dataset/
│   ├── ml_qa_dataset.jsonl  # 181 ML Q&A pairs (Alpaca format)
│   ├── train.jsonl          # 144 training examples (80%)
│   ├── val.jsonl            # 37 validation examples (20%)
│   └── split_dataset.py     # Train/val split script
├── ml_adapter/
│   ├── adapter_config.json  # LoRA config (r=8, target: c_attn)
│   └── adapter_model.safetensors  # Trained weights (1.2 MB)
└── grafana/
    └── provisioning/
        ├── datasources/     # Auto-connects Grafana to Prometheus
        └── dashboards/      # 12-panel monitoring dashboard JSON
```

---

## Key Results

| Metric | Base (Gemma 2b) | Fine-Tuned (GPT-2 + LoRA) |
|---|---|---|
| Avg response latency | ~47s | ~5.9s |
| Speed | — | **8.1x faster** |
| Answer grounding | PDF-only (RAG) | Broad ML domain knowledge |
| Hallucination detection | "could not find" signal | Same signal |
| Adapter size on disk | 1.7 GB (Ollama) | **1.2 MB** |

---

## Skills Demonstrated

`RAG` · `LLM Fine-tuning` · `LoRA / PEFT` · `Vector Search` · `FAISS` · `Embeddings`  
`FastAPI` · `REST API Design` · `Async Python` · `SQLite`  
`Prometheus` · `Grafana` · `Docker Compose` · `MLOps` · `Model Monitoring`  
`HuggingFace Transformers` · `Ollama` · `Whisper STT` · `Streamlit`  
`Model Evaluation` · `Dataset Creation` · `Production System Design`
