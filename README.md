# Production-Grade Local AI Chat Platform

A full-stack AI system built from scratch — featuring RAG (Retrieval-Augmented Generation), LoRA fine-tuning, real-time monitoring with Prometheus & Grafana, and a live model toggle API.

---

## What This Project Does

Ask questions about any PDF document. The AI retrieves the most relevant sections and answers using a local LLM — no cloud, no API keys, runs entirely on your machine.

**Two AI modes, switchable at runtime:**
- **Base mode** — Gemma 2b (via Ollama) answers questions grounded in the PDF using RAG
- **Fine-tuned mode** — GPT-2 + LoRA adapter trained on a custom ML dataset, **8x faster** than the base model

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│                  localhost:8000                          │
│                                                          │
│  /chat ──► RAG Pipeline ──► Base Model (Gemma via Ollama)│
│        └──► Fine-tuned GPT-2 + LoRA adapter             │
│  /model ──► Toggle between base / fine-tuned            │
│  /metrics ──► Prometheus scrape endpoint                 │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   RAG Pipeline        │
        │  PDF → Chunks →       │
        │  Embeddings (MiniLM)  │
        │  → FAISS Index        │
        └───────────────────────┘
                    │
        ┌───────────▼───────────┐     ┌──────────────────┐
        │   Prometheus          │────►│  Grafana          │
        │   localhost:9090      │     │  localhost:3000   │
        │   Scrapes every 5s    │     │  Live Dashboard   │
        └───────────────────────┘     └──────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| RAG Pipeline | PDF → chunked → embedded → FAISS vector search → LLM |
| Local LLM | Gemma 2b via Ollama — no cloud, no API key |
| LoRA Fine-tuning | GPT-2 fine-tuned on 181 custom ML Q&A pairs (0.24% of params trained) |
| Model Toggle API | Switch models at runtime via `POST /model` |
| Prometheus Metrics | Latency, hallucination rate, CPU & RAM usage, active model |
| Grafana Dashboard | 6-panel live dashboard, auto-provisioned |
| Conversation Memory | SQLite stores all Q&A history per model |
| Docker Compose | One-command setup for Prometheus + Grafana |

---

## Tech Stack

**AI / ML**
- `sentence-transformers` — MiniLM-L6-v2 for document embeddings
- `faiss-cpu` — vector similarity search
- `Ollama` — local LLM runtime (Gemma 2b)
- `transformers` + `peft` + `trl` — LoRA fine-tuning pipeline
- `torch` — model inference

**Backend**
- `FastAPI` — REST API with async endpoints
- `SQLite` — conversation memory with schema migrations
- `pypdf` — PDF text extraction

**MLOps / Monitoring**
- `Prometheus` — metrics collection (custom counters, histograms, gauges)
- `Grafana` — dashboard with auto-provisioned datasource + panels
- `Docker Compose` — container orchestration
- `psutil` — system resource monitoring

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### 2. Pull the LLM
```bash
ollama pull gemma:2b
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your PDF
Place any PDF file in the project root and update the filename in `main.py`:
```python
reader = PdfReader("your-document.pdf")
```

### 5. Start the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start monitoring (Prometheus + Grafana)
```bash
docker-compose up -d
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/chat?prompt=your question` | Ask the AI a question |
| `GET` | `/model` | See which model is active |
| `POST` | `/model?name=base` | Switch to Gemma 2b (RAG mode) |
| `POST` | `/model?name=finetuned` | Switch to fine-tuned GPT-2 + LoRA |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |

**Example:**
```bash
# Ask a question
curl "http://localhost:8000/chat?prompt=What is Naive Bayes?"

# Switch to fine-tuned model
curl -X POST "http://localhost:8000/model?name=finetuned"

# Response format
{
  "model": "finetuned",
  "response": "Naive Bayes is a probabilistic classifier...",
  "latency": 5.96
}
```

---

## Monitoring Dashboard

Open **http://localhost:3000** (login: `admin` / `admin`)

The **"Local AI Chat Monitor"** dashboard shows:

| Panel | Metric |
|---|---|
| AI Response Latency | Avg + P95 latency per request |
| Hallucination Rate | Requests where AI said "could not find" |
| CPU Usage | Live gauge, red above 85% |
| RAM Usage | Live gauge, red above 90% |
| Active Model | Shows Base or Fine-Tuned |
| Total Requests / Hallucinations | Lifetime counters |

---

## LoRA Fine-Tuning

A custom ML dataset of **181 Q&A pairs** was created and used to fine-tune GPT-2 with LoRA:

```bash
# Run fine-tuning (~15 min on CPU)
python finetune.py

# Compare base vs fine-tuned on 30 questions
python compare_models.py
```

**Training results:**
- Trainable parameters: **294,912** (0.24% of GPT-2)
- Training time: ~5 minutes on CPU
- Loss: 4.15 → 3.71 over 3 epochs
- Fine-tuned model is **8.1x faster** than Gemma 2b

The LoRA adapter is saved in `ml_adapter/` — only **1.2 MB** compared to 500 MB for the full model.

---

## Project Structure

```
├── main.py                  # FastAPI app — RAG + model toggle + metrics
├── finetune.py              # LoRA fine-tuning pipeline
├── compare_models.py        # Base vs fine-tuned evaluation (30 questions)
├── evaluate_rag.py          # RAG accuracy evaluation
├── test_adapter.py          # Adapter vs base model comparison
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Prometheus + Grafana setup
├── prometheus.yml           # Prometheus scrape config
├── dataset/
│   ├── ml_qa_dataset.jsonl  # 181 ML Q&A pairs (full dataset)
│   ├── train.jsonl          # 144 training examples (80%)
│   ├── val.jsonl            # 37 validation examples (20%)
│   └── split_dataset.py     # Train/val split script
├── ml_adapter/
│   ├── adapter_config.json  # LoRA config
│   └── adapter_model.safetensors  # Trained weights (1.2 MB)
└── grafana/
    └── provisioning/
        ├── datasources/     # Auto-connects Grafana to Prometheus
        └── dashboards/      # Pre-built monitoring dashboard
```

---

## Key Results

| Metric | Base (Gemma 2b) | Fine-Tuned (GPT-2 + LoRA) |
|---|---|---|
| Avg response latency | 47.7s | 5.9s |
| Speed | — | **8.1x faster** |
| Hallucination rate | Low (RAG-grounded) | Moderate |
| Scope | PDF-only knowledge | Broad ML domain |
| Model size on disk | 1.7 GB (Ollama) | 1.2 MB (adapter) |

---

## Skills Demonstrated

`RAG` · `LLM Fine-tuning` · `LoRA / PEFT` · `Vector Search` · `FAISS` · `Embeddings`  
`FastAPI` · `REST API Design` · `SQLite` · `Prometheus` · `Grafana` · `Docker`  
`MLOps` · `Model Monitoring` · `HuggingFace Transformers` · `Ollama` · `Python`
