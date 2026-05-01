import re
import io
import torch
import requests
import sqlite3
import numpy as np
import psutil
import time
import whisper

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import faiss
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = FastAPI()

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT       = Counter("request_count",            "Total API requests")
REQUEST_LATENCY     = Histogram("request_latency_seconds","Request latency in seconds")
FAILED_RESPONSES    = Counter("failed_responses",         "Responses not found in context")
RETRIEVAL_SUCCESS   = Counter("retrieval_success",        "Requests where answer was found in context")
RETRIEVAL_DISTANCE  = Gauge("retrieval_avg_distance",     "Avg FAISS L2 distance of top-k chunks (lower=better)")
CPU_USAGE           = Gauge("cpu_usage_percent",          "CPU usage percent")
MEMORY_USAGE        = Gauge("memory_usage_percent",       "RAM usage percent")
ACTIVE_MODEL        = Gauge("active_model",               "Active model: 0=base 1=finetuned")

# ── SQLite memory ─────────────────────────────────────────────────────────────
conn   = sqlite3.connect("chat_memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        model    TEXT,
        question TEXT,
        answer   TEXT
    )
""")
# migrate existing table if model column is missing
try:
    cursor.execute("ALTER TABLE conversations ADD COLUMN model TEXT DEFAULT 'base'")
except Exception:
    pass  # column already exists
conn.commit()

# ── RAG: embedding model + PDF + FAISS ───────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

reader = PdfReader("NBayesLogReg.pdf")
pdf_text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        pdf_text += extracted + " "
pdf_text = re.sub(r'\s+', ' ', pdf_text)
print("PDF loaded:", len(pdf_text), "characters")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

chunks     = chunk_text(pdf_text)
embeddings = embed_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)
print(f"FAISS index ready: {len(chunks)} chunks")

# ── Fine-tuned model (GPT-2 + LoRA adapter) ──────────────────────────────────
print("Loading fine-tuned model (GPT-2 + LoRA)...")
ft_tokenizer  = AutoTokenizer.from_pretrained("./ml_adapter")
ft_tokenizer.pad_token = ft_tokenizer.eos_token
_gpt2_base    = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
ft_model_obj  = PeftModel.from_pretrained(_gpt2_base, "./ml_adapter")
ft_model_obj.eval()
ft_pipe = pipeline(
    "text-generation",
    model=ft_model_obj,
    tokenizer=ft_tokenizer,
)
print("Fine-tuned model ready.")

# ── Whisper tiny (loaded once at startup) ─────────────────────────────────────
print("Loading Whisper tiny...")
whisper_model = whisper.load_model("tiny")
print("Whisper ready.")

# ── Model toggle state ────────────────────────────────────────────────────────
current_model = "base"   # "base" | "finetuned"
ACTIVE_MODEL.set(0)

# ── Helper: retrieve PDF context via RAG ──────────────────────────────────────
def retrieve_context(prompt: str, k: int = 3):
    q_emb   = embed_model.encode([prompt])
    D, I    = faiss_index.search(np.array(q_emb), k=k)
    avg_dist = float(np.mean(D[0]))
    RETRIEVAL_DISTANCE.set(round(avg_dist, 4))
    retrieved = [{"chunk_id": int(idx), "text": chunks[idx], "distance": round(float(D[0][j]), 4)}
                 for j, idx in enumerate(I[0])]
    context   = "\n".join(r["text"] for r in retrieved)
    return context, retrieved

# ── Helper: load recent conversation memory ───────────────────────────────────
def load_memory(limit: int = 5) -> str:
    cursor.execute(
        "SELECT question, answer FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    return "".join(f"User: {q}\nAI: {a}\n" for q, a in reversed(rows))

# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.get("/model")
def get_model():
    """Return the currently active model."""
    return {"active_model": current_model}


@app.post("/model")
def set_model(name: str):
    """Switch between 'base' (Gemma via Ollama) and 'finetuned' (GPT-2 + LoRA)."""
    global current_model
    if name not in ("base", "finetuned"):
        raise HTTPException(status_code=400, detail="name must be 'base' or 'finetuned'")
    current_model = name
    ACTIVE_MODEL.set(0 if name == "base" else 1)
    print(f"[MODEL SWITCH] → {name}")
    return {"active_model": current_model}


@app.get("/chat")
def chat(prompt: str):
    global current_model
    start_time = time.time()

    context, sources = retrieve_context(prompt)
    memory_text      = load_memory()

    # ── Base model: Gemma 2b via Ollama (RAG-grounded) ────────────────────────
    if current_model == "base":
        full_prompt = f"""You are an AI assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I could not find that information in the document."

Previous conversation:
{memory_text}

Context:
{context}

Question: {prompt}

Answer:"""
        resp   = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma:2b", "prompt": full_prompt, "stream": False},
        )
        answer = resp.json().get("response", "No response")

    # ── Fine-tuned model: GPT-2 + LoRA (ML domain expert) ────────────────────
    else:
        ft_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        out       = ft_pipe(
            ft_prompt,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=ft_tokenizer.eos_token_id,
        )
        raw    = out[0]["generated_text"]
        answer = raw.split("### Response:\n")[-1].strip()
        if not answer:
            answer = "I could not find that information in the document."

    # ── Metrics ───────────────────────────────────────────────────────────────
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    REQUEST_COUNT.inc()

    if "could not find" in answer.lower():
        FAILED_RESPONSES.inc()
    else:
        RETRIEVAL_SUCCESS.inc()

    cpu    = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory().percent
    CPU_USAGE.set(cpu)
    MEMORY_USAGE.set(memory)

    print(f"[{current_model.upper()}] latency={latency:.2f}s | CPU={cpu}% | RAM={memory}%")

    # ── Persist to SQLite ─────────────────────────────────────────────────────
    cursor.execute(
        "INSERT INTO conversations (model, question, answer) VALUES (?, ?, ?)",
        (current_model, prompt, answer),
    )
    conn.commit()

    return {
        "model":    current_model,
        "response": answer,
        "latency":  round(latency, 2),
        "sources":  sources if current_model == "base" else [],
    }


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    """
    Accept a WAV/MP3 audio file, transcribe with Whisper tiny,
    then pass the transcription through the active AI model.
    Returns: transcription + AI response + latency.
    """
    audio_bytes = await file.read()
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    result      = whisper_model.transcribe(audio_array, fp16=False, language="en")
    question    = result["text"].strip()

    if not question:
        raise HTTPException(status_code=400, detail="Could not transcribe audio — please speak clearly.")

    # Re-use the /chat logic by calling it internally
    import httpx
    async with httpx.AsyncClient() as client:
        chat_resp = await client.get(
            "http://localhost:8000/chat",
            params={"prompt": question},
            timeout=120,
        )
    chat_data = chat_resp.json()

    return {
        "transcription": question,
        "model":         chat_data.get("model"),
        "response":      chat_data.get("response"),
        "latency":       chat_data.get("latency"),
    }
