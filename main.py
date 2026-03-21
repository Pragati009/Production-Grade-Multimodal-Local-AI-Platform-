import re

from fastapi import FastAPI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import sqlite3


app = FastAPI()

#Create a database
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT
)
""")

conn.commit()

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Read PDF
reader = PdfReader("NBayesLogReg.pdf")
text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted + " "

# Clean the text (remove extra spaces and line breaks)
text = re.sub(r'\s+', ' ', text)

print("First 500 characters of PDF:")
print(text[:500])

# Better chunking (word-based instead of sentence)
def chunk_text(text, chunk_size=200, overlap = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)

print("Total chunks created:", len(chunks))

# Create embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# # Chat memory for multi-turn conversations
# chat_memory = []  # List of {"question":..., "answer":...}

@app.get("/chat")
def chat(prompt: str):
    # Convert query to embedding
    query_embedding = model.encode([prompt])
    D, I = index.search(np.array(query_embedding), k=10)  # Top 7 chunks

    # Build context
    context = ""
    for idx in I[0]:
        context += chunks[idx] + "\n"

    print("\nRetrieved Context:")
    print(context)

    # Debug: show chunks returned
    # print("\n---DEBUG: Chunks sent to AI---")
    # for idx in I[0]:
    #     print(f"{idx}: {chunks[idx]}")
    # print("---END DEBUG---")

    # Include previous conversation in memory
    # Load last 5 conversations from database
    cursor.execute("""
    SELECT question, answer 
    FROM conversations
    ORDER BY id DESC
    LIMIT 5
    """)

    rows = cursor.fetchall()

    memory_text = ""

    for q, a in reversed(rows):
        memory_text += f"User: {q}\nAI: {a}\n"
    

    # Build full prompt for AI
    full_prompt = f"""
You are an AI assistant. Answer the question ONLY using the information provided in the context below.
If the answer is not in the context, say "I could not find that information in the document."

Previous conversation:
{memory_text}

Context:
{context}

Question: {prompt}

Answer:
"""

    # Send prompt to Gemma via Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": full_prompt,
            "stream": False
        }
    )

    result = response.json()
    answer = result.get("response", "No response")

    # chat_memory.append({"question": prompt, "answer": answer})
    # Save conversation in SQLite
    cursor.execute(
        "INSERT INTO conversations (question, answer) VALUES (?, ?)",
        (prompt, answer)
    )

    conn.commit()
    return {"response": answer}