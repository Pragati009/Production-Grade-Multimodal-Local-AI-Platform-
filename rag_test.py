from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

reader = PdfReader("NBayesLogReg.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

    print(text[:1000])

# Chunk text
sentences = re.split(r'(?<=[.?\n])\s+', text)
chunks = [s.strip() for s in sentences if s.strip()]

# Embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Total chunks stored:", len(chunks))

# Ask questions directly
while True:
    query = input("\nAsk a question: ")
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    print("\nMost relevant chunks:")
    for idx in I[0]:
        print("----")
        print(chunks[idx])