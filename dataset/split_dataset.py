import json
import random

random.seed(42)

with open("ml_qa_dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

random.shuffle(data)

split = int(0.8 * len(data))
train = data[:split]
val = data[split:]

with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with open("val.jsonl", "w", encoding="utf-8") as f:
    for item in val:
        f.write(json.dumps(item) + "\n")

print(f"Total  : {len(data)}")
print(f"Train  : {len(train)}")
print(f"Val    : {len(val)}")
print("Files  : train.jsonl, val.jsonl")
