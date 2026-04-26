"""
Base GPT-2 vs Fine-Tuned LoRA adapter — head-to-head comparison.
Measures: correct answers, hallucinations, latency.
"""

import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

MODEL_NAME  = "gpt2"
ADAPTER_DIR = "./ml_adapter"
MAX_TOKENS  = 120

# ── 30 test questions with expected keywords ──────────────────────────────────
TESTS = [
    {"q": "What is Naive Bayes?",
     "keywords": ["probabilistic", "bayes", "classifier", "probability"]},
    {"q": "What assumption does Naive Bayes make?",
     "keywords": ["independent", "independence", "conditional"]},
    {"q": "What is Laplace smoothing?",
     "keywords": ["zero", "smoothing", "count", "probability"]},
    {"q": "What is the sigmoid function?",
     "keywords": ["sigmoid", "0", "1", "probability", "exp"]},
    {"q": "What is Logistic Regression?",
     "keywords": ["classification", "probability", "sigmoid", "binary"]},
    {"q": "What is the cost function for Logistic Regression?",
     "keywords": ["cross-entropy", "log loss", "log", "loss"]},
    {"q": "What is a decision boundary?",
     "keywords": ["boundary", "separate", "class", "hyperplane", "line"]},
    {"q": "What is overfitting?",
     "keywords": ["training", "generalize", "noise", "test", "overfit"]},
    {"q": "What is underfitting?",
     "keywords": ["simple", "bias", "high error", "underfit", "complex"]},
    {"q": "What is gradient descent?",
     "keywords": ["gradient", "loss", "learning rate", "optimize", "minimize"]},
    {"q": "What is the learning rate?",
     "keywords": ["learning rate", "step", "gradient", "update"]},
    {"q": "What is cross-validation?",
     "keywords": ["fold", "validation", "split", "train", "evaluate"]},
    {"q": "What is the F1 score?",
     "keywords": ["precision", "recall", "harmonic", "f1"]},
    {"q": "What is precision?",
     "keywords": ["true positive", "false positive", "precision", "correct"]},
    {"q": "What is recall?",
     "keywords": ["true positive", "false negative", "recall", "sensitivity"]},
    {"q": "What is a confusion matrix?",
     "keywords": ["true positive", "false positive", "matrix", "predicted"]},
    {"q": "What is a Random Forest?",
     "keywords": ["tree", "ensemble", "random", "forest", "voting"]},
    {"q": "What is a decision tree?",
     "keywords": ["split", "node", "tree", "branch", "leaf"]},
    {"q": "What is K-means clustering?",
     "keywords": ["cluster", "centroid", "k", "unsupervised"]},
    {"q": "What is PCA?",
     "keywords": ["principal", "component", "variance", "dimension"]},
    {"q": "What is regularization?",
     "keywords": ["overfit", "penalty", "l1", "l2", "regularization"]},
    {"q": "What is L1 regularization?",
     "keywords": ["lasso", "sparse", "zero", "absolute", "l1"]},
    {"q": "What is L2 regularization?",
     "keywords": ["ridge", "squared", "weight", "l2", "shrink"]},
    {"q": "What is a neural network?",
     "keywords": ["neuron", "layer", "weight", "network", "activation"]},
    {"q": "What is backpropagation?",
     "keywords": ["gradient", "backward", "chain rule", "weight", "loss"]},
    {"q": "What is dropout?",
     "keywords": ["dropout", "random", "overfit", "regularization", "zero"]},
    {"q": "What is the bias-variance tradeoff?",
     "keywords": ["bias", "variance", "tradeoff", "overfit", "underfit"]},
    {"q": "What is transfer learning?",
     "keywords": ["pretrained", "transfer", "fine-tune", "task", "reuse"]},
    {"q": "What is LoRA?",
     "keywords": ["lora", "low-rank", "adapter", "fine-tuning", "parameter"]},
    {"q": "What is RAG?",
     "keywords": ["retrieval", "generation", "augmented", "document", "context"]},
]

HALLUCINATION_SIGNALS = [
    "i don't know", "i cannot", "i'm not sure", "as an ai",
    "i apologize", "i am unable", "no information",
]

def is_repetitive(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 6:
        return False
    # Flag if any 4-word phrase repeats
    phrases = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
    return len(phrases) != len(set(phrases))

def score(answer: str, keywords: list) -> tuple[bool, bool]:
    a = answer.lower()
    correct = any(kw.lower() in a for kw in keywords)
    hallucination = (
        len(answer.strip()) < 20
        or any(sig in a for sig in HALLUCINATION_SIGNALS)
        or is_repetitive(answer)
    )
    return correct, hallucination

def generate(pipe, question: str) -> tuple[str, float]:
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    t0 = time.time()
    out = pipe(
        prompt,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    latency = time.time() - t0
    text = out[0]["generated_text"].split("### Response:\n")[-1].strip()
    return text, latency

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base GPT-2...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
base_pipe  = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

print("Loading fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
ft_model.eval()
ft_pipe  = pipeline("text-generation", model=ft_model, tokenizer=tokenizer)

# ── Run comparison ────────────────────────────────────────────────────────────
results = []
print(f"\nRunning {len(TESTS)} questions on both models...\n")
print(f"{'#':<3} {'Question':<45} {'Base':^7} {'Tuned':^7}")
print("-" * 70)

for i, test in enumerate(TESTS, 1):
    q, kws = test["q"], test["keywords"]

    base_ans,  base_lat  = generate(base_pipe,  q)
    tuned_ans, tuned_lat = generate(ft_pipe,     q)

    base_correct,  base_halluc  = score(base_ans,  kws)
    tuned_correct, tuned_halluc = score(tuned_ans, kws)

    base_label  = "CORRECT" if base_correct  else ("HALLUC" if base_halluc  else "WRONG")
    tuned_label = "CORRECT" if tuned_correct else ("HALLUC" if tuned_halluc else "WRONG")

    results.append({
        "question":       q,
        "base_answer":    base_ans,
        "tuned_answer":   tuned_ans,
        "base_correct":   base_correct,
        "tuned_correct":  tuned_correct,
        "base_halluc":    base_halluc,
        "tuned_halluc":   tuned_halluc,
        "base_latency":   round(base_lat, 2),
        "tuned_latency":  round(tuned_lat, 2),
    })

    print(f"{i:<3} {q[:44]:<45} {base_label:^7} {tuned_label:^7}")

# ── Summary ───────────────────────────────────────────────────────────────────
n = len(results)
base_correct  = sum(r["base_correct"]  for r in results)
tuned_correct = sum(r["tuned_correct"] for r in results)
base_halluc   = sum(r["base_halluc"]   for r in results)
tuned_halluc  = sum(r["tuned_halluc"]  for r in results)
base_lat_avg  = round(sum(r["base_latency"]  for r in results) / n, 2)
tuned_lat_avg = round(sum(r["tuned_latency"] for r in results) / n, 2)

print("\n" + "=" * 70)
print(f"{'METRIC':<30} {'BASE GPT-2':^15} {'FINE-TUNED':^15}")
print("-" * 70)
print(f"{'Correct answers':<30} {base_correct:^15} {tuned_correct:^15}")
print(f"{'Accuracy':<30} {f'{base_correct/n*100:.1f}%':^15} {f'{tuned_correct/n*100:.1f}%':^15}")
print(f"{'Hallucinations':<30} {base_halluc:^15} {tuned_halluc:^15}")
print(f"{'Avg latency (sec)':<30} {base_lat_avg:^15} {tuned_lat_avg:^15}")
print("=" * 70)

winner = "FINE-TUNED" if tuned_correct > base_correct else ("BASE GPT-2" if base_correct > tuned_correct else "TIE")
improvement = tuned_correct - base_correct
print(f"\nWinner: {winner}  |  Improvement: +{improvement} correct answers")

# ── Save full results to JSON ─────────────────────────────────────────────────
with open("comparison_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "total_questions":    n,
            "base_correct":       base_correct,
            "tuned_correct":      tuned_correct,
            "base_accuracy":      f"{base_correct/n*100:.1f}%",
            "tuned_accuracy":     f"{tuned_correct/n*100:.1f}%",
            "base_hallucinations":  base_halluc,
            "tuned_hallucinations": tuned_halluc,
            "base_avg_latency":   base_lat_avg,
            "tuned_avg_latency":  tuned_lat_avg,
            "winner":             winner,
        },
        "results": results,
    }, f, indent=2)

print("\nFull results saved to: comparison_results.json")
