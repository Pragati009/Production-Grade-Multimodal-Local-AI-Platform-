"""
Test the fine-tuned LoRA adapter against the base model.
Run this after finetune.py completes.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "./ml_adapter"

TEST_QUESTIONS = [
    "What is Naive Bayes?",
    "What is LoRA?",
    "What is the difference between precision and recall?",
    "What is gradient descent?",
    "What is overfitting?",
]

def generate(pipe, question, max_new_tokens=120):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    response = out[0]["generated_text"].split("### Response:\n")[-1].strip()
    return response

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

print("Loading fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
ft_model.eval()
ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=tokenizer)

print("\n" + "=" * 60)
for q in TEST_QUESTIONS:
    print(f"\nQ: {q}")
    print(f"\n[Base Model]\n{generate(base_pipe, q)}")
    print(f"\n[Fine-Tuned]\n{generate(ft_pipe, q)}")
    print("-" * 60)
