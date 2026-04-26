"""
CPU-friendly LoRA fine-tuning on the ML Q&A dataset.
Model  : GPT-2 (124M params, ~500MB RAM — CPU safe)
Method : LoRA (r=8) — only ~0.3% of parameters are trained
Time   : ~10-20 min on CPU
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "gpt2"
OUTPUT_DIR  = "./ml_adapter"
MAX_SEQ_LEN = 256   # keep short for CPU speed
EPOCHS      = 3
BATCH_SIZE  = 1
GRAD_ACCUM  = 8     # effective batch = 8

# ── Prompt format ─────────────────────────────────────────────────────────────
def format_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load model (float32 for CPU) ──────────────────────────────────────────────
print("Loading model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

# ── Apply LoRA ────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 attention projection layer
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train":      "dataset/train.jsonl",
        "validation": "dataset/val.jsonl",
    },
)

# ── Training arguments (SFTConfig includes max_seq_length in trl 1.x) ─────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_SEQ_LEN,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    bf16=False,
    report_to="none",
    optim="adamw_torch",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=format_prompt,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting LoRA fine-tuning...")
print(f"  Model   : {MODEL_NAME}")
print(f"  Samples : {len(dataset['train'])} train / {len(dataset['validation'])} val")
print(f"  Epochs  : {EPOCHS}")
print(f"  Device  : {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

trainer.train()

# ── Save adapter only (tiny — a few MB) ──────────────────────────────────────
print("\nSaving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to: {OUTPUT_DIR}/")
print("Done!")
