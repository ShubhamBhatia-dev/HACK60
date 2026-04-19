import torch
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# --- 1. CONFIGURATION ---
model_id = "Qwen/Qwen2.5-3B-Instruct"  
dataset_file = "jd_dataset_final.json"
ranks_to_test = [32]
all_metrics = {}

TEACHER_SYSTEM_PROMPT = """You are an expert Technical HR Recruiter. 
Transform the raw job description into the exact markdown structure provided.
CRITICAL RULES:
1. Professional Polishing: Flesh out keywords into professional bullet points.
2. ZERO Hallucinations: Do not invent skills.
3. CLIENT INDUSTRY RULE: ONLY extract the industry if explicitly stated. Otherwise, write 'Not mentioned'. DO NOT default to 'Multinational Technology Provider'.

EXPECTED STRUCTURE:
## Job Title
## Location
## Client Industry
## Detailed Responsibilities
## Skill Requirements
## Other Requirements"""

REQUIRED_HEADINGS = [
    "## Job Title", "## Location", "## Client Industry", 
    "## Detailed Responsibilities", "## Skill Requirements", "## Other Requirements"
]

# --- 2. DATA PREP ---
dataset = load_dataset("json", data_files=dataset_file, split="train").train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  

def format_and_tokenize(example):
    full_prompt = (
        f"<|im_start|>system\n{TEACHER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nTransform this raw JD:\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return tokenizer(full_prompt, truncation=True, max_length=768, padding="max_length")

tokenized_data = dataset.map(format_and_tokenize, remove_columns=dataset["train"].column_names)

# --- 3. METRICS SETUP ---
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def preprocess_logits_for_metrics(logits, labels):
    """
    OOM FIX: Logits ko pehle hi tokens mein convert kar deta hai taaki 
    RAM mein 8GB ka spike na aaye.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_res = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    refs_for_bleu = [[ref] for ref in decoded_labels]
    bleu_res = bleu_metric.compute(predictions=decoded_preds, references=refs_for_bleu)
    
    total_expected = len(decoded_preds) * len(REQUIRED_HEADINGS)
    found_headings = 0
    for pred in decoded_preds:
        for heading in REQUIRED_HEADINGS:
            if heading in pred:
                found_headings += 1
    structure_score = (found_headings / total_expected) * 100 if total_expected > 0 else 0

    return {
        "rouge1": round(rouge_res.get("rouge1", 0), 4),
        "rougeL": round(rouge_res.get("rougeL", 0), 4),
        "bleu": round(bleu_res.get("bleu", 0), 4),
        "structure_completeness_percent": round(structure_score, 2)
    }

# --- 4. TRAINING LOOP ---
for r in ranks_to_test:
    print(f"\n{'='*40}\n>>> RUNNING RANK: {r}\n{'='*40}")
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=r, lora_alpha=r*2, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    output_dir = f"./qwen_outputs/rank_{r}"
    
    args = TrainingArguments(
        output_dir=output_dir, 
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=1,        # OOM Fix
        eval_accumulation_steps=1,           # OOM Fix
        gradient_accumulation_steps=4,
        learning_rate=2e-4, 
        num_train_epochs=20, 
        eval_strategy="epoch", 
        logging_steps=10, 
        bf16=True, 
        optim="paged_adamw_8bit", 
        gradient_checkpointing=True,         # OOM Fix
        report_to="none"
    )

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=tokenized_data["train"], 
        eval_dataset=tokenized_data["test"], 
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics # Passed to Trainer
    )

    trainer.train()
    
    eval_results = trainer.evaluate()
    try:
        perplexity = math.exp(eval_results['eval_loss'])
    except OverflowError:
        perplexity = float("inf")
    
    print(f"\nFinal Perplexity for Rank {r}: {perplexity:.2f}")
    trainer.state.log_history.append({'step': trainer.state.global_step, 'eval_perplexity': perplexity})
    
    model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    all_metrics[f"Rank_{r}"] = trainer.state.log_history

    del model, trainer
    torch.cuda.empty_cache()

# --- 5. GRAPHS ---
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
for label, history in all_metrics.items():
    steps = [x['step'] for x in history if 'loss' in x]
    loss = [x['loss'] for x in history if 'loss' in x]
    plt.plot(steps, loss, label=f"Train Loss ({label})")
plt.title("Training Loss (Lower is better)")
plt.xlabel("Steps"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

plt.subplot(2, 2, 2)
for label, history in all_metrics.items():
    eval_steps = [x['step'] for x in history if 'eval_rougeL' in x]
    rouge_l = [x['eval_rougeL'] for x in history if 'eval_rougeL' in x]
    bleu = [x['eval_bleu'] for x in history if 'eval_bleu' in x]
    if eval_steps: 
        plt.plot(eval_steps, rouge_l, marker='o', linestyle='-', label=f"ROUGE-L ({label})")
        plt.plot(eval_steps, bleu, marker='x', linestyle='--', alpha=0.6, label=f"BLEU ({label})")
plt.title("Text Quality: ROUGE-L & BLEU")
plt.xlabel("Steps"); plt.ylabel("Score"); plt.legend(); plt.grid(True)

plt.subplot(2, 2, 3)
for label, history in all_metrics.items():
    eval_steps = [x['step'] for x in history if 'eval_structure_completeness_percent' in x]
    structure = [x['eval_structure_completeness_percent'] for x in history if 'eval_structure_completeness_percent' in x]
    if eval_steps:
        plt.plot(eval_steps, structure, marker='s', label=f"Structure Match % ({label})")
plt.title("Structural Completeness (% of Expected Headings)")
plt.xlabel("Steps"); plt.ylabel("Match Percentage"); plt.legend(); plt.grid(True)
plt.ylim(0, 105)

plt.subplot(2, 2, 4)
for label, history in all_metrics.items():
    ppl_values = [x['eval_perplexity'] for x in history if 'eval_perplexity' in x]
    if ppl_values:
        plt.bar(label, ppl_values[-1], alpha=0.7)
plt.title("Final Model Perplexity (Lower is better)")
plt.ylabel("Perplexity Score"); plt.grid(axis='y')

plt.tight_layout()
plt.savefig("qwen_comprehensive_metrics.png")
plt.show()
