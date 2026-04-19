# === HACKATHON BYPASS FOR TRL FSDP BUG ===
import torch
import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    torch.distributed.fsdp.FSDPModule = type("DummyFSDPModule", (), {})
# =========================================

import json
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from sklearn.model_selection import train_test_split

# --- 1. SETTINGS ---
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./qwen_outputs/rank_32/final_adapter" 
json_file = "final_output.json" 
new_output_dir = "./qwen-dpo-v2-r32-final"

# --- 2. DATA PREPARATION ---
def prepare_dpo_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Split: 90% training, 10% eval
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    def format_dict(data_list):
        return {
            "prompt": [f"Transform this raw JD: {item['raw_text']}" for item in data_list],
            "chosen": [item['llm_output'] for item in data_list],
            "rejected": [item['slm_output'] for item in data_list]
        }
        
    return Dataset.from_dict(format_dict(train_data)), Dataset.from_dict(format_dict(test_data))

train_dataset, eval_dataset = prepare_dpo_dataset(json_file)
print(f"Dataset Ready: {len(train_dataset)} train samples.")

# --- 3. LOAD & MERGE SFT ADAPTER ---
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Base Model and Merging SFT Weights...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True
)

# SFT merge karna zaroori hai taaki DPO reference model ke liye clear starting point ho
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() 
print("Merge Successful. Model is now ready for DPO Alignment.")

# --- 4. DPO CONFIGURATION (Optimized for Stability) ---
dpo_lora_config = LoraConfig(
    r=32,            
    lora_alpha=64,   
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir=new_output_dir,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, # More stable gradients
    learning_rate=1e-7,            # Very low LR to prevent performance drop
    num_train_epochs=1,
    logging_steps=5,
    bf16=True,
    optim="adamw_torch",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    report_to="none", 
    beta=0.1,                      # Strength of DPO penalty
)

# --- 5. INITIALIZE TRAINER ---
dpo_trainer = DPOTrainer(
    model,
    ref_model=None, # TRL will automatically create a frozen copy as reference
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer, 
    peft_config=dpo_lora_config,
)

# --- 6. TRAIN & SAVE ---
print("\nStarting DPO Training Loop...")
dpo_trainer.train()

print(f"\nSaving DPO Adapter to {new_output_dir}...")
dpo_trainer.save_model(new_output_dir)

# --- 7. AUTOMATED EVALUATION & GRAPH ---
print("\n--- Generating Comparison Graph (SFT vs DPO) ---")
torch.cuda.empty_cache() 
model.eval()

rouge = evaluate.load("rouge")
required_headings = ["## Job Title", "## Location", "## Client Industry", "## Detailed Responsibilities", "## Skill Requirements"]

sft_acc_scores, dpo_acc_scores = [], []
sft_fmt_scores, dpo_fmt_scores = [], []

# Evaluation on a larger subset of test data for better accuracy
test_samples = eval_dataset.select(range(min(20, len(eval_dataset))))

for example in test_samples:
    inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
    
    dpo_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # ROUGE-L Accuracy
    dpo_acc_scores.append(rouge.compute(predictions=[dpo_text], references=[example["chosen"]])['rougeL'] * 100)
    sft_acc_scores.append(rouge.compute(predictions=[example["rejected"]], references=[example["chosen"]])['rougeL'] * 100)
    
    # Format Check
    dpo_fmt_scores.append(sum([1 for h in required_headings if h in dpo_text]) / len(required_headings) * 100)
    sft_fmt_scores.append(sum([1 for h in required_headings if h in example["rejected"]]) / len(required_headings) * 100)

# Final Visualization
metrics = ['Accuracy (ROUGE-L)', 'Format Completeness']
sft_final = [np.mean(sft_acc_scores), np.mean(sft_fmt_scores)]
dpo_final = [np.mean(dpo_acc_scores), np.mean(dpo_fmt_scores)]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(x - width/2, sft_final, width, label='SFT Only (Base)', color='#ff9999', alpha=0.8)
ax.bar(x + width/2, dpo_final, width, label='SFT + DPO Aligned', color='#99ff99', alpha=0.9)

ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('DPO Alignment Impact: Boosting Accuracy & Formatting', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend()

# Adding data labels on bars
for i, v in enumerate(sft_final):
    ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
for i, v in enumerate(dpo_final):
    ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

plt.ylim(0, 115)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('dpo_metrics_final.png', dpi=300, bbox_inches='tight')

print("\nSuccess! Results saved in 'dpo_metrics_final.png'.")
