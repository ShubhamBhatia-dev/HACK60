import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 1. CONFIGURATION ---
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./qwen_outputs/rank_32/final_adapter" 
input_json_path = "11.json"  # Aapki original JSON file ka naam
output_json_path = "dpo_prepared_data.json"

# System Prompt jo aapne training mein use kiya tha (Important for consistency)
SYSTEM_PROMPT = """You are an expert Technical HR Recruiter. 
Transform the raw job description into the exact markdown structure provided.
## Job Title
## Location
## Client Industry
## Detailed Responsibilities
## Skill Requirements
## Other Requirements"""

# --- 2. LOAD MODEL & TOKENIZER ---
print("Loading Model and Adapter...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# --- 3. LOAD INPUT DATA ---
with open(input_json_path, 'r') as f:
    data = json.load(f)

# Pehle 200 records lena
subset_data = data[:200] 
final_dpo_list = []

# --- 4. INFERENCE LOOP ---
print(f"Generating outputs for 200 samples...")
for item in tqdm(subset_data):
    raw_text = item['raw_input']
    llm_output = item['output'] # Ye aapka 'Chosen' reference banega
    
    # Qwen ChatML format build karna
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{raw_text}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # Keep it low for stable formatting
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Output decode karna (sirf assistant ka part)
    decoded_output = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Naya structure banana
    final_dpo_list.append({
        "raw_text": raw_text,
        "slm_output": decoded_output.strip(), # Ye aapka 'Rejected' (bura) candidate ho sakta hai
        "llm_output": llm_output.strip()      # Ye aapka 'Chosen' (accha) candidate hai
    })

# --- 5. SAVE NEW JSON ---
with open(output_json_path, 'w') as f:
    json.dump(final_dpo_list, f, indent=4)

print(f"\nDone! Saved to {output_json_path}")
