import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
# Ensure ye wahi path hai jahan aapka best adapter save hua hai
adapter_path = "./qwen_outputs/rank_32/final_adapter" 
# Naya folder jahan merged model save hoga
merged_save_path = "./qwen_merged_model" 

print(f"Loading Base Model: {base_model_id} (in bfloat16)")
# Note: Hum quantization (4-bit) use nahi kar rahe kyunki merge original precision mein karna hota hai
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.bfloat16,
    device_map="cpu", # CPU par merge karenge taaki GPU memory ka issue na aaye
)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print(f"Applying LoRA Adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights... (This might take a minute)")
# Ye function adapter weights ko base weights mein add kar deta hai
merged_model = model.merge_and_unload() 

print(f"Saving fully merged model to {merged_save_path}...")
merged_model.save_pretrained(merged_save_path)
tokenizer.save_pretrained(merged_save_path)

print("Merging Complete! Ready for GGUF conversion.")
