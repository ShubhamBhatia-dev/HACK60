import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 1. CONFIGURATION ---
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
# Apne best adapter ka path yahan daalein (e.g., rank_32 ya rank_64)
adapter_path = "./qwen_outputs/rank_32/final_adapter" 

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

# --- 2. LOAD MODEL & TOKENIZER ---
print(f"Loading Tokenizer from Base Model ({base_model_id})...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading Base Model in 4-bit...")
# [FIX]: Use BitsAndBytesConfig instead of passing arguments directly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map="auto", 
    quantization_config=bnb_config # Ye sahi tarika hai 4-bit load karne ka
)

print(f"Applying LoRA Adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# --- 3. INFERENCE FUNCTION ---
def enhance_jd(raw_jd_text):
    print("\n--- Generating Enhanced JD ---\n")
    
    # Qwen ChatML Format
    full_prompt = (
        f"<|im_start|>system\n{TEACHER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nTransform this raw JD:\n{raw_jd_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,       # Generation limit
            temperature=0.3,          # Low temp for factual and strict formatting
            top_p=0.9,
            repetition_penalty=1.1,   # Prevents looping/repeating
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Prompt ke baad wale generated tokens ko extract karna
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    enhanced_jd = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return enhanced_jd

# --- 4. TEST EXECUTION ---
if __name__ == "__main__":
    # Test karne ke liye yahan koi bhi kaccha (raw) JD daalein
    sample_raw_jd = """
    gym trainer, 6 yr exp
    """
    
    print("\n[INPUT RAW JD]")
    print(sample_raw_jd.strip())
    
    try:
        result = enhance_jd(sample_raw_jd)
        print("\n[OUTPUT ENHANCED JD]")
        print("="*50)
        print(result)
        print("="*50)
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
