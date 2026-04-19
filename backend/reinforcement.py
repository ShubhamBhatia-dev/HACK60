"""
Self-Learning JD Enhancement System — GRPO Reinforcement Learning Pipeline
===========================================================================
Uses Group Relative Policy Optimization (GRPO) via Unsloth for efficient
fine-tuning of a <4B parameter SLM on job description enhancement.

MongoDB schema assumed:
  {
    job_id: str,
    prompt: str,           # raw JD input
    slm_output: str,       # SLM-generated JD (never null)
    llm_output: str | None,  # LLM-enhanced JD (may be null)
    user_edited: str | None  # human-edited output (may be null)
  }

Reward hierarchy (highest to lowest quality signal):
  user_edited  >  llm_output  >  slm_output

Usage:
  python rl_grpo_training.py --mongo-uri "mongodb://localhost:27017" \
                              --db-name "jd_system" \
                              --collection "jd_records" \
                              --model-name "unsloth/Qwen2.5-1.5B-Instruct" \
                              --output-dir "./models" \
                              --epochs 3

After training the LoRA adapter is merged and exported to GGUF (Q4_K_M).
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("grpo_rl")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "jd_system"
    collection: str = "jd_records"
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct"   # <4B, good instruction model
    output_dir: str = "./models"
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size: int = 2          # per-device; GRPO doubles memory vs SFT
    grad_accumulation: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    num_generations: int = 4     # G in GRPO (group size)
    beta: float = 0.001          # KL penalty
    min_samples: int = 30        # abort if fewer usable records
    gguf_quant: str = "q4_k_m"
    seed: int = 42

    # Reward weights
    w_structure: float = 0.40
    w_completeness: float = 0.25
    w_reference: float = 0.25
    w_length: float = 0.10


# ---------------------------------------------------------------------------
# MongoDB data loading
# ---------------------------------------------------------------------------
REQUIRED_SECTIONS = [
    "job title",
    "responsibilities",
    "requirements",
    "qualifications",
    "about",
]

def load_training_pairs(cfg: TrainConfig) -> List[dict]:
    """
    Pull records from MongoDB and build (prompt, reference) pairs.
    Reference quality: user_edited > llm_output > slm_output
    """
    client = MongoClient(cfg.mongo_uri)
    db = client[cfg.db_name]
    col = db[cfg.collection]

    records = list(col.find({"prompt": {"$exists": True}, "slm_output": {"$exists": True}}))
    log.info("Fetched %d records from MongoDB", len(records))

    pairs = []
    for r in records:
        prompt = (r.get("prompt") or "").strip()
        user_edited = (r.get("user_edited") or "").strip()
        llm_output = (r.get("llm_output") or "").strip()
        slm_output = (r.get("slm_output") or "").strip()

        if not prompt or not slm_output:
            continue

        # Pick best available reference
        if user_edited:
            reference = user_edited
            quality = "user_edited"
        elif llm_output:
            reference = llm_output
            quality = "llm_output"
        else:
            # slm_output as self-reference has low learning signal; skip unless
            # it's the only data we have
            reference = slm_output
            quality = "slm_output"

        pairs.append({
            "job_id": str(r.get("job_id", r.get("_id", ""))),
            "prompt": prompt,
            "reference": reference,
            "quality": quality,
        })

    client.close()
    log.info(
        "Training pairs: %d  (user_edited=%d  llm_output=%d  slm_only=%d)",
        len(pairs),
        sum(1 for p in pairs if p["quality"] == "user_edited"),
        sum(1 for p in pairs if p["quality"] == "llm_output"),
        sum(1 for p in pairs if p["quality"] == "slm_output"),
    )
    return pairs


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
SECTION_PATTERNS = [
    r"(?i)(job\s+title|position\s+title)",
    r"(?i)(responsibilities|duties|what\s+you.ll\s+do)",
    r"(?i)(requirements|qualifications|what\s+we.re\s+looking\s+for)",
    r"(?i)(about\s+(us|the\s+company|the\s+role))",
    r"(?i)(benefits|perks|what\s+we\s+offer)",
    r"(?i)(skills|experience|education)",
]

def reward_structure(text: str) -> float:
    """Reward for presence of well-formatted section headers."""
    if not text:
        return 0.0
    score = 0.0
    # Check markdown headers (## Section) or bold-ish markers
    header_lines = [l for l in text.splitlines() if re.match(r"^#{1,3}\s+\S", l.strip())]
    score += min(len(header_lines) / 5, 1.0) * 0.5

    # Check recognisable section names
    matched = sum(1 for pat in SECTION_PATTERNS if re.search(pat, text))
    score += (matched / len(SECTION_PATTERNS)) * 0.5
    return round(score, 4)


def reward_completeness(text: str) -> float:
    """Reward for covering expected JD content areas."""
    if not text:
        return 0.0
    checks = {
        "has_bullets": bool(re.search(r"^\s*[-*•]\s+\S", text, re.MULTILINE)),
        "has_responsibilities": bool(re.search(r"(?i)responsibilit", text)),
        "has_requirements": bool(re.search(r"(?i)(requirement|qualification)", text)),
        "has_role_summary": bool(re.search(r"(?i)(overview|summary|about the role|position)", text)),
        "has_experience": bool(re.search(r"(?i)\d+\+?\s+year", text)),
        "has_education": bool(re.search(r"(?i)(bachelor|master|degree|diploma)", text)),
    }
    return round(sum(checks.values()) / len(checks), 4)


def reward_reference_similarity(generated: str, reference: str) -> float:
    """
    Lightweight token-level F1 (similar to ROUGE-1) without external deps.
    Avoids importing rouge_score at reward-call frequency.
    """
    if not generated or not reference:
        return 0.0

    def tokenize(s: str):
        return set(re.findall(r"\b\w+\b", s.lower()))

    gen_tokens = tokenize(generated)
    ref_tokens = tokenize(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0

    overlap = len(gen_tokens & ref_tokens)
    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def reward_length(text: str, target_min: int = 300, target_max: int = 1200) -> float:
    """Penalise outputs that are too short or excessively long."""
    if not text:
        return 0.0
    words = len(text.split())
    if words < target_min:
        return round(words / target_min, 4)
    if words > target_max:
        return round(max(0.0, 1.0 - (words - target_max) / target_max), 4)
    return 1.0


def compute_reward(generated: str, reference: str, cfg: TrainConfig) -> float:
    """Composite reward in [0, 1]."""
    r_struct = reward_structure(generated)
    r_complete = reward_completeness(generated)
    r_ref = reward_reference_similarity(generated, reference)
    r_len = reward_length(generated)

    composite = (
        cfg.w_structure    * r_struct   +
        cfg.w_completeness * r_complete +
        cfg.w_reference    * r_ref      +
        cfg.w_length       * r_len
    )
    return round(composite, 4)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert HR content specialist. Convert the raw job description "
    "provided by the user into a structured, professional, ATS-friendly format. "
    "Use clear markdown section headers (##), bullet points for responsibilities "
    "and requirements, and ensure all key sections are present."
)

def format_chat_prompt(raw_jd: str) -> str:
    """Return the full instruction string for the model."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{raw_jd.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------
def train_grpo(cfg: TrainConfig, pairs: List[dict]):
    """Main GRPO training loop using Unsloth + TRL."""
    # ---- lazy imports so the script can be imported without GPU ----
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError as exc:
        log.error(
            "Missing dependency: %s\n"
            "Install with:  pip install unsloth trl datasets\n"
            "Unsloth GPU:   pip install unsloth[cu121-torch240]",
            exc,
        )
        sys.exit(1)

    if len(pairs) < cfg.min_samples:
        log.error(
            "Only %d training pairs found (minimum %d). "
            "Collect more feedback data before running RL.",
            len(pairs), cfg.min_samples,
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # 1. Load base model with Unsloth (4-bit QLoRA)
    # ----------------------------------------------------------------
    log.info("Loading model: %s", cfg.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,          # auto-detect bf16 / fp16
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    log.info("LoRA adapter attached (rank=%d, alpha=%d)", cfg.lora_rank, cfg.lora_alpha)

    # ----------------------------------------------------------------
    # 2. Build HuggingFace Dataset
    # ----------------------------------------------------------------
    def make_row(pair: dict) -> dict:
        return {
            "prompt": format_chat_prompt(pair["prompt"]),
            "reference": pair["reference"],
            "quality": pair["quality"],
        }

    dataset = Dataset.from_list([make_row(p) for p in pairs])
    split = dataset.train_test_split(test_size=min(0.1, 20 / len(pairs)), seed=cfg.seed)
    train_ds = split["train"]
    eval_ds  = split["test"]
    log.info("Dataset split — train: %d  eval: %d", len(train_ds), len(eval_ds))

    # ----------------------------------------------------------------
    # 3. Define reward function for TRL's GRPOTrainer
    #    TRL calls: reward_fn(prompts, completions, **kwargs) -> List[float]
    # ----------------------------------------------------------------
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        references = kwargs.get("reference", [""] * len(completions))
        rewards = []
        for gen, ref in zip(completions, references):
            r = compute_reward(gen, ref, cfg)
            rewards.append(r)
        return rewards

    # ----------------------------------------------------------------
    # 4. GRPOConfig
    # ----------------------------------------------------------------
    output_dir = Path(cfg.output_dir) / "grpo_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # GRPO-specific
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        beta=cfg.beta,
        # Logging & saving
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        seed=cfg.seed,
        # Reward normalisation (GRPO standard)
        reward_baseline=None,
    )

    # ----------------------------------------------------------------
    # 5. Trainer
    # ----------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    log.info("Starting GRPO training — epochs=%d  batch=%d  G=%d",
             cfg.epochs, cfg.batch_size, cfg.num_generations)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("Training complete in %.1f min", elapsed / 60)

    # ----------------------------------------------------------------
    # 6. Save LoRA adapter
    # ----------------------------------------------------------------
    adapter_dir = Path(cfg.output_dir) / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("LoRA adapter saved → %s", adapter_dir)

    return model, tokenizer, adapter_dir


# ---------------------------------------------------------------------------
# Merge + GGUF Export
# ---------------------------------------------------------------------------
def export_gguf(cfg: TrainConfig, model, tokenizer, adapter_dir: Path):
    """Merge LoRA weights and convert to GGUF for llama.cpp inference."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log.error("Unsloth not available for GGUF export.")
        return

    models_dir = Path(cfg.output_dir)
    merged_dir  = models_dir / "merged_model"
    gguf_path   = models_dir / f"jd_enhancer_{cfg.gguf_quant}.gguf"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Merge LoRA into base weights ---
    log.info("Merging LoRA adapter into base model weights …")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    log.info("Merged model saved → %s", merged_dir)

    # --- Convert to GGUF using Unsloth's built-in helper ---
    log.info("Converting to GGUF (%s) …", cfg.gguf_quant.upper())
    try:
        model.save_pretrained_gguf(
            str(models_dir / "jd_enhancer"),
            tokenizer,
            quantization_method=cfg.gguf_quant,
        )
        # Unsloth names the file automatically; rename for clarity
        candidates = list(models_dir.glob("*.gguf"))
        if candidates:
            final = models_dir / f"jd_enhancer_{cfg.gguf_quant}.gguf"
            if candidates[0] != final:
                shutil.move(str(candidates[0]), str(final))
            log.info("GGUF model saved → %s", final)
        else:
            log.warning("GGUF file not found after export — check Unsloth output above.")
    except Exception as exc:
        log.error("Unsloth GGUF export failed: %s", exc)
        log.info("Falling back to llama.cpp convert script …")
        _fallback_gguf_convert(cfg, merged_dir, models_dir)


def _fallback_gguf_convert(cfg: TrainConfig, merged_dir: Path, models_dir: Path):
    """
    Fallback: use llama.cpp's convert_hf_to_gguf.py if Unsloth export fails.
    Requires llama.cpp to be cloned locally or installed as a package.
    """
    convert_script = shutil.which("convert_hf_to_gguf.py")
    if convert_script is None:
        # Try common locations
        candidates = [
            Path("llama.cpp/convert_hf_to_gguf.py"),
            Path(os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")),
        ]
        for c in candidates:
            if c.exists():
                convert_script = str(c)
                break

    if convert_script is None:
        log.error(
            "convert_hf_to_gguf.py not found. Clone llama.cpp and ensure the "
            "script is on PATH, or install: pip install llama-cpp-python"
        )
        return

    fp16_gguf = models_dir / "jd_enhancer_f16.gguf"
    cmd_convert = [
        sys.executable, convert_script,
        str(merged_dir),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ]
    log.info("Running: %s", " ".join(cmd_convert))
    subprocess.run(cmd_convert, check=True)

    # Quantise
    quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
    if quantize_bin:
        q_type = cfg.gguf_quant.upper().replace("_", "_")
        final_gguf = models_dir / f"jd_enhancer_{cfg.gguf_quant}.gguf"
        cmd_quant = [quantize_bin, str(fp16_gguf), str(final_gguf), q_type]
        log.info("Quantising: %s", " ".join(cmd_quant))
        subprocess.run(cmd_quant, check=True)
        fp16_gguf.unlink(missing_ok=True)
        log.info("Quantised GGUF → %s", final_gguf)
    else:
        log.warning("llama-quantize not found; leaving f16 GGUF at %s", fp16_gguf)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_model(cfg: TrainConfig, model, tokenizer, pairs: List[dict], n_samples: int = 20):
    """
    Quick evaluation: generate outputs for a random sample and compute
    composite rewards. Prints a mini report.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log.warning("Unsloth not available for evaluation.")
        return

    FastLanguageModel.for_inference(model)
    import random
    random.seed(cfg.seed)
    sample = random.sample(pairs, min(n_samples, len(pairs)))

    rewards = []
    for pair in sample:
        prompt_text = format_chat_prompt(pair["prompt"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_completion_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        r = compute_reward(generated, pair["reference"], cfg)
        rewards.append(r)

    avg_r = sum(rewards) / len(rewards)
    log.info(
        "Eval on %d samples — avg_reward=%.4f  min=%.4f  max=%.4f",
        len(rewards), avg_r, min(rewards), max(rewards),
    )
    return avg_r


# ---------------------------------------------------------------------------
# Synthetic data bootstrapper (for first-run when DB is sparse)
# ---------------------------------------------------------------------------
SYNTHETIC_PAIRS = [
    {
        "prompt": "Need a python dev. Must know flask and sql. 3 yrs exp. Full time. NYC.",
        "reference": """## Software Engineer — Python / Backend

**Location:** New York City, NY | **Employment Type:** Full-Time

---

## About the Role

We are looking for a skilled Python Developer to join our engineering team. You will
design, build, and maintain backend services that power our core products.

## Responsibilities

- Develop and maintain RESTful APIs using Flask
- Design and optimise SQL database schemas and queries
- Collaborate with front-end engineers to integrate user-facing elements
- Participate in code reviews and contribute to engineering best practices
- Troubleshoot and debug production issues in a timely manner

## Requirements

- 3+ years of professional Python development experience
- Proficiency with Flask or a similar Python web framework
- Strong command of relational databases (PostgreSQL / MySQL)
- Familiarity with version control systems (Git)
- Strong problem-solving and communication skills

## Nice to Have

- Experience with Docker and CI/CD pipelines
- Knowledge of cloud platforms (AWS / GCP / Azure)

## What We Offer

- Competitive salary and equity
- Comprehensive health, dental, and vision insurance
- Flexible working hours and remote-friendly culture
""",
    },
    {
        "prompt": "Hiring data scientist. ML exp needed. Python, TensorFlow. Masters preferred. Remote.",
        "reference": """## Data Scientist

**Location:** Remote | **Employment Type:** Full-Time

---

## About the Role

Join our data science team to build and deploy machine-learning models that drive
product decisions and business growth.

## Responsibilities

- Design, train, and evaluate ML / deep-learning models using Python and TensorFlow
- Analyse large datasets to extract actionable insights
- Collaborate with product and engineering teams to deliver data-driven features
- Monitor model performance in production and iterate based on results
- Communicate findings clearly to both technical and non-technical stakeholders

## Requirements

- Proven experience in applied machine learning
- Strong proficiency in Python and TensorFlow (or PyTorch)
- Experience with data wrangling, feature engineering, and statistical analysis
- Master's degree in Computer Science, Statistics, or a related field preferred
- Ability to work independently in a fully remote environment

## Nice to Have

- Experience with MLOps tools (MLflow, Kubeflow)
- Familiarity with cloud ML platforms (SageMaker, Vertex AI)

## Benefits

- Fully remote position with flexible hours
- Learning and development budget
- Competitive compensation package
""",
    },
]

def maybe_bootstrap(cfg: TrainConfig, pairs: List[dict]) -> List[dict]:
    """If fewer than min_samples records, add synthetic pairs to bootstrap."""
    if len(pairs) >= cfg.min_samples:
        return pairs
    needed = cfg.min_samples - len(pairs)
    log.warning(
        "Only %d DB records; augmenting with %d synthetic pairs to reach minimum %d.",
        len(pairs), min(needed, len(SYNTHETIC_PAIRS)), cfg.min_samples,
    )
    extras = []
    for i in range(needed):
        base = SYNTHETIC_PAIRS[i % len(SYNTHETIC_PAIRS)]
        extras.append({
            "job_id": f"synthetic_{i}",
            "prompt": base["prompt"],
            "reference": base["reference"],
            "quality": "synthetic",
        })
    return pairs + extras


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="GRPO RL fine-tuning for JD enhancement")
    p.add_argument("--mongo-uri",     default="mongodb://localhost:27017")
    p.add_argument("--db-name",       default="jd_system")
    p.add_argument("--collection",    default="jd_records")
    p.add_argument("--model-name",    default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir",    default="./models")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=2)
    p.add_argument("--lora-rank",     type=int,   default=16)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--num-gen",       type=int,   default=4,
                   help="GRPO group size G (number of completions per prompt)")
    p.add_argument("--beta",          type=float, default=0.001,
                   help="KL divergence penalty coefficient")
    p.add_argument("--gguf-quant",    default="q4_k_m",
                   choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    p.add_argument("--min-samples",   type=int,   default=30)
    p.add_argument("--skip-gguf",     action="store_true",
                   help="Skip GGUF conversion (useful for quick tests)")
    p.add_argument("--eval-only",     action="store_true",
                   help="Load adapter from output-dir and evaluate without training")

    args = p.parse_args()
    cfg = TrainConfig(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection=args.collection,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        num_generations=args.num_gen,
        beta=args.beta,
        gguf_quant=args.gguf_quant,
        min_samples=args.min_samples,
    )
    return cfg, args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    cfg, args = parse_args()

    log.info("=" * 60)
    log.info("  GRPO RL Pipeline — JD Enhancement System")
    log.info("=" * 60)
    log.info("Model      : %s", cfg.model_name)
    log.info("Output dir : %s", cfg.output_dir)
    log.info("Epochs     : %d", cfg.epochs)
    log.info("Group size : %d", cfg.num_generations)
    log.info("GGUF quant : %s", cfg.gguf_quant)

    # 1. Load data
    pairs = load_training_pairs(cfg)
    pairs = maybe_bootstrap(cfg, pairs)

    if args.eval_only:
        # Load saved adapter for evaluation only
        log.info("Eval-only mode: loading adapter from %s/lora_adapter", cfg.output_dir)
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            log.error("Unsloth required for eval-only mode.")
            sys.exit(1)
        adapter_dir = Path(cfg.output_dir) / "lora_adapter"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        evaluate_model(cfg, model, tokenizer, pairs)
        return

    # 2. Train
    model, tokenizer, adapter_dir = train_grpo(cfg, pairs)

    # 3. Evaluate post-training
    log.info("Running post-training evaluation …")
    evaluate_model(cfg, model, tokenizer, pairs)

    # 4. Export GGUF
    if not args.skip_gguf:
        export_gguf(cfg, model, tokenizer, adapter_dir)
    else:
        log.info("Skipping GGUF export (--skip-gguf set).")

    log.info("Pipeline complete. Models saved in: %s", cfg.output_dir)
    log.info(
        "Update your llama.cpp server to point at:  %s/jd_enhancer_%s.gguf",
        cfg.output_dir, cfg.gguf_quant,
    )


