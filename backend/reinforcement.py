"""
reinforcement.py — GRPO RL Pipeline + Trigger System
=====================================================
Triggered automatically when ≥ 50 new feedback pairs are in MongoDB.

trigger.py (or a cron job) calls check_and_trigger() which decides:
  ≥ 50  new pairs → run GRPO (reinforcement.py)
  ≥ 200 new pairs → run SFT+DPO (postTrainer.py)

After training:
  • GGUF is exported and registered via latestPath.set_latest_gguf()
  • Metrics (reward, perplexity, ROUGE) are saved to training_col
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch
from pymongo import MongoClient

from config import dburl
from latestPath import set_latest_gguf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("grpo_rl")

# ── MongoDB (app DB) ──────────────────────────────────────────────────────────
_mongo       = MongoClient(dburl)
_db          = _mongo["slm_app"]
jobs_col     = _db["jobs"]
training_col = _db["training_runs"]
meta_col     = _db["training_meta"]

# ── Trigger thresholds ────────────────────────────────────────────────────────
THRESHOLD_RL   = 50    # new pairs → GRPO
THRESHOLD_POST = 200   # new pairs → SFT + DPO


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class RLConfig:
    model_name: str      = "unsloth/Qwen2.5-1.5B-Instruct"
    output_dir: str      = "./models"
    max_seq_length: int  = 2048
    lora_rank: int       = 16
    lora_alpha: int      = 32
    lora_dropout: float  = 0.05
    learning_rate: float = 5e-5
    epochs: int          = 3
    batch_size: int      = 2
    grad_accumulation: int = 4
    warmup_ratio: float  = 0.1
    weight_decay: float  = 0.01
    max_prompt_length: int     = 512
    max_completion_length: int = 1024
    num_generations: int = 4   # GRPO group size G
    beta: float          = 0.001
    gguf_quant: str      = "q4_k_m"
    seed: int            = 42
    # reward weights
    w_structure: float    = 0.40
    w_completeness: float = 0.25
    w_reference: float    = 0.25
    w_length: float       = 0.10


# ── Data loading ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert HR content specialist. Convert the raw job description "
    "provided by the user into a structured, professional, ATS-friendly Markdown format."
)


def chat_format(raw_jd: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{raw_jd.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_training_pairs(since_run_id: Optional[str] = None) -> List[dict]:
    """
    Load training pairs from the app's jobs collection.
    If since_run_id is provided, only records not seen in previous runs are returned.
    Priority: user_edited > llm_output > slm_output
    """
    query: dict = {"prompt": {"$exists": True}, "slm_output": {"$exists": True}}

    # If we know the last training cutoff timestamp, only pull new records
    last_ts = _get_last_training_timestamp()
    if last_ts:
        query["updated_at"] = {"$gt": last_ts}

    records = list(jobs_col.find(query))
    log.info("New records since last run: %d", len(records))

    pairs = []
    for r in records:
        prompt     = (r.get("prompt") or "").strip()
        user_edit  = (r.get("user_edited") or "").strip()
        llm_out    = (r.get("llm_output") or "").strip()
        slm_out    = (r.get("slm_output") or "").strip()

        if not prompt or not slm_out:
            continue

        if user_edit:
            reference, quality = user_edit, "user_edited"
        elif llm_out:
            reference, quality = llm_out, "llm_output"
        else:
            reference, quality = slm_out, "slm_output"

        pairs.append({
            "job_id":    str(r.get("job_id", r.get("_id", ""))),
            "prompt":    prompt,
            "reference": reference,
            "quality":   quality,
        })

    log.info(
        "Training pairs: %d  (user_edited=%d  llm=%d  slm_only=%d)",
        len(pairs),
        sum(1 for p in pairs if p["quality"] == "user_edited"),
        sum(1 for p in pairs if p["quality"] == "llm_output"),
        sum(1 for p in pairs if p["quality"] == "slm_output"),
    )
    return pairs


def _get_last_training_timestamp() -> Optional[str]:
    """Return ISO timestamp of the last completed training run, or None."""
    doc = training_col.find_one(
        {"status": "success"},
        sort=[("finished_at", -1)],
    )
    return doc.get("finished_at") if doc else None


def count_new_pairs() -> int:
    """How many new jobs have arrived since the last training run."""
    last_ts = _get_last_training_timestamp()
    query: dict = {"prompt": {"$exists": True}, "slm_output": {"$exists": True}}
    if last_ts:
        query["updated_at"] = {"$gt": last_ts}
    return jobs_col.count_documents(query)


# ── Reward functions ──────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    r"(?i)(job\s+title|position\s+title)",
    r"(?i)(responsibilities|duties|what\s+you.ll\s+do)",
    r"(?i)(requirements|qualifications|what\s+we.re\s+looking\s+for)",
    r"(?i)(about\s+(us|the\s+company|the\s+role))",
    r"(?i)(benefits|perks|what\s+we\s+offer)",
    r"(?i)(skills|experience|education)",
]


def reward_structure(text: str) -> float:
    if not text:
        return 0.0
    headers = [l for l in text.splitlines() if re.match(r"^#{1,3}\s+\S", l.strip())]
    s  = min(len(headers) / 5, 1.0) * 0.5
    s += (sum(1 for p in SECTION_PATTERNS if re.search(p, text)) / len(SECTION_PATTERNS)) * 0.5
    return round(s, 4)


def reward_completeness(text: str) -> float:
    if not text:
        return 0.0
    checks = {
        "bullets":          bool(re.search(r"^\s*[-*•]\s+\S", text, re.MULTILINE)),
        "responsibilities": bool(re.search(r"(?i)responsibilit", text)),
        "requirements":     bool(re.search(r"(?i)(requirement|qualification)", text)),
        "summary":          bool(re.search(r"(?i)(overview|summary|about the role|position)", text)),
        "experience":       bool(re.search(r"(?i)\d+\+?\s+year", text)),
        "education":        bool(re.search(r"(?i)(bachelor|master|degree|diploma)", text)),
    }
    return round(sum(checks.values()) / len(checks), 4)


def reward_reference_f1(generated: str, reference: str) -> float:
    if not generated or not reference:
        return 0.0
    def tok(s): return set(re.findall(r"\b\w+\b", s.lower()))
    g, r = tok(generated), tok(reference)
    if not g or not r:
        return 0.0
    ov = len(g & r)
    p  = ov / len(g)
    rc = ov / len(r)
    return round(2 * p * rc / (p + rc), 4) if (p + rc) else 0.0


def reward_length(text: str, lo: int = 300, hi: int = 1200) -> float:
    w = len(text.split())
    if w < lo:  return round(w / lo, 4)
    if w > hi:  return round(max(0.0, 1.0 - (w - hi) / hi), 4)
    return 1.0


def composite_reward(generated: str, reference: str, cfg: RLConfig) -> float:
    return round(
        cfg.w_structure    * reward_structure(generated) +
        cfg.w_completeness * reward_completeness(generated) +
        cfg.w_reference    * reward_reference_f1(generated, reference) +
        cfg.w_length       * reward_length(generated),
        4,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_perplexity(model, tokenizer, pairs: List[dict], max_n: int = 30) -> float:
    model.eval()
    total_loss, total_tok = 0.0, 0
    for p in pairs[:max_n]:
        text = chat_format(p["prompt"]) + p["reference"] + "<|im_end|>"
        enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        ids  = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        total_loss += out.loss.item() * ids.shape[1]
        total_tok  += ids.shape[1]
    return round(math.exp(total_loss / total_tok), 4) if total_tok else float("inf")


def evaluate_rewards(model, tokenizer, pairs: List[dict], cfg: RLConfig, n: int = 20) -> dict:
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    import random
    sample = random.sample(pairs, min(n, len(pairs)))
    rewards = []
    for p in sample:
        inputs = tokenizer(chat_format(p["prompt"]), return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7,
                                     do_sample=True, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        rewards.append(composite_reward(gen, p["reference"], cfg))

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "avg_reward": round(avg, 4),
        "min_reward": round(min(rewards), 4) if rewards else 0.0,
        "max_reward": round(max(rewards), 4) if rewards else 0.0,
    }


# ── GGUF Export ───────────────────────────────────────────────────────────────
def export_gguf(model, tokenizer, output_dir: Path, quant: str, tag: str) -> Optional[Path]:
    final_path = output_dir / f"jd_{tag}_{quant}.gguf"
    try:
        model.save_pretrained_gguf(
            str(output_dir / f"jd_{tag}"),
            tokenizer,
            quantization_method=quant,
        )
        candidates = [p for p in output_dir.glob("*.gguf") if p != final_path]
        if candidates:
            shutil.move(str(candidates[0]), str(final_path))
        log.info("GGUF → %s", final_path)
        return final_path
    except Exception as exc:
        log.error("GGUF export failed: %s", exc)
        return None


# ── GRPO Training ─────────────────────────────────────────────────────────────
def run_grpo(cfg: RLConfig, pairs: List[dict]) -> tuple:
    """Run GRPO and return (model, tokenizer, adapter_dir, elapsed_sec)."""
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError as exc:
        log.error("Missing dep: %s  →  pip install unsloth trl datasets", exc)
        raise

    log.info("Loading model: %s", cfg.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    def make_row(p): return {"prompt": chat_format(p["prompt"]),
                              "reference": p["reference"],
                              "quality":   p["quality"]}

    dataset = Dataset.from_list([make_row(p) for p in pairs])
    split   = dataset.train_test_split(test_size=min(0.1, 20 / len(pairs)), seed=cfg.seed)

    def reward_fn(prompts, completions, **kwargs):
        refs = kwargs.get("reference", [""] * len(completions))
        return [composite_reward(g, r, cfg) for g, r in zip(completions, refs)]

    out_dir = Path(cfg.output_dir) / "grpo_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        beta=cfg.beta,
        logging_steps=5, save_steps=50,
        eval_steps=50, eval_strategy="steps",
        save_total_limit=3, load_best_model_at_end=False,
        report_to="none", seed=cfg.seed,
    )

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=reward_fn, args=grpo_cfg,
        train_dataset=split["train"], eval_dataset=split["test"],
    )

    log.info("Starting GRPO — epochs=%d  G=%d  pairs=%d", cfg.epochs, cfg.num_generations, len(pairs))
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("GRPO complete in %.1f min", elapsed / 60)

    adapter_dir = Path(cfg.output_dir) / "rl_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return model, tokenizer, adapter_dir, elapsed


# ── Main training entrypoint ──────────────────────────────────────────────────
def run_rl_pipeline(cfg: RLConfig, pairs: List[dict]):
    run_id     = f"rl_{int(time.time())}"
    started_at = datetime.now(timezone.utc).isoformat()

    model, tokenizer, adapter_dir, elapsed = run_grpo(cfg, pairs)

    # Perplexity
    ppl = compute_perplexity(model, tokenizer, pairs)
    log.info("Perplexity: %.2f", ppl)

    # Reward evaluation
    reward_metrics = evaluate_rewards(model, tokenizer, pairs, cfg)
    log.info("Rewards: %s", reward_metrics)

    # GGUF export
    models_dir = Path(cfg.output_dir)
    gguf_path  = export_gguf(model, tokenizer, models_dir, cfg.gguf_quant, "rl")
    if gguf_path:
        set_latest_gguf(str(gguf_path))

    # Save run record
    doc = {
        "run_id":       run_id,
        "pipeline":     "grpo_rl",
        "model_base":   cfg.model_name,
        "gguf_path":    str(gguf_path) if gguf_path else "export_failed",
        "started_at":   started_at,
        "finished_at":  datetime.now(timezone.utc).isoformat(),
        "duration_min": round(elapsed / 60, 2),
        "config": {
            "epochs":          cfg.epochs,
            "num_generations": cfg.num_generations,
            "lr":              cfg.learning_rate,
            "lora_rank":       cfg.lora_rank,
            "gguf_quant":      cfg.gguf_quant,
        },
        "data":    {"rl_pairs": len(pairs)},
        "metrics": {
            "perplexity": ppl,
            **reward_metrics,
        },
        "status": "success",
    }
    training_col.insert_one(doc)
    log.info("Run saved → training_runs (%s)", run_id)
    return run_id


# ── Trigger System ────────────────────────────────────────────────────────────
def check_and_trigger(force: bool = False):
    """
    Called by a cron job or a background thread.
    Checks new-pair count and fires the appropriate pipeline.
    """
    n = count_new_pairs()
    log.info("New pairs since last training: %d", n)

    if n < THRESHOLD_RL and not force:
        log.info("Below RL threshold (%d). Skipping.", THRESHOLD_RL)
        return

    pairs = load_training_pairs()
    if not pairs:
        log.warning("No usable pairs. Skipping.")
        return

    if n >= THRESHOLD_POST:
        log.info("≥%d pairs → launching SFT+DPO post-training", THRESHOLD_POST)
        # Import and run post-trainer in-process
        try:
            from postTrainer import PostTrainConfig, main as pt_main
            pt_cfg = PostTrainConfig(min_pairs=1)  # threshold already checked here
            pt_main(pt_cfg)
        except Exception as exc:
            log.error("Post-training failed: %s", exc, exc_info=True)
    else:
        log.info("≥%d pairs → launching GRPO RL", THRESHOLD_RL)
        try:
            cfg = RLConfig()
            run_rl_pipeline(cfg, pairs)
        except Exception as exc:
            log.error("GRPO training failed: %s", exc, exc_info=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GRPO RL pipeline / trigger checker")
    p.add_argument("--check",       action="store_true", help="Check trigger conditions and run if met")
    p.add_argument("--force",       action="store_true", help="Force training even if below threshold")
    p.add_argument("--model-name",  default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir",  default="./models")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--num-gen",     type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--lora-rank",   type=int,   default=16)
    p.add_argument("--gguf-quant",  default="q4_k_m")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check or args.force:
        check_and_trigger(force=args.force)
    else:
        # Direct training run
        pairs = load_training_pairs()
        if not pairs:
            log.error("No training pairs found.")
            sys.exit(1)
        cfg = RLConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            num_generations=args.num_gen,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
            gguf_quant=args.gguf_quant,
        )
        run_rl_pipeline(cfg, pairs)
