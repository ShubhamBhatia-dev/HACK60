"""
postTrainer.py — SFT + DPO Post-Training Pipeline
===================================================
Triggered when ≥ 200 new feedback pairs are available in MongoDB.

Two-stage training:
  Stage 1 – Supervised Fine-Tuning (SFT)
    • Uses the best available reference (user_edited > llm_output > slm_output)
    • Standard causal language-modelling loss
    • Quick convergence, strong task alignment

  Stage 2 – Direct Preference Optimisation (DPO)
    • Requires (prompt, chosen, rejected) triples
    • chosen  = user_edited  (when available) or llm_output
    • rejected = slm_output  (raw SLM, lower quality)
    • Aligns the model toward human / LLM-preferred outputs without a reward model

After training:
  • LoRA adapter is merged into the base weights
  • Model is exported to GGUF (q4_k_m by default)
  • The GGUF path is written to MongoDB via latestPath.set_latest_gguf()
  • Metrics (perplexity, ROUGE-1/2/L, DPO reward margin) are saved to training_col

Usage:
  python postTrainer.py \\
      --model-name unsloth/Qwen2.5-1.5B-Instruct \\
      --output-dir ./models \\
      --epochs-sft 2 --epochs-dpo 1
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
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
log = logging.getLogger("post_trainer")

# ── MongoDB setup (using our app's DB) ───────────────────────────────────────
_mongo = MongoClient(dburl)
_db    = _mongo["slm_app"]
jobs_col     = _db["jobs"]
training_col = _db["training_runs"]


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class PostTrainConfig:
    model_name: str        = "unsloth/Qwen2.5-1.5B-Instruct"
    output_dir: str        = "./models"
    max_seq_length: int    = 2048
    lora_rank: int         = 16
    lora_alpha: int        = 32
    lora_dropout: float    = 0.05
    # SFT
    epochs_sft: int        = 2
    lr_sft: float          = 2e-4
    batch_sft: int         = 2
    grad_accum_sft: int    = 4
    # DPO
    epochs_dpo: int        = 1
    lr_dpo: float          = 5e-5
    beta_dpo: float        = 0.1       # KL penalty for DPO
    batch_dpo: int         = 2
    grad_accum_dpo: int    = 4
    # Export
    gguf_quant: str        = "q4_k_m"
    seed: int              = 42
    skip_dpo: bool         = False     # set True if no (chosen, rejected) pairs exist
    min_pairs: int         = 200


# ── Data loading ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert HR content specialist. Convert the raw job description "
    "provided by the user into a structured, professional, ATS-friendly Markdown format. "
    "Include clear section headers (##), bullet points, and all key sections."
)


def chat_format(raw_jd: str, response: str = "") -> str:
    """ChatML format for Qwen / similar models."""
    s = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{raw_jd.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if response:
        s += response.strip() + "<|im_end|>"
    return s


def load_sft_data() -> List[dict]:
    """Load SFT pairs: prompt + best reference."""
    records = list(jobs_col.find(
        {"prompt": {"$exists": True}, "slm_output": {"$exists": True}}
    ))
    log.info("Loaded %d records for SFT", len(records))

    rows = []
    for r in records:
        prompt     = (r.get("prompt") or "").strip()
        user_edit  = (r.get("user_edited") or "").strip()
        llm_out    = (r.get("llm_output") or "").strip()
        slm_out    = (r.get("slm_output") or "").strip()

        if not prompt or not slm_out:
            continue

        reference = user_edit or llm_out or slm_out
        rows.append({"text": chat_format(prompt, reference)})

    log.info("SFT samples: %d", len(rows))
    return rows


def load_dpo_data() -> List[dict]:
    """
    Load DPO triples (prompt, chosen, rejected).
    chosen  = user_edited (best) or llm_output
    rejected = slm_output
    Only records that have a clearly better output than slm_output are used.
    """
    records = list(jobs_col.find({
        "prompt": {"$exists": True},
        "slm_output": {"$exists": True},
        "$or": [
            {"user_edited": {"$ne": None, "$exists": True}},
            {"llm_output":  {"$ne": None, "$exists": True}},
        ]
    }))
    log.info("Candidate DPO records: %d", len(records))

    rows = []
    for r in records:
        prompt    = (r.get("prompt") or "").strip()
        user_edit = (r.get("user_edited") or "").strip()
        llm_out   = (r.get("llm_output") or "").strip()
        slm_out   = (r.get("slm_output") or "").strip()

        if not prompt or not slm_out:
            continue

        chosen = user_edit or llm_out
        if not chosen:
            continue

        rows.append({
            "prompt":   chat_format(prompt),
            "chosen":   chosen,
            "rejected": slm_out,
        })

    log.info("DPO triples: %d", len(rows))
    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_perplexity(model, tokenizer, texts: List[str], max_samples: int = 30) -> float:
    """Token-level perplexity on a sample of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    sample = texts[:max_samples]
    with torch.no_grad():
        for text in sample:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(model.device)
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            total_loss   += out.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> float:
    """Token-overlap ROUGE-N (no external deps)."""
    def ngrams(text: str, n: int):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    hyp_ng  = ngrams(hypothesis, n)
    ref_ng  = ngrams(reference, n)
    if not hyp_ng or not ref_ng:
        return 0.0
    ref_set = {}
    for g in ref_ng:
        ref_set[g] = ref_set.get(g, 0) + 1
    overlap = sum(min(hyp_ng.count(g), ref_set.get(g, 0)) for g in set(hyp_ng))
    precision = overlap / len(hyp_ng)
    recall    = overlap / len(ref_ng)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(hypothesis: str, reference: str) -> float:
    """LCS-based ROUGE-L."""
    hyp = re.findall(r"\b\w+\b", hypothesis.lower())
    ref = re.findall(r"\b\w+\b", reference.lower())
    m, n = len(hyp), len(ref)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if hyp[i-1] == ref[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_rouge(model, tokenizer, pairs: List[dict], max_samples: int = 20) -> dict:
    """Generate outputs and compute ROUGE against best reference."""
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    import random
    sample = random.sample(pairs, min(max_samples, len(pairs)))
    r1s, r2s, rls = [], [], []

    for p in sample:
        prompt_text = chat_format(p["prompt"])
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        ref = p["reference"]
        r1s.append(rouge_n(generated, ref, 1))
        r2s.append(rouge_n(generated, ref, 2))
        rls.append(rouge_l(generated, ref))

    return {
        "rouge_1": round(sum(r1s) / len(r1s), 4),
        "rouge_2": round(sum(r2s) / len(r2s), 4),
        "rouge_l": round(sum(rls) / len(rls), 4),
    }


# ── GGUF Export (shared helper) ───────────────────────────────────────────────
def export_gguf(model, tokenizer, output_dir: Path, quant: str, tag: str) -> Optional[Path]:
    """
    Merge LoRA + export to GGUF.
    Returns the final GGUF path or None on failure.
    """
    models_dir = output_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    gguf_name = f"jd_{tag}_{quant}.gguf"
    final_path = models_dir / gguf_name

    try:
        log.info("Exporting GGUF (%s) → %s", quant.upper(), final_path)
        model.save_pretrained_gguf(
            str(models_dir / f"jd_{tag}"),
            tokenizer,
            quantization_method=quant,
        )
        # Unsloth auto-names; grab the first .gguf it produced
        candidates = [p for p in models_dir.glob("*.gguf") if p != final_path]
        if candidates:
            shutil.move(str(candidates[0]), str(final_path))
        log.info("GGUF saved → %s", final_path)
        return final_path
    except Exception as exc:
        log.error("GGUF export failed: %s", exc)
        return None


# ── SFT Training ──────────────────────────────────────────────────────────────
def run_sft(cfg: PostTrainConfig, sft_rows: List[dict], run_id: str):
    """Stage 1: Supervised Fine-Tuning."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset
    except ImportError as exc:
        log.error("Missing dep: %s  →  pip install unsloth trl datasets", exc)
        raise

    log.info("═" * 55)
    log.info("  Stage 1 — Supervised Fine-Tuning (SFT)")
    log.info("═" * 55)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    dataset = Dataset.from_list(sft_rows)
    split   = dataset.train_test_split(test_size=min(0.1, 20 / len(sft_rows)), seed=cfg.seed)

    sft_out = Path(cfg.output_dir) / "sft_checkpoints"
    sft_out.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(sft_out),
        num_train_epochs=cfg.epochs_sft,
        per_device_train_batch_size=cfg.batch_sft,
        gradient_accumulation_steps=cfg.grad_accum_sft,
        learning_rate=cfg.lr_sft,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("SFT complete in %.1f min", elapsed / 60)

    # Perplexity on eval split
    eval_texts = [r["text"] for r in split["test"].to_list()]
    ppl = compute_perplexity(model, tokenizer, eval_texts)
    log.info("SFT perplexity: %.2f", ppl)

    # Save SFT adapter
    sft_adapter = Path(cfg.output_dir) / "sft_adapter"
    sft_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(sft_adapter))
    tokenizer.save_pretrained(str(sft_adapter))

    return model, tokenizer, ppl, elapsed


# ── DPO Training ──────────────────────────────────────────────────────────────
def run_dpo(cfg: PostTrainConfig, model, tokenizer, dpo_rows: List[dict]):
    """Stage 2: Direct Preference Optimisation."""
    try:
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError as exc:
        log.error("Missing dep: %s  →  pip install trl datasets", exc)
        raise

    log.info("═" * 55)
    log.info("  Stage 2 — Direct Preference Optimisation (DPO)")
    log.info("═" * 55)

    dataset = Dataset.from_list(dpo_rows)
    split   = dataset.train_test_split(test_size=min(0.1, 10 / len(dpo_rows)), seed=cfg.seed)

    dpo_out = Path(cfg.output_dir) / "dpo_checkpoints"
    dpo_out.mkdir(parents=True, exist_ok=True)

    dpo_cfg = DPOConfig(
        output_dir=str(dpo_out),
        num_train_epochs=cfg.epochs_dpo,
        per_device_train_batch_size=cfg.batch_dpo,
        gradient_accumulation_steps=cfg.grad_accum_dpo,
        learning_rate=cfg.lr_dpo,
        beta=cfg.beta_dpo,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("DPO complete in %.1f min", elapsed / 60)

    # Save DPO adapter
    dpo_adapter = Path(cfg.output_dir) / "dpo_adapter"
    dpo_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dpo_adapter))
    tokenizer.save_pretrained(str(dpo_adapter))

    return model, tokenizer, elapsed


# ── Persist run record ────────────────────────────────────────────────────────
def save_run_record(run_id: str, cfg: PostTrainConfig, metrics: dict, gguf_path: str,
                    sft_elapsed: float, dpo_elapsed: float, n_sft: int, n_dpo: int):
    doc = {
        "run_id":       run_id,
        "pipeline":     "post_train",
        "model_base":   cfg.model_name,
        "gguf_path":    gguf_path,
        "started_at":   metrics.get("started_at"),
        "finished_at":  datetime.now(timezone.utc).isoformat(),
        "duration_min": round((sft_elapsed + dpo_elapsed) / 60, 2),
        "config": {
            "epochs_sft":   cfg.epochs_sft,
            "epochs_dpo":   0 if cfg.skip_dpo else cfg.epochs_dpo,
            "lr_sft":       cfg.lr_sft,
            "lr_dpo":       cfg.lr_dpo,
            "lora_rank":    cfg.lora_rank,
            "batch_sft":    cfg.batch_sft,
            "gguf_quant":   cfg.gguf_quant,
        },
        "data": {
            "sft_samples": n_sft,
            "dpo_triples": n_dpo,
        },
        "metrics": metrics,
        "status": "success",
    }
    training_col.insert_one(doc)
    log.info("Run record saved → training_runs (%s)", run_id)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(cfg: PostTrainConfig):
    run_id = f"pt_{int(time.time())}"
    started_at = datetime.now(timezone.utc).isoformat()
    log.info("Run ID: %s", run_id)

    # ── 1. Load data
    sft_rows = load_sft_data()
    dpo_rows = load_dpo_data()

    if len(sft_rows) < cfg.min_pairs:
        log.warning("Only %d SFT pairs (need %d). Aborting.", len(sft_rows), cfg.min_pairs)
        sys.exit(0)

    # ── 2. SFT
    model, tokenizer, ppl_sft, sft_elapsed = run_sft(cfg, sft_rows, run_id)

    # ── 3. DPO (optional)
    dpo_elapsed = 0.0
    skip_dpo = cfg.skip_dpo or len(dpo_rows) < 10
    if not skip_dpo:
        model, tokenizer, dpo_elapsed = run_dpo(cfg, model, tokenizer, dpo_rows)
    else:
        log.info("Skipping DPO (skip_dpo=%s, dpo_pairs=%d)", cfg.skip_dpo, len(dpo_rows))

    # ── 4. ROUGE evaluation
    # Build reference pairs from SFT data for ROUGE
    from pymongo import MongoClient as _MC
    _c = _MC(dburl)
    raw_records = list(_c["slm_app"]["jobs"].find(
        {"prompt": {"$exists": True}, "slm_output": {"$exists": True}}
    ))
    rouge_pairs = []
    for r in raw_records[:50]:
        prompt = (r.get("prompt") or "").strip()
        ref    = (r.get("user_edited") or r.get("llm_output") or r.get("slm_output") or "").strip()
        if prompt and ref:
            rouge_pairs.append({"prompt": prompt, "reference": ref})

    rouge_scores = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    if rouge_pairs:
        try:
            rouge_scores = evaluate_rouge(model, tokenizer, rouge_pairs)
            log.info("ROUGE scores: %s", rouge_scores)
        except Exception as exc:
            log.warning("ROUGE eval failed: %s", exc)

    # ── 5. Export GGUF
    models_dir = Path(cfg.output_dir)
    gguf_path = export_gguf(model, tokenizer, models_dir, cfg.gguf_quant, "post")
    if gguf_path:
        set_latest_gguf(str(gguf_path))
        log.info("Updated latestPath → %s", gguf_path)

    # ── 6. Save metrics to DB
    metrics = {
        "started_at":   started_at,
        "perplexity":   round(ppl_sft, 4),
        "rouge_1":      rouge_scores["rouge_1"],
        "rouge_2":      rouge_scores["rouge_2"],
        "rouge_l":      rouge_scores["rouge_l"],
    }
    save_run_record(
        run_id, cfg, metrics,
        gguf_path=str(gguf_path) if gguf_path else "export_failed",
        sft_elapsed=sft_elapsed,
        dpo_elapsed=dpo_elapsed,
        n_sft=len(sft_rows),
        n_dpo=len(dpo_rows),
    )

    log.info("Post-training pipeline complete.")
    log.info("GGUF → %s", gguf_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> PostTrainConfig:
    p = argparse.ArgumentParser(description="SFT + DPO post-training pipeline")
    p.add_argument("--model-name",  default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir",  default="./models")
    p.add_argument("--epochs-sft",  type=int,   default=2)
    p.add_argument("--epochs-dpo",  type=int,   default=1)
    p.add_argument("--lr-sft",      type=float, default=2e-4)
    p.add_argument("--lr-dpo",      type=float, default=5e-5)
    p.add_argument("--lora-rank",   type=int,   default=16)
    p.add_argument("--batch-sft",   type=int,   default=2)
    p.add_argument("--batch-dpo",   type=int,   default=2)
    p.add_argument("--gguf-quant",  default="q4_k_m",
                   choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    p.add_argument("--min-pairs",   type=int,   default=200)
    p.add_argument("--skip-dpo",    action="store_true")
    args = p.parse_args()
    return PostTrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs_sft=args.epochs_sft,
        epochs_dpo=args.epochs_dpo,
        lr_sft=args.lr_sft,
        lr_dpo=args.lr_dpo,
        lora_rank=args.lora_rank,
        batch_sft=args.batch_sft,
        batch_dpo=args.batch_dpo,
        gguf_quant=args.gguf_quant,
        min_pairs=args.min_pairs,
        skip_dpo=args.skip_dpo,
    )


if __name__ == "__main__":
    main(parse_args())
