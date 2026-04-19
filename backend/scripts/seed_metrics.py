"""
seed_metrics.py — Insert realistic demo training-run records into MongoDB.
Run once:  python seed_metrics.py
"""
from pymongo import MongoClient
from config import dburl

client = MongoClient(dburl)
col = client["slm_app"]["training_runs"]

# Don't re-seed if already populated
if col.count_documents({}) >= 3:
    print("Already seeded, skipping.")
    exit(0)

DEMO_RUNS = [
    {
        "run_id":       "rl_1712980800",
        "pipeline":     "grpo_rl",
        "model_base":   "unsloth/Qwen2.5-1.5B-Instruct",
        "gguf_path":    "./models/jd_rl_q4_k_m.gguf",
        "started_at":   "2025-04-13T04:00:00Z",
        "finished_at":  "2025-04-13T05:47:21Z",
        "duration_min": 107.35,
        "config": {
            "epochs": 3, "num_generations": 4, "lr": 5e-5,
            "lora_rank": 16, "gguf_quant": "q4_k_m",
        },
        "data": {"rl_pairs": 62},
        "metrics": {
            "perplexity": 18.42,
            "avg_reward": 0.5813,
            "min_reward": 0.3210,
            "max_reward": 0.7940,
            "rouge_1": 0.4721,
            "rouge_2": 0.2103,
            "rouge_l": 0.3894,
        },
        "status": "success",
    },
    {
        "run_id":       "rl_1714276800",
        "pipeline":     "grpo_rl",
        "model_base":   "unsloth/Qwen2.5-1.5B-Instruct",
        "gguf_path":    "./models/jd_rl_q4_k_m.gguf",
        "started_at":   "2025-04-28T09:15:00Z",
        "finished_at":  "2025-04-28T11:09:44Z",
        "duration_min": 114.73,
        "config": {
            "epochs": 3, "num_generations": 4, "lr": 5e-5,
            "lora_rank": 16, "gguf_quant": "q4_k_m",
        },
        "data": {"rl_pairs": 91},
        "metrics": {
            "perplexity": 15.87,
            "avg_reward": 0.6124,
            "min_reward": 0.3780,
            "max_reward": 0.8210,
            "rouge_1": 0.5039,
            "rouge_2": 0.2387,
            "rouge_l": 0.4102,
        },
        "status": "success",
    },
    {
        "run_id":       "pt_1716350400",
        "pipeline":     "post_train",
        "model_base":   "unsloth/Qwen2.5-1.5B-Instruct",
        "gguf_path":    "./models/jd_post_q4_k_m.gguf",
        "started_at":   "2025-05-22T06:00:00Z",
        "finished_at":  "2025-05-22T09:14:33Z",
        "duration_min": 194.55,
        "config": {
            "epochs_sft": 2, "epochs_dpo": 1,
            "lr_sft": 2e-4, "lr_dpo": 5e-5,
            "lora_rank": 16, "batch_sft": 2, "gguf_quant": "q4_k_m",
        },
        "data": {"sft_samples": 214, "dpo_triples": 173},
        "metrics": {
            "perplexity": 11.34,
            "rouge_1": 0.5612,
            "rouge_2": 0.2891,
            "rouge_l": 0.4673,
        },
        "status": "success",
    },
    {
        "run_id":       "rl_1719100800",
        "pipeline":     "grpo_rl",
        "model_base":   "unsloth/Qwen2.5-1.5B-Instruct",
        "gguf_path":    "./models/jd_rl_q4_k_m.gguf",
        "started_at":   "2025-06-23T02:30:00Z",
        "finished_at":  "2025-06-23T04:22:11Z",
        "duration_min": 112.18,
        "config": {
            "epochs": 3, "num_generations": 4, "lr": 5e-5,
            "lora_rank": 16, "gguf_quant": "q4_k_m",
        },
        "data": {"rl_pairs": 58},
        "metrics": {
            "perplexity": 13.09,
            "avg_reward": 0.6471,
            "min_reward": 0.4120,
            "max_reward": 0.8530,
            "rouge_1": 0.5284,
            "rouge_2": 0.2614,
            "rouge_l": 0.4431,
        },
        "status": "success",
    },
    {
        "run_id":       "pt_1722556800",
        "pipeline":     "post_train",
        "model_base":   "unsloth/Qwen2.5-1.5B-Instruct",
        "gguf_path":    "./models/jd_post_q4_k_m.gguf",
        "started_at":   "2025-08-02T08:00:00Z",
        "finished_at":  "2025-08-02T11:31:07Z",
        "duration_min": 211.12,
        "config": {
            "epochs_sft": 2, "epochs_dpo": 1,
            "lr_sft": 2e-4, "lr_dpo": 5e-5,
            "lora_rank": 16, "batch_sft": 2, "gguf_quant": "q4_k_m",
        },
        "data": {"sft_samples": 307, "dpo_triples": 254},
        "metrics": {
            "perplexity": 9.71,
            "rouge_1": 0.5934,
            "rouge_2": 0.3217,
            "rouge_l": 0.5018,
        },
        "status": "success",
    },
]

col.insert_many(DEMO_RUNS)
print(f"Seeded {len(DEMO_RUNS)} demo training runs.")
