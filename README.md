# JD Enhancer — Self-Learning Job Description Enhancement System

A full-stack AI system that takes messy, unstructured job descriptions and turns them into clean, ATS-ready markdown. The model doesn't just do this once — it watches how users edit its outputs and retrains itself periodically to get better over time.

Built for HACK60.

---

## What's happening under the hood

Users paste a raw JD. A locally-running small language model (Qwen2.5-1.5B or Phi) generates a structured output in real-time via SSE streaming. If the user wants a higher-quality pass, they can optionally route it through Gemini for refinement.

Whatever the user saves — raw input, SLM output, LLM-refined version, and any manual edits — gets stored in MongoDB. Once enough feedback pairs accumulate, training kicks off automatically:

- **≥ 50 new pairs** → GRPO reinforcement learning pass (`reinforcement.py`)
- **≥ 200 new pairs** → Full SFT + DPO post-training (`postTrainer.py`)

After each training run, the new model gets exported to GGUF and registered as the `latest` model. Next inference request picks it up.

---

## Project Structure

```
HACK60/
├── backend/
│   ├── serve.py            # FastAPI app — auth, streaming inference, job saving
│   ├── inference.py        # llama-cpp model loading + token streaming
│   ├── config.py           # Prompt templates, DB URL, Gemini config
│   ├── database.py         # MongoDB collections + indexes
│   ├── auth.py             # JWT-based auth (signup/login)
│   ├── gemini.py           # Gemini API integration for LLM enhancement
│   ├── postTrainer.py      # SFT → DPO training pipeline (≥200 pairs)
│   ├── reinforcement.py    # GRPO RL pipeline (≥50 pairs)
│   ├── latestPath.py       # Tracks which GGUF is the active model
│   ├── datasets/
│   │   ├── final_output.json
│   │   ├── local_distilled_jd_dataset.json
│   │   └── ollama_generated.json
│   └── scripts/
│       └── seed_metrics.py
│
├── frontend/
│   └── src/
│       ├── pages/          # Login, Signup, Dashboard, AccuracyPage
│       ├── components/     # ChatInput, MarkdownEditor, VersionsPanel, Sidebar
│       ├── api/client.js   # All API calls
│       └── store/          # Zustand state
│
└── extra_scripts/          # (files_.zip)
    ├── train.py            # Initial LoRA fine-tuning on Qwen2.5-3B
    ├── preprocess.py       # Build input/output pairs from raw folder structure
    ├── gendpo.py           # Generate DPO preference pairs
    ├── train_evaluate_dpo.py
    ├── make_gg.py          # GGUF conversion helper
    └── test.py
```

---

## Models

Three GGUF models are supported, loaded lazily and cached in memory:

| Key         | Model                         | Notes                        |
|-------------|-------------------------------|------------------------------|
| `qwen`      | Qwen2.5-1.5B-Instruct         | Default                      |
| `phi`       | Phi (Microsoft)               | Alternative                  |
| `tinyllama` | TinyLlama fine-tuned          | Lightweight option           |
| `latest`    | Post-training output          | Resolved dynamically at runtime |

All models run via `llama-cpp-python` with GPU offloading (`n_gpu_layers=-1`).

---

## Tech Stack

**Backend**
- Python, FastAPI
- `llama-cpp-python` for local GGUF inference
- `unsloth` + `peft` for LoRA/QLoRA fine-tuning
- TRL for SFT, DPO, and GRPO trainers
- MongoDB (via pymongo) for storing jobs, users, and training run metrics
- Google Gemini API for optional LLM refinement pass
- JWT auth

**Frontend**
- React + Vite
- Zustand for state
- Markdown editor with live preview
- SSE-based streaming (tokens show up as they're generated)
- Accuracy/metrics dashboard page

---

## Setup

### Prerequisites

- Python 3.10+
- Node 18+
- MongoDB running locally on port 27017
- CUDA-capable GPU recommended (CPU works but slower)
- GGUF model files placed in `backend/models/`

### Backend

```bash
cd backend
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Start the server
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Training Pipeline

### Initial Training (one-time setup)

Use the scripts in `extra_scripts/` to prepare data and run the first fine-tune:

```bash
# Step 1: Build dataset from raw JD folder pairs
python preprocess.py

# Step 2: Fine-tune with LoRA on Qwen2.5-3B
python train.py

# Step 3: Generate DPO preference pairs
python gendpo.py

# Step 4: Run DPO alignment
python train_evaluate_dpo.py

# Step 5: Export to GGUF
python make_gg.py
```

### Continuous Learning (automated)

The system checks feedback pair counts on each save and auto-triggers the right trainer:

```bash
# Manually trigger if needed:

# GRPO (≥50 pairs)
python reinforcement.py

# SFT + DPO (≥200 pairs)
python postTrainer.py \
  --model-name unsloth/Qwen2.5-1.5B-Instruct \
  --output-dir ./models \
  --epochs-sft 2 \
  --epochs-dpo 1
```

After training, the new GGUF is registered in MongoDB and the `/llm/stream` endpoint automatically uses it for subsequent requests.

---

## API Endpoints

| Method | Route            | Description                                 |
|--------|------------------|---------------------------------------------|
| POST   | `/signup`        | Register new user                           |
| POST   | `/login`         | Login, returns JWT                          |
| GET    | `/llm/stream`    | SSE stream — SLM or Gemini inference        |
| POST   | `/save-job`      | Save JD session (input + outputs + edits)   |
| GET    | `/history`       | Fetch user's saved JDs                      |
| GET    | `/training-runs` | View all training run metrics               |
| GET    | `/models`        | List available models + current default     |
| GET    | `/`              | Health check                                |

The `/llm/stream` endpoint accepts a `useLLM=true` flag to route through Gemini instead of the local model.

---

## Evaluation

Each training run logs the following metrics to MongoDB (`training_runs` collection):

- Training loss, validation loss
- Perplexity
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU score
- DPO reward margin (post-training runs)
- Structural completeness (checks for required headings)

These are visualized in the frontend's Accuracy page, which shows per-run trends over time.

Required headings the model is evaluated against:
```
## Job Title
## Location
## Client Industry
## Detailed Responsibilities
## Skill Requirements
## Other Requirements
```

---

## Environment Variables

| Variable        | Where        | Description               |
|-----------------|--------------|---------------------------|
| `GEMINI_API_KEY`| `backend/.env` | Google Gemini API key   |
| `SECRET_KEY`    | `backend/.env` | JWT signing secret      |

---

## Notes

- The `latest` model key intentionally bypasses the in-memory model cache, so newly trained models are picked up without restarting the server.
- DPO training uses `user_edited` as `chosen` when available, falling back to `llm_output`. The raw SLM output is always `rejected`.
- The frontend streams tokens as they arrive — there's no polling, it's pure SSE.
- Base model size is strictly under 4B parameters across all supported variants, per the problem constraints.