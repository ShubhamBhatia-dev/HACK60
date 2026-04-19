# JD Enhancer — Self-Learning Job Description Enhancement System

> A self-hosted AI pipeline that converts unstructured job descriptions into professional, ATS-ready formats — and continuously improves itself through user feedback and automated retraining.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![MongoDB](https://img.shields.io/badge/MongoDB-Local-47A248?style=flat&logo=mongodb&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Training Pipeline](#training-pipeline)
- [API Reference](#api-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Environment Variables](#environment-variables)

---

## Overview

HR teams frequently deal with raw, inconsistent job descriptions that require significant manual effort before they can be posted on hiring platforms or parsed by Applicant Tracking Systems. This project addresses that problem with a locally-hosted AI system that:

- Accepts a raw job description as input
- Generates a structured, professional markdown output in real-time via a fine-tuned Small Language Model (SLM)
- Optionally routes the output through Google Gemini for a higher-quality refinement pass
- Stores every interaction (raw input, SLM output, LLM-refined version, user edits) in MongoDB
- Automatically triggers retraining when enough feedback pairs have accumulated

The model improves over time without manual intervention. Each retraining cycle produces a new GGUF artifact that is registered as the active inference model — no server restart required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│   Auth  ·  Markdown Editor  ·  Version History  ·  Metrics  │
└──────────────────────┬───────────────────────────────────────┘
                       │  SSE / REST
┌──────────────────────▼───────────────────────────────────────┐
│                   FastAPI Backend                            │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │  inference  │   │   gemini.py  │   │    auth / JWT    │  │
│  │ (llama-cpp) │   │  (optional)  │   │                  │  │
│  └──────┬──────┘   └──────────────┘   └──────────────────┘  │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────────┐ │
│  │                      MongoDB                            │ │
│  │  users · jobs · training_runs · training_meta           │ │
│  └──────────────────────────┬───────────────────────────── ┘ │
└─────────────────────────────│────────────────────────────────┘
                              │  Feedback threshold check
              ┌───────────────▼──────────────────┐
              │        Training Trigger           │
              │                                   │
              │  ≥  50 pairs → GRPO (RL)          │
              │  ≥ 200 pairs → SFT + DPO          │
              └───────────────┬──────────────────-┘
                              │
                    ┌─────────▼──────────┐
                    │  New GGUF export   │
                    │  registered as     │
                    │  `latest` model    │
                    └────────────────────┘
```

---

## Project Structure

```
HACK60/
├── backend/
│   ├── serve.py                  # FastAPI application entry point
│   ├── inference.py              # llama-cpp model loading and token streaming
│   ├── config.py                 # Prompt templates, database URL, Gemini config
│   ├── database.py               # MongoDB collections and index definitions
│   ├── auth.py                   # JWT authentication (signup / login)
│   ├── gemini.py                 # Google Gemini API integration
│   ├── postTrainer.py            # SFT → DPO automated training pipeline
│   ├── reinforcement.py          # GRPO reinforcement learning pipeline
│   ├── latestPath.py             # Active model path registry (MongoDB-backed)
│   ├── models/                   # GGUF model files (not tracked in Git)
│   ├── datasets/
│   │   ├── final_output.json
│   │   ├── local_distilled_jd_dataset.json
│   │   └── ollama_generated.json
│   └── scripts/
│       └── seed_metrics.py
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Login.jsx
│       │   ├── Signup.jsx
│       │   ├── Dashboard.jsx
│       │   └── AccuracyPage.jsx
│       ├── components/
│       │   ├── ChatInput.jsx
│       │   ├── MarkdownEditor.jsx
│       │   ├── VersionsPanel.jsx
│       │   └── Sidebar.jsx
│       ├── api/client.js
│       └── store/useAppStore.js
│
└── scripts/                      # Offline data preparation and initial training
    ├── preprocess.py             # Build input/output pairs from folder structure
    ├── train.py                  # Initial LoRA fine-tuning (Qwen2.5-3B)
    ├── gendpo.py                 # Generate DPO preference pairs
    ├── train_evaluate_dpo.py     # DPO alignment training + evaluation
    ├── make_gg.py                # GGUF export helper
    └── test.py
```

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| API Server | FastAPI |
| Local Inference | `llama-cpp-python` (GGUF, GPU offloaded) |
| Fine-tuning | `unsloth`, `peft`, `trl` (SFT / DPO / GRPO) |
| LLM Refinement | Google Gemini API |
| Database | MongoDB (pymongo) |
| Authentication | JWT (python-jose) |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 18 + Vite |
| State Management | Zustand |
| Streaming | Server-Sent Events (SSE) |
| Editor | Markdown editor with live preview |

### Supported Inference Models

| Key | Base Model | Notes |
|---|---|---|
| `qwen` *(default)* | Qwen2.5-1.5B-Instruct | Primary model |
| `phi` | Microsoft Phi | Alternative |
| `tinyllama` | TinyLlama (fine-tuned) | Lightweight option |
| `latest` | Post-training artifact | Resolved dynamically at runtime |

All models run via `llama-cpp-python` with full GPU offloading. The `latest` key bypasses the in-memory model cache so newly trained checkpoints are available immediately without restarting the server.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB instance running on `localhost:27017`
- CUDA-capable GPU (recommended; CPU inference is supported but significantly slower)
- GGUF model files placed under `backend/models/`

### Backend

```bash
cd backend
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and SECRET_KEY

# Start the API server
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend development server starts at `http://localhost:5173` and proxies API requests to the backend.

---

## Training Pipeline

### Stage 0 — Data Preparation (one-time)

Raw job descriptions and their enhanced counterparts must be structured as folder pairs before training:

```
dataset/
└── 001/
    ├── raw_jd.txt
    └── enhanced_job_description.md
```

Run the preprocessing script to consolidate them into a single training file:

```bash
python scripts/preprocess.py
```

### Stage 1 — Initial Fine-Tuning

Fine-tune the base model using LoRA on the prepared dataset:

```bash
python scripts/train.py
```

This runs supervised fine-tuning on Qwen2.5-3B-Instruct with 4-bit quantization (QLoRA) via PEFT. Evaluation metrics (ROUGE, BLEU, perplexity) are logged after each epoch.

### Stage 2 — DPO Alignment

Generate preference pairs and run Direct Preference Optimisation:

```bash
# Generate (prompt, chosen, rejected) triples
python scripts/gendpo.py

# Run DPO training
python scripts/train_evaluate_dpo.py
```

The `chosen` response is the LLM-refined or user-edited version; `rejected` is the raw SLM output.

### Stage 3 — Export to GGUF

```bash
python scripts/make_gg.py
```

### Continuous Learning (Automated)

Once the system is live, retraining is triggered automatically based on accumulated feedback:

| Threshold | Training Type | Script |
|---|---|---|
| ≥ 50 new feedback pairs | GRPO Reinforcement Learning | `reinforcement.py` |
| ≥ 200 new feedback pairs | SFT + DPO Post-Training | `postTrainer.py` |

To trigger manually:

```bash
# GRPO pass
python backend/reinforcement.py

# Full SFT + DPO pass
python backend/postTrainer.py \
  --model-name unsloth/Qwen2.5-1.5B-Instruct \
  --output-dir ./models \
  --epochs-sft 2 \
  --epochs-dpo 1
```

After each run, the new GGUF is registered in MongoDB. All subsequent requests to `/llm/stream?model=latest` will use the updated checkpoint.

---

## API Reference

All protected routes require an `Authorization: Bearer <token>` header.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/signup` | No | Register a new user |
| `POST` | `/login` | No | Authenticate and receive JWT |
| `GET` | `/llm/stream` | Yes | SSE stream — SLM or Gemini inference |
| `POST` | `/save-job` | Yes | Persist a JD session with all outputs |
| `GET` | `/history` | Yes | Retrieve the authenticated user's saved JDs |
| `GET` | `/training-runs` | No | List all training runs with metrics |
| `GET` | `/models` | No | List available models and current default |
| `GET` | `/` | No | Health check |

#### `/llm/stream` Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `string` | required | Raw job description input |
| `useLLM` | `boolean` | `false` | Route through Gemini instead of local SLM |
| `slm_response` | `string` | `""` | SLM output to refine (used when `useLLM=true`) |
| `model` | `string` | `qwen` | Model key to use for local inference |

---

## Evaluation Metrics

Each training run records the following metrics to the `training_runs` MongoDB collection, visualised in the frontend Accuracy page:

| Metric | Description |
|---|---|
| Training Loss | Cross-entropy loss on training split |
| Validation Loss | Cross-entropy loss on held-out split |
| Perplexity | Derived from validation loss |
| ROUGE-1 / ROUGE-2 / ROUGE-L | N-gram overlap with reference outputs |
| BLEU | Precision-based translation metric |
| DPO Reward Margin | Log-probability difference between chosen and rejected (post-training only) |
| Structural Completeness | Percentage of required headings present in generated output |

Required output headings validated during evaluation:

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

Create a `.env` file inside the `backend/` directory:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for LLM refinement |
| `SECRET_KEY` | Secret used for signing JWT tokens |

---

