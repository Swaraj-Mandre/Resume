# RecruitSheriff — Fine-Tuned LLM for Resume Analysis

A fine-tuned language model that scores how well a resume matches a job description. Given a resume (PDF upload or text) and job description as input, the model outputs a match score (0–100), key strengths, skill gaps, and tailored interview questions.

Built entirely from scratch — individually owned. Dataset generation, fine-tuning, preference optimization, evaluation, and deployment are all done by one person on consumer hardware.

---

## What Makes This Different

Most resume tools are prompt-engineering wrappers around GPT or Gemini. They send your resume to a generic cloud model and return whatever it says. This project takes a fundamentally different approach:

- A open-source model is fine-tuned on domain-specific resume scoring data
- The scoring behavior is baked into the model's weights permanently via SFT + DPO
- No cloud API calls at inference time — the model runs locally

---

## Architecture

```
User uploads PDF resume + pastes Job Description
                    ↓
        pdfplumber extracts resume text
                    ↓
    Fine-tuned LLaMA 3.2 3B scores the resume
                    ↓
  Match Score + Strengths + Gaps + Interview Questions
                    ↓
        FastAPI serves it, HTML frontend shows it
```

---

## Tech Stack

| Layer | Tool | Version | Purpose |
|---|---|---|---|
| Base model | LLaMA 3.2 3B Instruct (Meta) | 3B params | Foundation model for fine-tuning |
| Fine-tuning framework | Unsloth | 2026.4.8 | Memory-efficient QLoRA training |
| Fine-tuning method | QLoRA (SFT + DPO) | — | Parameter-efficient fine-tuning |
| Training library | TRL (HuggingFace) | — | SFTTrainer + DPOTrainer |
| Dataset generation | Ollama (local) | 0.18.3 | Runs LLaMA 3.2 3B locally for data generation |
| Deep learning | PyTorch | 2.10.0+cu128 | Training backend |
| CUDA | CUDA Toolkit | 12.8 | GPU acceleration |
| Attention | Xformers | 0.0.35 | Attention optimization (Flash Attention fallback) |
| PDF parsing | pdfplumber | — | Extract text from uploaded PDF resumes |
| API backend | FastAPI | — | Serve fine-tuned model as REST API |
| Model hosting | Hugging Face Hub | — | Public model deployment |
| Language | Python | 3.11 | All scripts — data, training, evaluation, API |

---

## Training Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB |
| CUDA Compute Capability | 8.9 |
| Training precision | fp16 |
| Optimizer | PagedAdamW 8-bit |
| Platform | Windows 11 |

---

## Project Structure

```
RecruitSheriff/
│
├── data/
│   ├── generate_dataset.py     # Ollama local API → 600 JSONL training pairs
│   ├── dataset.jsonl           # 600 SFT training examples
│   └── dpo_dataset.jsonl       # DPO preference pairs (chosen vs rejected)
│
├── training/
│   ├── finetune.py             # SFT — QLoRA fine-tuning on LLaMA 3.2 3B
│   ├── dpo_train.py            # DPO — preference optimization on SFT adapter
│   └── output/                 # saved LoRA adapter weights
│
├── evaluation/
│   └── evaluate.py             # compare base vs SFT vs DPO model outputs
│
├── app/
│   ├── main.py                 # FastAPI backend — PDF upload + model inference
│   ├── static/
│   │   └── index.html          # frontend — PDF upload + results display
│   └── requirements.txt
│
└── README.md
```

---

## Phase 1 — Dataset Generation

600 training examples generated locally using Ollama running LLaMA 3.2 3B Instruct. No external API, no cost, no rate limits.

Each example follows the Alpaca instruction format:

```json
{
  "instruction": "Analyze this resume against the job description.",
  "input": "Resume: [text]\n\nJob Description: [text]",
  "output": "Match Score: 74/100. Strengths: 1. ... 2. ... Gaps: 1. ... 2. ... Top Interview Questions: 1. ... 2. ... 3. ..."
}
```

Synthetic data generation is standard practice in domain-specific fine-tuning when labeled real-world data is unavailable. The same LLaMA 3.2 3B model used for data generation is later fine-tuned — keeping the pipeline single-model, minimal storage (~2GB total for model files).

---

## Phase 2A — Supervised Fine-Tuning (SFT)

### What SFT does
Teaches the model the task format and domain behavior by training on (input → output) pairs. The model learns to produce structured resume scoring outputs consistently.

### QLoRA — How It Fits on 8GB VRAM

Full fine-tuning of a 3B model requires ~24GB VRAM. QLoRA solves this with two stacked techniques:

**Quantization:** Base model weights compressed from 16-bit floats to 4-bit NF4 integers. Memory drops from ~6GB to ~2GB with negligible quality loss for downstream tasks.

**LoRA:** Thin trainable adapter matrices (A and B) injected into frozen transformer layers. Only these adapters are updated during training. The base model never changes.

Result: fine-tuning a 3B model on a consumer 8GB GPU in under 4 minutes.

### LoRA Configuration

| Hyperparameter | Value | Reason |
|---|---|---|
| Rank (r) | 16 | Standard for instruction fine-tuning — balances capacity and memory |
| Alpha | 32 | 2× rank — standard scaling factor |
| Dropout | 0.05 | Light regularization to reduce overfitting |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention + MLP projection layers |
| Bias | none | Standard for QLoRA |

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Per device batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LR scheduler | Linear with warmup |
| Warmup steps | 10 |
| Optimizer | PagedAdamW 8-bit |
| Precision | fp16 |
| Max sequence length | 2048 tokens |

### Actual Training Metrics (Real — No Estimates)

| Metric | Value |
|---|---|
| Base model | LLaMA 3.2 3B Instruct |
| Total parameters | 3,237,063,680 |
| Trainable parameters (LoRA) | 24,313,856 |
| Trainable % | 0.75% |
| Training examples | 300 (initial run) |
| Total steps | 114 |
| Training time | 3 min 16 sec |
| Train samples/sec | 4.59 |
| Train steps/sec | 0.581 |
| Final training loss | 0.2937 |
| Overall train loss | 0.6228 |

### Loss Curve (Actual Values)

| Step | Epoch | Loss | Grad Norm |
|---|---|---|---|
| 10 | 0.27 | 2.625 | 1.506 |
| 20 | 0.53 | 0.915 | 0.902 |
| 30 | 0.80 | 0.599 | 0.954 |
| 40 | 1.05 | 0.475 | 0.922 |
| 50 | 1.32 | 0.373 | 0.881 |
| 60 | 1.59 | 0.387 | 0.613 |
| 70 | 1.85 | 0.365 | 0.611 |
| 80 | 2.11 | 0.347 | 0.536 |
| 90 | 2.37 | 0.289 | 0.524 |
| 100 | 2.64 | 0.303 | 0.644 |
| 114 | 3.00 | 0.293 | 0.563 |

Loss dropped from 2.625 → 0.293 across 3 epochs. Consistent decrease with no spikes — stable training, no overfitting detected.

---

## Phase 2B — Direct Preference Optimization (DPO)

### What DPO does
After SFT teaches the model the task, DPO teaches it *preference* — given the same resume and JD, which of two outputs is better. This aligns the model toward higher quality, more helpful responses without needing a separate reward model.

DPO is the same technique used in production LLM alignment (used in models like LLaMA 2 Chat, Mistral Instruct). Standard pipeline: SFT first → DPO on top.

### DPO Dataset Format
```json
{
  "prompt": "Analyze this resume against the job description.\n\nResume: [...]\n\nJob Description: [...]",
  "chosen": "Match Score: 78/100. Strengths: specific, detailed reasoning...",
  "rejected": "Match Score: 78/100. Looks good overall."
}
```

Each pair has the same prompt — one detailed, structured output (chosen) and one vague, low-quality output (rejected). The model learns to prefer the chosen style.

---

## Phase 3 — Evaluation

Unseen resume-JD pairs tested against three model states:
- Base LLaMA 3.2 3B (no fine-tuning)
- SFT fine-tuned adapter
- DPO fine-tuned adapter

Evaluation criteria:
- Output format correctness (does it follow Match Score / Strengths / Gaps / Questions structure)
- Score consistency (similar resumes → similar scores)
- Reasoning quality (are strengths and gaps grounded in the actual resume text)
- Hallucination detection (does it invent skills not present in the resume)

---

## Phase 4 — Deployment

### FastAPI Backend
Exposes one endpoint: `POST /analyze`
- Accepts PDF file upload + JD text
- Extracts resume text via pdfplumber
- Runs fine-tuned model inference
- Returns structured JSON response

### Frontend
Simple HTML page — upload PDF resume, paste JD, click analyze, see results.

### Model Hosting
LoRA adapter pushed to Hugging Face Hub. Anyone can download and use the fine-tuned model.

---

## Key Concepts Learned

**QLoRA:** Quantization + LoRA stacked together. Makes fine-tuning billion-parameter models possible on consumer GPUs by reducing memory from 24GB → 4GB for a 3B model.

**LoRA rank and alpha:** Rank controls adapter capacity. Alpha controls scaling strength. These two hyperparameters directly control the trade-off between learning capacity and memory usage.

**Loss curve interpretation:** Loss measures prediction error. Decreasing loss = model learning. Flattening loss = convergence. Spiking loss = unstable training. Reading loss curves is how you know whether to stop, continue, or fix your data.

**Gradient accumulation:** Simulates larger batch sizes on memory-constrained hardware by accumulating gradients over multiple forward passes before updating weights. Batch size 2 × accumulation 4 = effective batch size 8.

**SFT vs DPO:** SFT teaches what to do (task format and domain). DPO teaches what's better (output quality and preference). Production LLMs use both in sequence.

**Tokenization:** Text is never fed to the model directly. It's converted to integer token IDs using the model's vocabulary. Sequence length limits (2048 tokens here) determine how much text fits in one training example.

**Synthetic data generation:** When labeled domain data doesn't exist, use a capable model to generate it. Quality of generated data directly determines quality of the fine-tuned model.

**Local model serving with Ollama:** Running LLMs locally via a REST API at `localhost:11434`. No cloud, no cost, no rate limits. Used for dataset generation in Phase 1.

**Unsloth:** Custom CUDA kernels that make QLoRA 2× faster and use 60% less VRAM than HuggingFace's native implementation. Same training concepts — just optimized execution.

---

## Storage Breakdown

| Item | Location | Size |
|---|---|---|
| LLaMA 3.2 3B (HuggingFace cache) | D drive | ~2.4 GB |
| Ollama + llama3.2:3b | D drive | ~2.0 GB |
| Python venv + dependencies | D drive | ~8.0 GB |
| Training output / LoRA adapters | D drive | ~0.4 GB |
| Dataset files (JSONL) | D drive | ~10 MB |
| Total | D drive | ~12.8 GB |

All model files redirected to D drive via environment variables. C drive impact from this project: pip cache only (~2GB, clearable with `pip cache purge`).

---

## Environment Setup (Windows)

```powershell
# Redirect all model storage to D drive — run once in PowerShell
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\Project\HuggingFace", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\Project\Ollama\models", "User")
```

```bash
# Install dependencies
pip install unsloth trl datasets transformers accelerate bitsandbytes torch
pip install pdfplumber fastapi uvicorn requests
```

---

## How to Run

```bash
# 1. Generate dataset
ollama serve
python data/generate_dataset.py

# 2. SFT fine-tuning
python training/finetune.py

# 3. DPO training
python training/dpo_train.py

# 4. Evaluate
python evaluation/evaluate.py

# 5. Run app
uvicorn app.main:app --reload
# Open browser: http://localhost:8000
```
