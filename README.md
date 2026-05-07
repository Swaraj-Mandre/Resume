# Resume ATS Fine-Tuning (Unsloth + Ollama)

This repository contains the **current** pipeline for building a resume-vs-job-description ATS scoring model:

1. Generate training data locally with Ollama
2. Fine-tune Llama 3.2 3B with Unsloth (QLoRA)
3. Run evaluation on unseen resume/JD pairs

## Current project structure

```text
.
├── data/
│   ├── generate_dataset.py
│   └── dataset.jsonl
├── training/
│   └── finetune.py
├── evaluation/
│   └── evaluate.py
├── requirements-training.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-training.txt
```

## Run

```bash
# 1) Generate dataset
python data\generate_dataset.py

# 2) Fine-tune model
python training\finetune.py

# 3) Evaluate model
python evaluation\evaluate.py
```

## Notes

- Dataset generation uses local Ollama API (`http://localhost:11434`).
- Fine-tuned outputs are written under `training/output/` (ignored in git).
- This repo does not include old Groq/HTML/CSS app files.
