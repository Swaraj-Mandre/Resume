import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer # Transformer Reinforcement Learning | Supervised Fine-Tuning Trainer
# (where we give the model an input and the "correct" answer to learn from).
from transformers import TrainingArguments

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
# Base (Pred-Trained) Models  : Model trained on massive data on internet, but doesn't know hoe to respond. 
# Instruct (Fine-Tuned) Models : Take base model through extra stage called SFT and RLHF (Reinforcement Learning from Human Feedback)
MAX_SEQ_LENGTH = 2048 # (roughtly 1,500 words)
DATASET_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "training/output"

# Load model + tokenizer in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # unsloth-detects precision on your GPU (bf16 on RTX 40-series)
    load_in_4bit=True,
)

# Attach LoRA adapters 
model = FastLanguageModel.get_peft_model( # Parameter-Efficient Fine-Tuning
    model,
    r=16, #Works like sticky note attached to every whole page. Create small, thin matrices of size 16
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"], #MLP layers, which we going to target
    lora_alpha=32, # Controls how "strong" the new knowledge is.....usually r * 2 
    lora_dropout=0.05, # "turns-off" 5% of neurons in LoRA . Helps to avoid Overfitting , rather than let him memorize 300 exampples. we drop connections, so it is forced to find out alternative paths for right answer
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42, #Reproducibility 
)

# Format each example into chat template
def format_example(example):
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert HR recruiter and ATS system.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{example['instruction']}

{example['input']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""
    return {"text": text}

# Load and format dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_example)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    # Model looks 2 resume , rather updatng immediately it remebers what it learned in 2, repeats 4 time and then updates using combined knowledge of all 8 (2*4)
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4, # Step model takes while correcting itself
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    # 8bit uses less memory than standard 32-but optimizers. If GPU run out of some memory, it spills some data on systrem ram (Drive D) instead of throuhing error "OutofMemory"
    seed=42,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=args,
)

# Train
print("Starting training...")
trainer.train()

# Save adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")