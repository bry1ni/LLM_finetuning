import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from peft import LoraConfig

# Load base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048, load_in_4bit=True,
)

# Define LoRA Configuration
lora_config = LoraConfig(
    r=16,                      # Low-rank adaptation
    lora_alpha=32,             # Scaling factor for LoRA layers
    lora_dropout=0.05,         # Dropout for LoRA layers (helps regularization)
    target_modules=[           # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",               # No additional biases are trained
    task_type="CAUSAL_LM",     # Task type (Causal Language Modeling)
)

# Apply LoRA to the model using the structured configuration
model = FastLanguageModel.get_peft_model(model, lora_config)

# Set up chat template and prepare dataset
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(
    lambda examples: {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["conversations"]
        ]
    },
    batched=True
)

# Set up trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("finetuned_model")
