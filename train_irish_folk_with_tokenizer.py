# train_irish_folk_with_tokenizer.py - Optimized for best Irish folk music output
import os
import sys
import torch
import numpy as np
import pickle
import traceback
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, random_split

# Debug helper - print memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    import psutil
    process = psutil.Process(os.getpid())
    print(f"RAM used: {process.memory_info().rss / 1024**2:.2f} MB")

class IrishFolkDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = self.input_ids[idx].astype(np.int64)
        input_ids = torch.tensor(item, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

# Paths
processed_dir = "./data/processed"
output_dir = "./models/third/huggingface-gpt2"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    if not os.path.exists(os.path.join(processed_dir, "inputs.npy")):
        print("Processed data not found. Running data processing...")
        from utils.data_processor import process_dataset
        raw_dir = "./data/raw"
        limit = 10000
        inputs, targets, note_to_int, int_to_note, vocab_size = process_dataset(
            raw_dir, processed_dir, sequence_length=128, limit=limit
        )
        print("Data processing complete.")
    else:
        try:
            print("Loading preprocessed data...")
            inputs = np.load(os.path.join(processed_dir, "inputs.npy"))
            with open(os.path.join(processed_dir, "vocab_size.txt"), "r") as f:
                vocab_size = int(f.read().strip())
            with open(os.path.join(processed_dir, "int_to_note.pkl"), "rb") as f:
                int_to_note = pickle.load(f)
            print(f"Loaded input data with shape: {inputs.shape}")
            print(f"Loaded vocabulary size: {vocab_size}")
            print_memory_usage()
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            traceback.print_exc()
            sys.exit(1)

    try:
        print("Creating dataset...")
        dataset = IrishFolkDataset(inputs)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"Dataset split: {train_size} training, {val_size} validation")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("Creating tokenizer...")
        tokenizer = PreTrainedTokenizerFast(
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            additional_special_tokens=[f"[NOTE_{i}]" for i in range(vocab_size)] + ["[SEP]", "[REST]", "[TIME_SHIFT]"]
        )
        print(f"Created tokenizer with vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"Error creating tokenizer: {e}")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    try:
        print("Creating model...")
        config = GPT2Config(
            vocab_size=vocab_size + 10,
            n_positions=128,
            n_embd=512,
            n_layer=8,
            n_head=8,
            bos_token_id=vocab_size,
            eos_token_id=vocab_size + 1,
            pad_token_id=vocab_size + 2,
        )
        model = GPT2LMHeadModel(config).to(device)
        print("Model created and loaded to device")
        print_memory_usage()
    except Exception as e:
        print(f"Error creating model: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("Creating data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    except Exception as e:
        print(f"Error creating data collator: {e}")
        traceback.print_exc()
        sys.exit(1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        resume_from_checkpoint=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        gradient_checkpointing=True,
    )

    try:
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
        print("Trainer initialized")
        print_memory_usage()
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("Starting training...")
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

    try:
        print("Saving model...")
        trainer.save_model(os.path.join(output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        with open(os.path.join(output_dir, "final_model", "int_to_note.pkl"), "wb") as f:
            pickle.dump(int_to_note, f)
        print("Model and tokenizer saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"Unhandled exception: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Script completed successfully!")

