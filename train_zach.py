# train_irish_folk_with_tokenizer.py
import os
import torch
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, random_split

class IrishFolkDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = " ".join([str(token) for token in self.data[idx]])
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # for causal language modeling, labels are the same as inputs
        }

processed_dir = "data/processed"
output_dir = "models/gpt2-v.1.0"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

inputs = np.load(os.path.join(processed_dir, 'inputs.npy')) # @maxjtwelftree Im just copying this from your code since i cant see the data
print(f"Loaded input data with shape: {inputs.shape}")

# Initialize GPT2 tokenizer (just standard english tokenizer (I dont think this is great but could work for v.1.0))
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"Loaded GPT2 tokenizer with vocabulary size: {len(tokenizer)}")

# Create dataset
dataset = IrishFolkDataset(inputs, tokenizer) # @maxjtwelftree This probs needs to be modified to work with the data

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

# Create model
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512, # This is the maximum MIDI sequence we pass in (will depend on how many tokens in a sequnce)
                     # If its really big, we maybe should crop down to 8 bars or something.   
    n_embd=768,
    n_layer=12,
    n_head=12,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

model = GPT2LMHeadModel(config).to(device)
print("Model created and loaded to device")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # False for causal language modeling
)
print("Data collator created for causal language modeling")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10, # Depends on number of training examples
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
)
print("Training arguments configured")

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
print("Trainer initialized")

# Train model
print("Starting training...")
trainer.train()

# Evaluate model
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save final model
trainer.save_model(os.path.join(output_dir, "final_model"))
print(f"Model saved to {os.path.join(output_dir, 'final_model')}")

# Save tokenizer
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
print(f"Tokenizer saved to {os.path.join(output_dir, 'final_model')}")