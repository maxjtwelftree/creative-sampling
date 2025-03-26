# train_irish_folk_with_tokenizer.py with robust error handling
import os
import sys
import torch
import numpy as np
import pickle
import traceback
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, random_split

# Debug helper - print memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    import psutil
    process = psutil.Process(os.getpid())
    print(f"RAM used: {process.memory_info().rss / 1024**2:.2f} MB")

class IrishFolkDataset(Dataset):
    def __init__(self, input_ids, vocab_size):
        self.input_ids = input_ids
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Get sequence of 64 tokens
        item = self.input_ids[idx].astype(np.int64)
        
        # For causal language modeling, we use the sequence as both input and target
        input_ids = torch.tensor(item, dtype=torch.long)
        
        # Create a simple attention mask (all 1s since we don't have padding)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # for causal language modeling
        }

# Paths
processed_dir = "./data/processed"
output_dir = "./models/huggingface-gpt2"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Check if processed data exists
    if not os.path.exists(os.path.join(processed_dir, 'inputs.npy')):
        print("Processed data not found. Running data processing...")
        
        # Process a smaller number of files for testing
        from utils.data_processor import process_dataset
        raw_dir = "./data/raw"
        
        # Use a small limit for initial testing
        limit = 10000  # Start with a small number
        inputs, targets, note_to_int, int_to_note, vocab_size = process_dataset(raw_dir, processed_dir, sequence_length=64, limit=limit)
        print(f"Using limit of {limit} files for processing")
        print("Data processing complete.")
    else:
        # Load data in a try block to catch errors
        try:
            print("Loading preprocessed data...")
            inputs = np.load(os.path.join(processed_dir, 'inputs.npy'))
            
            # Optionally load only a subset to test
            # inputs = inputs[:1000]  # Uncomment to use only a small portion
            
            # Load vocabulary size
            with open(os.path.join(processed_dir, 'vocab_size.txt'), 'r') as f:
                vocab_size = int(f.read().strip())
                
            # Load note mappings for reference
            with open(os.path.join(processed_dir, 'int_to_note.pkl'), 'rb') as f:
                int_to_note = pickle.load(f)
                
            print(f"Loaded input data with shape: {inputs.shape}")
            print(f"Loaded vocabulary with size: {vocab_size}")
            print_memory_usage()
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Create dataset with error handling
    try:
        print("Creating dataset...")
        dataset = IrishFolkDataset(inputs, vocab_size)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)  # For reproducibility
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Create tokenizer with error handling
    try:
        print("Creating tokenizer...")
        # Create a simple tokenizer for our needs
        tokenizer = PreTrainedTokenizerFast(
            model_max_length=64,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            additional_special_tokens=[f"[NOTE_{i}]" for i in range(vocab_size)]
        )
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Created tokenizer with vocabulary size: {len(tokenizer)}")
    except Exception as e:
        print(f"Error creating tokenizer: {e}")
        print("Trying fallback tokenizer...")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # Create model with proper error handling
    try:
        print("Creating model...")
        config = GPT2Config(
            vocab_size=vocab_size + 1,  # Add 1 for padding token
            n_positions=64,  # Your sequence length
            n_embd=256,  # Smaller embedding size
            n_layer=6,   # Fewer layers
            n_head=8,    # Fewer attention heads
            bos_token_id=vocab_size,
            eos_token_id=vocab_size,
            pad_token_id=vocab_size
        )
        model = GPT2LMHeadModel(config).to(device)
        print("Model created and loaded to device")
        print_memory_usage()
    except Exception as e:
        print(f"Error creating model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Create data collator
    try:
        print("Creating data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # False for causal language modeling
        )
    except Exception as e:
        print(f"Error creating data collator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Training arguments - reduce batch size and use gradient accumulation
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # Mixed precision
        report_to="none",
    )
    print("Training arguments configured")

    # Initialize trainer
    try:
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        print("Trainer initialized")
        print_memory_usage()
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Train model
    try:
        print("Starting training...")
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate model
    try:
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

    # Save model
    try:
        print("Saving model...")
        trainer.save_model(os.path.join(output_dir, "final_model"))
        print(f"Model saved to {os.path.join(output_dir, 'final_model')}")
        
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        with open(os.path.join(output_dir, "final_model", "int_to_note.pkl"), 'wb') as f:
            pickle.dump(int_to_note, f)
        print(f"Tokenizer and vocabulary mapping saved")
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"Unhandled exception: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Script completed successfully!")
