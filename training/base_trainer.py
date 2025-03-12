# ~/irish_folk_model/training/base_trainer.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

def load_dataset(processed_dir):
    """Load the processed dataset."""
    try:
        inputs = np.load(os.path.join(processed_dir, 'inputs.npy'))
        targets = np.load(os.path.join(processed_dir, 'targets.npy'))
        
        with open(os.path.join(processed_dir, 'vocab_size.txt'), 'r') as f:
            vocab_size = int(f.read().strip())
        
        # Create dataset using TensorDataset instead of MidiDataset
        dataset = TensorDataset(
            torch.tensor(inputs, dtype=torch.long),  # Use long (integer) type
            torch.tensor(targets, dtype=torch.long)
        )

        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        return train_dataset, val_dataset, vocab_size
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def create_model(vocab_size, sequence_length=64):
    """Create a GPT-2 model for music generation."""
    config = GPT2Config(
        vocab_size=vocab_size + 1,  # Add 1 for padding token
        n_positions=sequence_length,
        n_ctx=sequence_length,
        n_embd=256,
        n_layer=6,
        n_head=8,
        bos_token_id=vocab_size,
        eos_token_id=vocab_size
    )
    
    model = GPT2LMHeadModel(config)
    return model

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """Train the music generation model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.long()
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=inputs)
            
            # Fix: Use only the last token prediction for loss calculation
            last_logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            loss = criterion(last_logits, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.long()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(input_ids=inputs)
                
                # Fix: Use only the last token prediction
                last_logits = outputs.logits[:, -1, :]
                loss = criterion(last_logits, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model if it has the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join("./models/base_model", 'best_model.pt'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, os.path.join("./models/base_model", f'checkpoint_epoch_{epoch+1}.pt'))
