import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Train Irish folk music generation model")
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Directory with raw MIDI files")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Directory to save processed data")
    parser.add_argument("--model_dir", type=str, default="./models/base_model", help="Directory to save models")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (for testing)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Process data
    print("Processing MIDI files...")
    sys.path.append(os.path.abspath('./utils'))
    from utils.data_processor import process_dataset
    process_dataset(args.data_dir, args.processed_dir, args.seq_length, args.limit)
    
    # Train model
    print("Training base model...")
    from training.base_trainer import train_model, create_model, load_dataset
    
    # Load dataset
    train_dataset, val_dataset, vocab_size = load_dataset(args.processed_dir)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Set device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(vocab_size, args.seq_length)
    model.to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, epochs=args.epochs)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
