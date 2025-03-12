# ~/irish_folk_model/analyze_probabilities.py
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from training.base_trainer import create_model

def load_model_and_mappings(model_path, processed_dir):
    # Load mappings and vocab size
    with open(os.path.join(processed_dir, 'int_to_note.pkl'), 'rb') as f:
        int_to_note = pickle.load(f)
    
    with open(os.path.join(processed_dir, 'note_to_int.pkl'), 'rb') as f:
        note_to_int = pickle.load(f)
    
    with open(os.path.join(processed_dir, 'vocab_size.txt'), 'r') as f:
        vocab_size = int(f.read().strip())
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(vocab_size, sequence_length=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model, note_to_int, int_to_note, vocab_size, device

def get_top_predictions(model, sequence, note_to_int, int_to_note, vocab_size, device, top_k=10):
    """Get the top k predicted next notes for a sequence."""
    if len(sequence) < 64:
        sequence = [0] * (64 - len(sequence)) + sequence
    elif len(sequence) > 64:
        sequence = sequence[-64:]
    
    model.eval()
    with torch.no_grad():
        x = torch.tensor([sequence], dtype=torch.long).to(device)
        output = model(input_ids=x)
        
        logits = output.logits[0, -1, :vocab_size]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        
        # Get top k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        
        # Convert to notes
        top_notes = [int_to_note[idx] for idx in top_indices]
        
        return list(zip(top_notes, top_probs))

if __name__ == "__main__":
    # Setup argument parser
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model predictions")
    parser.add_argument("--model_path", type=str, default="./models/base_model/best_model.pt")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    args = parser.parse_args()
    
    # Load model and mappings
    model, note_to_int, int_to_note, vocab_size, device = load_model_and_mappings(
        args.model_path, args.processed_dir
    )
    
    # Load a sample sequence
    inputs = np.load(os.path.join(args.processed_dir, 'inputs.npy'))
    
    # Get a random sequence
    idx = np.random.randint(0, len(inputs))
    sequence = inputs[idx].astype(int).tolist()
    
    # Get predictions
    predictions = get_top_predictions(model, sequence, note_to_int, int_to_note, vocab_size, device)
    
    print("Sample sequence:")
    print([int_to_note[id] for id in sequence[-10:]])
    
    print("\nTop predicted next notes:")
    for note, prob in predictions:
        print(f"{note}: {prob:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.bar([note for note, _ in predictions], [prob for _, prob in predictions])
    plt.title("Top predicted next notes")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("note_predictions.png")
    print("Saved visualization to note_predictions.png")
