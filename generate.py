# ~/irish_folk_model/generate.py
import os
import argparse
import torch
import pickle
import numpy as np
from music21 import note, chord, stream, instrument
from training.base_trainer import create_model

def generate_music(model, start_notes, int_to_note, vocab_size, sequence_length=64, predict_length=500, temperature=1.0, device='cuda'):
    """Generate music using the trained model."""
    # Convert to input format
    pattern = start_notes
    generated_notes = []
    
    # Generate notes
    model.eval()
    
    # Do a test prediction to get the actual output size
    with torch.no_grad():
        x = torch.tensor([pattern[-sequence_length:]], dtype=torch.long).to(device)
        output = model(input_ids=x)
        actual_vocab_size = output.logits.shape[-1]
        print(f"Model output size: {actual_vocab_size}, Expected vocab size: {vocab_size}")
        
        # Use the smaller of the two to avoid index errors
        use_vocab_size = min(vocab_size, actual_vocab_size)
        
        for _ in range(predict_length):
            # Prepare input sequence
            x = torch.tensor([pattern[-sequence_length:]], dtype=torch.long).to(device)
            
            # Generate prediction
            output = model(input_ids=x)
            
            # Apply temperature to logits
            logits = output.logits[0, -1, :use_vocab_size] / temperature
            
            # Convert to probability distribution
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            # Make sure probabilities sum to 1
            probs = probs / np.sum(probs)
            
            # Debug info for first few iterations
            if len(generated_notes) < 2:
                print(f"Prob array shape: {probs.shape}, Vocab used: {use_vocab_size}")
            
            # Sample from the distribution
            prediction = np.random.choice(use_vocab_size, p=probs)
            
            # Add to generated notes - ensure we don't exceed int_to_note keys
            if prediction in int_to_note:
                generated_notes.append(int_to_note[prediction])
            else:
                # Fallback: use a random existing note
                valid_idx = np.random.choice(list(int_to_note.keys()))
                generated_notes.append(int_to_note[valid_idx])
                print(f"Warning: Generated token {prediction} not in vocabulary, using fallback")
            
            # Update pattern
            pattern.append(prediction)
    
    return generated_notes

def notes_to_midi(notes, filename):
    """Convert notes to a MIDI file."""
    output_notes = []
    offset = 0
    
    for pattern in notes:
        try:
            # Pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # Pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
        except Exception as e:
            print(f"Error with note {pattern}: {e}")
            continue
            
        # Increase offset for next note
        offset += 0.5
    
    # Create stream
    midi_stream = stream.Stream(output_notes)
    
    # Write to file
    midi_stream.write('midi', fp=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using trained model")
    parser.add_argument("--model_path", type=str, default="./models/base_model/best_model.pt", help="Path to trained model")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Directory to save generated MIDI")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--length", type=int, default=500, help="Length of generated sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load mappings and vocab size
    with open(os.path.join(args.processed_dir, 'int_to_note.pkl'), 'rb') as f:
        int_to_note = pickle.load(f)
    
    with open(os.path.join(args.processed_dir, 'note_to_int.pkl'), 'rb') as f:
        note_to_int = pickle.load(f)
    
    with open(os.path.join(args.processed_dir, 'vocab_size.txt'), 'r') as f:
        vocab_size = int(f.read().strip())
    
    print(f"Loaded vocab size: {vocab_size}")
    print(f"int_to_note has {len(int_to_note)} entries")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(vocab_size, sequence_length=64)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Load a random sequence to start generation
    inputs = np.load(os.path.join(args.processed_dir, 'inputs.npy'))
    
    # Generate multiple samples
    for i in range(args.num_samples):
        # Pick a random starting sequence
        start_idx = np.random.randint(0, len(inputs))
        start_seq = inputs[start_idx].astype(int).tolist()
        
        # Generate music
        print(f"Generating sample {i+1}/{args.num_samples}...")
        generated_notes = generate_music(
            model, start_seq, int_to_note, vocab_size, 
            predict_length=args.length, temperature=args.temperature,
            device=device
        )
        
        # Save to MIDI file
        output_file = os.path.join(args.output_dir, f"sample_{i+1}.mid")
        notes_to_midi(generated_notes, output_file)
        print(f"Saved to {output_file}")
    
    print("Generation complete!")
