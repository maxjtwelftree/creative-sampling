# ~/irish_folk_model/interactive.py
import os
import torch
import pickle
import numpy as np
from music21 import note, chord, stream, instrument, converter
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

def generate_with_seed(model, seed_notes, note_to_int, int_to_note, vocab_size, 
                       length=100, temperature=1.0, device='cuda'):
    """Generate music with a specific seed phrase."""
    # Convert seed notes to token IDs
    try:
        seed_ids = [note_to_int[note] for note in seed_notes]
    except KeyError:
        print("One or more notes in the seed are not in the vocabulary.")
        return []
    
    # Pad if needed
    if len(seed_ids) < 64:
        seed_ids = [0] * (64 - len(seed_ids)) + seed_ids
    elif len(seed_ids) > 64:
        seed_ids = seed_ids[-64:]
    
    # Generate
    pattern = seed_ids.copy()
    generated_notes = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([pattern[-64:]], dtype=torch.long).to(device)
            output = model(input_ids=x)
            
            logits = output.logits[0, -1, :vocab_size] / temperature
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            prediction = np.random.choice(vocab_size, p=probs)
            
            generated_notes.append(int_to_note[prediction])
            pattern.append(prediction)
    
    return generated_notes

def notes_to_midi(notes, filename):
    output_notes = []
    offset = 0
    
    for pattern in notes:
        try:
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                output_notes.append(new_note)
        except Exception as e:
            print(f"Error with note {pattern}: {e}")
            continue
            
        offset += 0.5
    
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)
    print(f"Saved to {filename}")

def load_midi_as_seed(midi_file, note_to_int):
    """Load a MIDI file and convert to seed notes."""
    notes = []
    try:
        midi = converter.parse(midi_file)
        notes_to_parse = None
        
        try:
            s2 = instrument.partitionByInstrument(midi)
            if s2 and s2.parts:
                notes_to_parse = s2.parts[0].recurse() 
            else:
                notes_to_parse = midi.flat.notes
        except:
            notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
        # Filter to only notes in vocabulary
        notes = [note for note in notes if note in note_to_int]
        
        return notes
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive music generation")
    parser.add_argument("--model_path", type=str, default="./models/base_model/best_model.pt")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    parser.add_argument("--output", type=str, default="./interactive_output.mid")
    parser.add_argument("--seed_midi", type=str, help="Optional MIDI file to use as seed")
    parser.add_argument("--length", type=int, default=200, help="Length to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # Load model and mappings
    model, note_to_int, int_to_note, vocab_size, device = load_model_and_mappings(
        args.model_path, args.processed_dir
    )
    
    # Get seed phrase
    if args.seed_midi:
        seed_notes = load_midi_as_seed(args.seed_midi, note_to_int)
        if not seed_notes:
            print("Could not extract usable notes from MIDI. Using random seed.")
            # Use a random sequence from the training data
            inputs = np.load(os.path.join(args.processed_dir, 'inputs.npy'))
            start_idx = np.random.randint(0, len(inputs))
            seed_ids = inputs[start_idx].astype(int).tolist()
            seed_notes = [int_to_note[id] for id in seed_ids]
    else:
        # Use a random sequence from the training data
        inputs = np.load(os.path.join(args.processed_dir, 'inputs.npy'))
        start_idx = np.random.randint(0, len(inputs))
        seed_ids = inputs[start_idx].astype(int).tolist()
        seed_notes = [int_to_note[id] for id in seed_ids]
    
    print(f"Generating with seed: {seed_notes[:10]}... (truncated)")
    
    # Generate
    generated_notes = generate_with_seed(
        model, seed_notes, note_to_int, int_to_note, vocab_size,
        length=args.length, temperature=args.temperature, device=device
    )
    
    # Save
    notes_to_midi(generated_notes, args.output)
