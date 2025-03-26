import os
import torch
import pickle
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from music21 import note, chord, stream

# Force synchronous CUDA operations for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_model_and_mappings(model_path, processed_dir):
    """Load the trained model and mapping dictionaries."""
    # Load mapping directly from the model directory
    try:
        with open(os.path.join(model_path, 'int_to_note.pkl'), 'rb') as f:
            int_to_note = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: int_to_note.pkl not found in model directory. Trying processed directory...")
        with open(os.path.join(processed_dir, 'int_to_note.pkl'), 'rb') as f:
            int_to_note = pickle.load(f)
    
    # Force CPU to avoid CUDA errors
    device = torch.device("cpu")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path, use_safetensors=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Set pad token for generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model.to(device)
    model.eval()
    
    # Get vocab size from model config
    vocab_size = model.config.vocab_size
    max_length = model.config.n_positions
    print(f"Model maximum sequence length: {max_length}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create note_to_int from int_to_note
    note_to_int = {v: k for k, v in int_to_note.items()}
    
    return model, tokenizer, note_to_int, int_to_note, vocab_size, max_length, device


    """Create a short seed sequence matching the tokenizer's expectations."""
    # Find valid tokens by checking which ones the tokenizer recognizes properly
    valid_tokens = []
    for i in range(10):  # Test first 10 tokens
        token_text = str(i)
        token_ids = tokenizer.encode(token_text)
        if 0 < len(token_ids) < 3:  # Should encode to a single token or two
            valid_tokens.append(token_text)
    
    if not valid_tokens:
        valid_tokens = ["0", "1", "2"]  # Fallback
    
    seed_text = " ".join(valid_tokens[:3])
    print(f"Using valid seed tokens: {seed_text}")
    
    # Debug tokenization
    seed_tokens = tokenizer(seed_text, return_tensors="pt")
    print(f"Tokenized to: {seed_tokens['input_ids'][0].tolist()}")
    
    return seed_tokens

def generate_music_token_by_token(model, tokenizer, seed_tokens, int_to_note, vocab_size, max_length, device):
    """Generate music one token at a time for more control."""
    try:
        current_input_ids = seed_tokens["input_ids"].clone()
        generated_tokens = []
        
        # Generate tokens one by one
        for _ in range(max_length - current_input_ids.shape[1]):
            with torch.no_grad():
                outputs = model(input_ids=current_input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                # Apply temperature and sampling
                temperature = 0.7
                next_token_logits = next_token_logits / temperature
                filtered_logits = torch.softmax(next_token_logits, dim=-1)
                
                # Sample from distribution
                next_token = torch.multinomial(filtered_logits, num_samples=1)
                
                # Log the token for debugging
                token_id = next_token.item()
                token_text = tokenizer.decode(next_token)
                print(f"Generated token: {token_id} -> '{token_text}'")
                
                # Append to generated tokens
                generated_tokens.append(token_id)
                
                # Update input_ids for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we reach end of sequence token
                if token_id == tokenizer.eos_token_id:
                    break
        
        # Debug all generated tokens
        print(f"All generated token IDs: {generated_tokens}")
        
        # Try to map tokens to notes
        generated_notes = []
        for token_id in generated_tokens:
            if token_id in int_to_note:
                note_value = int_to_note[token_id]
                generated_notes.append(note_value)
                print(f"Mapped token {token_id} to note {note_value}")
            else:
                print(f"Token {token_id} not found in mapping dictionary")
        
        return generated_notes
    
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return []

def notes_to_midi(notes, filename):
    """Convert sequence of notes to MIDI file."""
    if not notes:
        print("No valid notes generated")
        return False
    
    output_notes = []
    offset = 0
    
    for pattern in notes:
        try:
            if ('.' in str(pattern)) or str(pattern).isdigit():
                # Handle chord notation
                if '.' in str(pattern):
                    notes_in_chord = str(pattern).split('.')
                    chord_notes = []
                    for current_note in notes_in_chord:
                        try:
                            note_num = int(current_note) 
                            if 0 <= note_num <= 127:  # MIDI note range check
                                new_note = note.Note(note_num)
                                chord_notes.append(new_note)
                        except Exception as e:
                            print(f"Skipping invalid note in chord {current_note}: {e}")
                    if chord_notes:
                        new_chord = chord.Chord(chord_notes)
                        new_chord.offset = offset
                        output_notes.append(new_chord)
                else:
                    # Handle single note as MIDI number
                    try:
                        note_num = int(pattern)
                        if 0 <= note_num <= 127:  # MIDI note range check
                            new_note = note.Note(note_num)
                            new_note.offset = offset
                            output_notes.append(new_note)
                    except Exception as e:
                        print(f"Skipping invalid note {pattern}: {e}")
            else:
                # Handle string-type notes if they exist in your vocabulary
                try:
                    new_note = note.Note(pattern)
                    new_note.offset = offset
                    output_notes.append(new_note)
                except Exception as e:
                    print(f"Skipping invalid note format: {pattern}")
                    continue
        except Exception as e:
            print(f"Error processing note {pattern}: {e}")
            continue
            
        offset += 0.5
    
    if not output_notes:
        print("No valid notes were created")
        return False
        
    midi_stream = stream.Stream(output_notes)
    try:
        midi_stream.write('midi', fp=filename)
        print(f"Successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving MIDI file: {e}")
        return False

def get_seed_sequence(tokenizer, note_to_int, vocab_size):
    """Create a short seed sequence matching the tokenizer's expectations."""
    # Use the simplest tokens that are definitely in vocabulary
    seed_notes = ["0", "1", "2"]
    seed_text = " ".join(seed_notes)
    print(f"Using simple numeric seed: {seed_text}")
    
    # Tokenize with safety checks
    seed_tokens = tokenizer(seed_text, return_tensors="pt", padding=True)
    
    # Verify token IDs are within vocabulary range
    if torch.max(seed_tokens["input_ids"]) >= vocab_size:
        print(f"Warning: Input contains tokens outside vocabulary range. Clamping values.")
        seed_tokens["input_ids"] = torch.clamp(
            seed_tokens["input_ids"], 
            0, 
            vocab_size - 1
        )
    
    return seed_tokens


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MIDI using trained GPT-2 model")
    parser.add_argument("--model_path", type=str, default="models/huggingface-gpt2/final_model")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="generated_tune.mid")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer, note_to_int, int_to_note, vocab_size, max_length, device = load_model_and_mappings(
        args.model_path, args.processed_dir
    )
    
    # Examine vocabularies for debugging
    print("\nExamining vocabulary mappings...")
    print(f"int_to_note dictionary has {len(int_to_note)} entries")
    print(f"note_to_int dictionary has {len(note_to_int)} entries")
    
    # Sample keys from each
    print("\nSample from int_to_note (first 5 entries):")
    sample_keys = list(int_to_note.items())[:5]
    for key, value in sample_keys:
        print(f"  {key} -> {value}")
    
    # Generate seed sequence with validation
    seed_tokens = get_seed_sequence(tokenizer, note_to_int, vocab_size)
    
    # Generate music token by token with extensive logging
    print("\nGenerating music tokens one by one...")
    generated_notes = generate_music_token_by_token(
        model, tokenizer, seed_tokens, int_to_note, vocab_size, max_length, device
    )
    
    # Save as MIDI
    if generated_notes:
        print(f"\nGenerated {len(generated_notes)} notes")
        print(f"Sample of generated notes: {generated_notes[:10]}")
        notes_to_midi(generated_notes, args.output)
    else:
        print("\nFailed to generate any valid notes")

