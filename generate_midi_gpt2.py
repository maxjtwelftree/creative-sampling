import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
import numpy as np
from music21 import note, chord, stream

def load_model_and_tokenizer(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Fix attention mask issue
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def generate_notes(model, tokenizer, seed_text, max_length=100, temperature=1.0):
    # Prepare input with proper attention mask
    inputs = tokenizer(
        seed_text, 
        return_tensors='pt', 
        padding=True,
        truncation=True
    )
    
    # Safety check for token IDs
    if torch.max(inputs.input_ids) >= model.config.vocab_size:
        invalid_tokens = [int(tok) for tok in inputs.input_ids[0] if tok >= model.config.vocab_size]
        print(f"Warning: Invalid token IDs detected: {invalid_tokens}")
        # Filter or replace invalid tokens
        inputs.input_ids = torch.clamp(inputs.input_ids, 0, model.config.vocab_size - 1)
    
    try:
        # Generate with proper configuration
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=min(max_length, model.config.max_position_embeddings),
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Process the generated text into musical notes
        # This assumes your model was trained to output space-separated note tokens
        notes = generated_text.split()
        
        # Validate notes are in a MIDI-compatible format
        valid_notes = []
        for note_str in notes:
            # Keep only notes that are either digits or contain dots (for chords)
            if note_str.replace('.', '').isdigit():
                valid_notes.append(note_str)
        
        return valid_notes
        
    except Exception as e:
        print(f"Generation error: {e}")
        return []

def notes_to_midi(notes, filename):
    if not notes:
        print("No valid notes to convert to MIDI")
        return False
        
    output_notes = []
    offset = 0
    
    for pattern in notes:
        try:
            if ('.' in pattern) or pattern.isdigit():
                if '.' in pattern:
                    # Handle chord notation (e.g., "60.64.67")
                    notes_in_chord = pattern.split('.')
                    chord_notes = []
                    for current_note in notes_in_chord:
                        try:
                            new_note = note.Note(int(current_note))
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
                        new_note = note.Note(int(pattern))
                        new_note.offset = offset
                        output_notes.append(new_note)
                    except Exception as e:
                        print(f"Skipping invalid note {pattern}: {e}")
            else:
                # Skip invalid notes
                print(f"Skipping non-numeric note: {pattern}")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MIDI using GPT-2")
    parser.add_argument("--model_path", type=str, default="models/huggingface-gpt2/final_model")
    parser.add_argument("--output", type=str, default="generated_tune.mid")
    parser.add_argument("--seed_text", type=str, default="60 62 64 67")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    print(f"Generating with seed text: '{args.seed_text}'")
    generated_notes = generate_notes(
        model, tokenizer, args.seed_text, 
        max_length=args.max_length, temperature=args.temperature
    )
    
    print(f"Generated {len(generated_notes)} notes")
    if generated_notes:
        print(f"Sample of generated notes: {generated_notes[:10]}...")
        
    # Convert to MIDI and save
    success = notes_to_midi(generated_notes, args.output)
    
    if success:
        print(f"Generation complete: {args.output}")
    else:
        print("Failed to generate valid MIDI file")

