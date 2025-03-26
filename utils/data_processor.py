# ~/irish_folk_model/utils/data_processor.py
import os
import numpy as np
import pickle
from music21 import converter, instrument, note, chord
import glob
import logging
from tqdm import tqdm
import time
import torch
from torch.utils.data import TensorDataset

def process_midi_files(midi_dir, limit=None, save_interval=1000, processed_dir=None):
    """
    Extract notes and chords from MIDI files with progress bar and resumable processing.
    
    Args:
        midi_dir: Directory containing MIDI files
        limit: Maximum number of files to process (None for all)
        save_interval: Save intermediate results every N files
        processed_dir: Directory to save intermediate results
    """
    all_notes = []
    
    # Check if we have previously processed notes
    if processed_dir and os.path.exists(os.path.join(processed_dir, 'partial_notes.pkl')):
        with open(os.path.join(processed_dir, 'partial_notes.pkl'), 'rb') as f:
            all_notes = pickle.load(f)
        print(f"Loaded {len(all_notes)} previously processed notes")
        logging.info(f"Loaded {len(all_notes)} previously processed notes")
    
    # Find all MIDI files recursively
    midi_files = []
    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_files.append(os.path.join(root, file))
    
    # Limit number of files if specified
    if limit:
        midi_files = midi_files[:limit]
    
    print(f"Found {len(midi_files)} MIDI files in {midi_dir} and subdirectories")
    logging.info(f"Found {len(midi_files)} MIDI files in {midi_dir} and subdirectories")
    
    if len(midi_files) == 0:
        print(f"ERROR: No MIDI files found in {midi_dir} or its subdirectories!")
        logging.error(f"No MIDI files found in {midi_dir} or its subdirectories!")
        return all_notes
    
    # Skip files we've already processed
    processed_count = len(all_notes) // 20  # Rough estimate of notes per file
    if processed_count > 0:
        print(f"Skipping approximately {processed_count} already processed files")
        midi_files = midi_files[processed_count:]
    
    # Process each MIDI file with progress bar
    for i, file_path in enumerate(tqdm(midi_files, desc="Processing MIDI files")):
        try:
            midi = converter.parse(file_path)
            notes_to_parse = None
            
            try:  # File has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                if s2 and s2.parts:
                    notes_to_parse = s2.parts[0].recurse() 
                else:
                    notes_to_parse = midi.flat.notes
            except Exception as e:
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    all_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    all_notes.append('.'.join(str(n) for n in element.normalOrder))
            
            # Save intermediate results
            if processed_dir and i > 0 and i % save_interval == 0:
                with open(os.path.join(processed_dir, 'partial_notes.pkl'), 'wb') as f:
                    pickle.dump(all_notes, f)
                print(f"\nSaved {len(all_notes)} notes after processing {i} files")
                
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            logging.error(f"Error processing {file_path}: {e}")
        
        # Add small delay to allow interruption
        if i % 10 == 0:
            time.sleep(0.01)
    
    print(f"\nExtracted {len(all_notes)} notes from MIDI files")
    logging.info(f"Extracted {len(all_notes)} notes from MIDI files")
    
    # Save final results
    if processed_dir:
        with open(os.path.join(processed_dir, 'all_notes.pkl'), 'wb') as f:
            pickle.dump(all_notes, f)
        
        # Remove partial file if successful
        if os.path.exists(os.path.join(processed_dir, 'partial_notes.pkl')):
            os.remove(os.path.join(processed_dir, 'partial_notes.pkl'))
    
    return all_notes

def create_mapping(notes):
    """Create mapping between notes and integers."""
    vocabulary = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(vocabulary)}
    int_to_note = {i: note for i, note in enumerate(vocabulary)}
    
    return note_to_int, int_to_note, len(vocabulary)

# Modify utils/data_processor.py to include better error handling

def prepare_sequences(notes, note_to_int, sequence_length=64):
    """Prepare the sequences for training with improved error handling."""
    network_input = []
    network_output = []
    
    try:
        # Process in smaller batches to reduce memory pressure
        batch_size = 5000
        total_sequences = len(notes) - sequence_length
        
        for batch_start in range(0, total_sequences, batch_size):
            batch_end = min(batch_start + batch_size, total_sequences)
            print(f"Processing sequences {batch_start} to {batch_end} of {total_sequences}")
            
            for i in range(batch_start, batch_end):
                sequence_in = notes[i:i + sequence_length]
                sequence_out = notes[i + sequence_length]
                
                # Skip sequences with notes not in vocabulary
                try:
                    input_sequence = [note_to_int[char] for char in sequence_in]
                    output_value = note_to_int[sequence_out]
                    
                    network_input.append(input_sequence)
                    network_output.append(output_value)
                except KeyError as e:
                    print(f"Skipping sequence with unknown note: {e}")
                    continue
            
            # Save intermediate results to avoid memory issues
            print(f"Processed {len(network_input)} sequences so far...")
    
    except Exception as e:
        print(f"Error in sequence preparation: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert to numpy arrays in chunks to avoid memory issues
    try:
        print("Converting to numpy arrays...")
        # Convert in chunks of 10,000 sequences
        max_chunk = 10000
        network_input_np = []
        
        for i in range(0, len(network_input), max_chunk):
            chunk = network_input[i:i + max_chunk]
            network_input_np.append(np.array(chunk))
        
        network_input = np.vstack(network_input_np)
        network_output = np.array(network_output)
        
        print(f"Final array shapes - input: {network_input.shape}, output: {network_output.shape}")
    except Exception as e:
        print(f"Error converting to numpy arrays: {e}")
        import traceback
        traceback.print_exc()
    
    return network_input, network_output

def process_dataset(raw_dir, processed_dir, sequence_length=64, limit=None):
    """Process the dataset with improved error handling."""
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # Extract notes from MIDI files
        print("Starting MIDI processing...")
        notes = process_midi_files(raw_dir, limit=limit)
        
        if len(notes) == 0:
            raise ValueError("No notes were extracted from MIDI files")
        
        print(f"Extracted {len(notes)} notes from MIDI files")
        
        # Save notes to file for recovery if later steps fail
        notes_file = os.path.join(processed_dir, 'extracted_notes.pkl')
        print(f"Saving extracted notes to {notes_file}")
        with open(notes_file, 'wb') as f:
            pickle.dump(notes, f)
        
        # Create mappings
        note_to_int, int_to_note, vocab_size = create_mapping(notes)
        
        # Save mappings
        with open(os.path.join(processed_dir, 'note_to_int.pkl'), 'wb') as f:
            pickle.dump(note_to_int, f)
        
        with open(os.path.join(processed_dir, 'int_to_note.pkl'), 'wb') as f:
            pickle.dump(int_to_note, f)
        
        # Save vocabulary size
        with open(os.path.join(processed_dir, 'vocab_size.txt'), 'w') as f:
            f.write(str(vocab_size))
        
        # Prepare sequences
        print("Preparing sequences...")
        inputs, targets = prepare_sequences(notes, note_to_int, sequence_length)
        
        # Save processed data
        print("Saving processed data...")
        np.save(os.path.join(processed_dir, 'inputs.npy'), inputs)
        np.save(os.path.join(processed_dir, 'targets.npy'), targets)
        
        print(f"Processed {len(notes)} notes with vocabulary size {vocab_size}")
        print(f"Created {len(inputs)} sequences of length {sequence_length}")
        
        return inputs, targets, note_to_int, int_to_note, vocab_size
    
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process MIDI files for music generation")
    parser.add_argument("--data_dir", type=str, default="../data/raw", help="Directory with raw MIDI files")
    parser.add_argument("--processed_dir", type=str, default="../data/processed", help="Directory to save processed data")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length for training")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (for testing)")
    args = parser.parse_args()
    
    process_dataset(args.data_dir, args.processed_dir, args.seq_length, args.limit)
