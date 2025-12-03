#!/usr/bin/env python3
"""MossFormer2 8kHz Separation - Correct Bytes Handling"""

import argparse
import os
import sys
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import struct

def bytes_to_audio(byte_data, sample_rate=8000):
    """Convert bytes to numpy audio array"""
    # MossFormer2 outputs PCM bytes - convert to float32
    # Assuming 16-bit PCM (2 bytes per sample)
    num_samples = len(byte_data) // 2
    
    # Unpack bytes as signed 16-bit integers
    audio = np.array(struct.unpack(f'{num_samples}h', byte_data), dtype=np.float32)
    
    # Normalize from int16 range to float32 [-1, 1]
    audio = audio / 32768.0
    
    return audio

def separate_with_chunks(input_path, output_dir, chunk_seconds=30):
    print(f"\n{'='*80}")
    print("MOSSFORMER2 SEPARATION (8kHz - Chunked)")
    print(f"{'='*80}\n")
    
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import librosa
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    print(f"Input: {input_path}")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    
    if sr != 8000:
        print(f"\nResampling to 8kHz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)
        sr = 8000
        print("  ✓ Resampled")
    
    print("\nLoading model...")
    separation = pipeline(
        Tasks.speech_separation,
        model='damo/speech_mossformer2_separation_temporal_8k',
        device='cuda'
    )
    print("  ✓ Model loaded on GPU!")
    
    chunk_samples = chunk_seconds * 8000
    num_chunks = int(np.ceil(len(audio) / chunk_samples))
    
    print(f"\nProcessing {num_chunks} chunks of {chunk_seconds}s each")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    speaker1_chunks = []
    speaker2_chunks = []
    
    for i in tqdm(range(num_chunks), desc="Separating"):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start:end]
        chunk_len = len(chunk)
        
        chunk_path = f'/tmp/chunk_{i}.wav'
        sf.write(chunk_path, chunk, 8000)
        
        try:
            torch.cuda.empty_cache()
            
            result = separation(chunk_path)
            outputs = result.get('output_pcm_list', result)
            
            # Convert bytes to audio
            if isinstance(outputs[0], bytes):
                s1 = bytes_to_audio(outputs[0], 8000)
                s2 = bytes_to_audio(outputs[1], 8000)
            else:
                # Fallback for other formats
                s1 = np.array(outputs[0], dtype=np.float32).squeeze()
                s2 = np.array(outputs[1], dtype=np.float32).squeeze()
            
            # Ensure correct length
            if len(s1) < chunk_len:
                s1 = np.pad(s1, (0, chunk_len - len(s1)), mode='constant')
            elif len(s1) > chunk_len:
                s1 = s1[:chunk_len]
            
            if len(s2) < chunk_len:
                s2 = np.pad(s2, (0, chunk_len - len(s2)), mode='constant')
            elif len(s2) > chunk_len:
                s2 = s2[:chunk_len]
            
            speaker1_chunks.append(s1)
            speaker2_chunks.append(s2)
            
            # Debug first chunk
            if i == 0:
                print(f"\n  First chunk check:")
                print(f"    Speaker 1 RMS: {np.sqrt(np.mean(s1**2)):.6f}")
                print(f"    Speaker 2 RMS: {np.sqrt(np.mean(s2**2)):.6f}")
            
        except Exception as e:
            print(f"\n⚠️  Chunk {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            speaker1_chunks.append(np.zeros(chunk_len, dtype=np.float32))
            speaker2_chunks.append(np.zeros(chunk_len, dtype=np.float32))
        
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            torch.cuda.empty_cache()
    
    print("\n✓ Merging chunks...")
    speaker1 = np.concatenate(speaker1_chunks)
    speaker2 = np.concatenate(speaker2_chunks)
    
    print(f"  Final audio length: {len(speaker1)/8000:.2f} seconds")
    print(f"  Speaker 1 RMS: {np.sqrt(np.mean(speaker1**2)):.6f}")
    print(f"  Speaker 2 RMS: {np.sqrt(np.mean(speaker2**2)):.6f}")
    
    print("\nSaving results...")
    for i, (name, data) in enumerate([('speaker1.wav', speaker1), ('speaker2.wav', speaker2)], 1):
        path = os.path.join(output_dir, name)
        
        # Normalize
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data)) * 0.95
        
        sf.write(path, data, 8000, subtype='PCM_16')
        
        duration = len(data) / 8000
        rms = np.sqrt(np.mean(data ** 2))
        
        print(f"\n✓ Speaker {i}:")
        print(f"  File: {path}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  RMS: {rms:.4f}")
    
    # Quality metrics
    print(f"\n{'='*80}")
    print("QUALITY METRICS")
    print(f"{'='*80}\n")
    
    min_len = min(len(speaker1), len(speaker2))
    s1 = speaker1[:min_len]
    s2 = speaker2[:min_len]
    
    correlation = np.corrcoef(s1, s2)[0, 1]
    print(f"Cross-correlation: {correlation:.4f}")
    if correlation < 0.3:
        print("  ✓ Excellent separation")
    elif correlation < 0.6:
        print("  ⚙️  Good separation")
    else:
        print("  ⚠️  Moderate separation")
    
    energy1 = np.sum(s1 ** 2)
    energy2 = np.sum(s2 ** 2)
    energy_ratio = min(energy1, energy2) / max(energy1, energy2) if max(energy1, energy2) > 0 else 0
    
    print(f"\nEnergy ratio: {energy_ratio:.4f}")
    if energy_ratio > 0.3:
        print("  ✓ Both speakers present")
    else:
        print("  ⚠️  One speaker dominates")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Results: {output_dir}/")
    print(f"  • speaker1.wav")
    print(f"  • speaker2.wav")
    print()
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="MossFormer2 8kHz separation with chunking"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=30, 
                       help='Chunk size in seconds (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    return separate_with_chunks(args.input, args.output_dir, args.chunk_size)

if __name__ == '__main__':
    sys.exit(main())
