#!/usr/bin/env python3
"""
STEP 3: Declipping

Repairs clipped audio using interpolation.

Usage:
    python scripts/preprocess/03_declip.py --input audio.wav --output declipped.wav --threshold 0.95
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import sys


def declip_audio(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Basic declipping using interpolation
    """
    clipped_indices = np.where(np.abs(audio) > threshold)[0]
    
    if len(clipped_indices) == 0:
        return audio
    
    repaired = audio.copy()
    
    # Group consecutive clipped samples
    gaps = np.diff(clipped_indices)
    gap_starts = np.concatenate(([0], np.where(gaps > 1)[0] + 1))
    gap_ends = np.concatenate((np.where(gaps > 1)[0], [len(clipped_indices) - 1]))
    
    for start_idx, end_idx in zip(gap_starts, gap_ends):
        clip_start = clipped_indices[start_idx]
        clip_end = clipped_indices[end_idx]
        
        # Get surrounding unclipped values
        left_val = audio[max(0, clip_start - 1)]
        right_val = audio[min(len(audio) - 1, clip_end + 1)]
        
        # Linear interpolation
        interp_length = clip_end - clip_start + 1
        repaired[clip_start:clip_end+1] = np.linspace(left_val, right_val, interp_length)
    
    return repaired


def process_declip(input_path: str, output_path: str, threshold: float = 0.95):
    """Main declipping process"""
    
    print(f"\n{'='*80}")
    print("DECLIPPING")
    print(f"{'='*80}\n")
    
    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    # Check for clipping
    peak = np.max(np.abs(audio))
    clipped_samples = np.sum(np.abs(audio) > threshold)
    clipped_percent = (clipped_samples / len(audio)) * 100
    
    print(f"  Sample rate: {sr} Hz")
    print(f"  Peak amplitude: {peak:.4f}")
    print(f"  Clipped samples: {clipped_samples} ({clipped_percent:.2f}%)")
    
    if clipped_samples == 0:
        print("\n✓ No clipping detected. No processing needed.")
        # Just save original
        sf.write(output_path, audio, sr, subtype='PCM_16')
        print(f"✓ Saved to: {output_path}")
        print()
        return
    
    # Apply declipping
    print(f"\nApplying declipping (threshold: {threshold})...")
    audio_declipped = declip_audio(audio, threshold)
    
    # Check results
    new_peak = np.max(np.abs(audio_declipped))
    new_clipped = np.sum(np.abs(audio_declipped) > threshold)
    
    print(f"  ✓ Declipped {clipped_samples} samples")
    print(f"  New peak amplitude: {new_peak:.4f}")
    print(f"  Remaining clipped: {new_clipped}")
    
    # Save
    sf.write(output_path, audio_declipped, sr, subtype='PCM_16')
    print(f"\n✓ Saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Repair clipped audio using interpolation"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output audio file')
    parser.add_argument('--threshold', type=float, default=0.95, 
                       help='Clipping threshold (default: 0.95)')
    
    args = parser.parse_args()
    
    process_declip(args.input, args.output, args.threshold)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
