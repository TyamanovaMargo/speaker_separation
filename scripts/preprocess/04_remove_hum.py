#!/usr/bin/env python3
"""
STEP 4: Remove Hum

Removes power line hum (50Hz, 60Hz, 100Hz, 120Hz) using notch filters.

Usage:
    python scripts/preprocess/04_remove_hum.py --input audio.wav --output dehum.wav --freq 50 60 100 120
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import sys


def remove_hum(audio: np.ndarray, sr: int, hum_freqs: list) -> np.ndarray:
    """
    Apply notch filters to remove power line hum
    """
    filtered = audio.copy()
    
    for hum_freq in hum_freqs:
        # Design notch filter
        q_factor = 30.0  # Quality factor (higher = narrower notch)
        w0 = hum_freq / (sr / 2)  # Normalize frequency
        
        b, a = signal.iirnotch(w0, q_factor, fs=sr)
        filtered = signal.filtfilt(b, a, filtered)
    
    return filtered


def process_hum_removal(input_path: str, output_path: str, hum_freqs: list):
    """Main hum removal process"""
    
    print(f"\n{'='*80}")
    print("HUM REMOVAL")
    print(f"{'='*80}\n")
    
    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    
    # Apply hum removal
    print(f"\nRemoving hum at frequencies: {hum_freqs} Hz...")
    audio_filtered = remove_hum(audio, sr, hum_freqs)
    
    print(f"  ✓ Applied {len(hum_freqs)} notch filters")
    
    # Calculate energy reduction
    original_energy = np.sum(audio ** 2)
    filtered_energy = np.sum(audio_filtered ** 2)
    energy_reduction = ((original_energy - filtered_energy) / original_energy) * 100
    
    print(f"  Energy reduction: {energy_reduction:.2f}%")
    
    # Save
    sf.write(output_path, audio_filtered, sr, subtype='PCM_16')
    print(f"\n✓ Saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Remove power line hum using notch filters"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output audio file')
    parser.add_argument('--freq', nargs='+', type=int, default=[50, 60, 100, 120],
                       help='Hum frequencies to remove (default: 50 60 100 120)')
    
    args = parser.parse_args()
    
    process_hum_removal(args.input, args.output, args.freq)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
