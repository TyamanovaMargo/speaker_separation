#!/usr/bin/env python3
"""
STEP 6: Normalize Audio

Normalizes audio amplitude to prevent clipping in later stages.

Usage:
    python scripts/preprocess/06_normalize.py --input audio.wav --output normalized.wav --level -3
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import sys


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        audio: Input audio
        target_db: Target level in dB (default: -3dB)
    
    Returns:
        Normalized audio
    """
    # Convert dB to linear scale
    target_linear = 10 ** (target_db / 20)
    
    # Find current peak
    peak = np.max(np.abs(audio))
    
    if peak > 0:
        # Normalize
        normalized = audio * (target_linear / peak)
    else:
        normalized = audio
    
    return normalized


def process_normalize(input_path: str, output_path: str, target_db: float = -3.0):
    """Main normalization process"""
    
    print(f"\n{'='*80}")
    print("AUDIO NORMALIZATION")
    print(f"{'='*80}\n")
    
    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    original_peak = np.max(np.abs(audio))
    original_rms = np.sqrt(np.mean(audio ** 2))
    
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    print(f"  Original peak: {original_peak:.4f} ({20*np.log10(original_peak):.2f} dB)")
    print(f"  Original RMS: {original_rms:.4f}")
    
    # Normalize
    print(f"\nNormalizing to {target_db} dB...")
    audio_normalized = normalize_audio(audio, target_db)
    
    new_peak = np.max(np.abs(audio_normalized))
    new_rms = np.sqrt(np.mean(audio_normalized ** 2))
    
    print(f"  ✓ Normalization complete")
    print(f"  New peak: {new_peak:.4f} ({20*np.log10(new_peak):.2f} dB)")
    print(f"  New RMS: {new_rms:.4f}")
    
    # Save
    sf.write(output_path, audio_normalized, sr, subtype='PCM_16')
    print(f"\n✓ Saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Normalize audio amplitude"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output audio file')
    parser.add_argument('--level', type=float, default=-3.0,
                       help='Target level in dB (default: -3.0)')
    
    args = parser.parse_args()
    
    process_normalize(args.input, args.output, args.level)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
