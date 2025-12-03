#!/usr/bin/env python3
"""
STEP 2: Resample Audio

Resamples audio to target sample rate (default 16kHz).

Usage:
    python scripts/preprocess/02_resample.py --input audio.wav --output resampled.wav --sr 16000
"""

import argparse
import librosa
import soundfile as sf
import sys


def resample_audio(input_path: str, output_path: str, target_sr: int = 16000):
    """Resample audio to target sample rate"""
    
    print(f"\n{'='*80}")
    print("AUDIO RESAMPLING")
    print(f"{'='*80}\n")
    
    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    print(f"  Original sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    
    # Resample if needed
    if sr != target_sr:
        print(f"\nResampling to {target_sr} Hz...")
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"  ✓ Resampled from {sr} Hz to {target_sr} Hz")
        print(f"  New samples: {len(audio_resampled)}")
    else:
        print(f"\n✓ Already at target sample rate ({target_sr} Hz)")
        audio_resampled = audio
    
    # Save
    sf.write(output_path, audio_resampled, target_sr, subtype='PCM_16')
    print(f"\n✓ Saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Resample audio to target sample rate"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output audio file')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate (default: 16000)')
    
    args = parser.parse_args()
    
    resample_audio(args.input, args.output, args.sr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
