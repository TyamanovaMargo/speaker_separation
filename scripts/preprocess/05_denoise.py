#!/usr/bin/env python3
"""
STEP 5: Noise Reduction

Reduces background noise using spectral subtraction.

Usage:
    python scripts/preprocess/05_denoise.py --input audio.wav --output denoised.wav --strength 0.8
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import sys


def denoise_audio(audio: np.ndarray, sr: int, strength: float = 0.8) -> np.ndarray:
    """
    Simple noise reduction using noisereduce library
    """
    try:
        import noisereduce as nr
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=strength
        )
        return reduced
    
    except ImportError:
        print("⚠️  noisereduce not installed. Install with: pip install noisereduce")
        print("   Returning original audio without denoising.")
        return audio


def process_denoise(input_path: str, output_path: str, strength: float = 0.8):
    """Main denoising process"""
    
    print(f"\n{'='*80}")
    print("NOISE REDUCTION")
    print(f"{'='*80}\n")
    
    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    
    # Calculate original SNR estimate
    noise_len = int(0.5 * sr)
    if len(audio) > noise_len:
        noise_sample = audio[:noise_len]
        signal_sample = audio[noise_len:]
        noise_power = np.mean(noise_sample ** 2)
        signal_power = np.mean(signal_sample ** 2)
        if noise_power > 0:
            original_snr = 10 * np.log10(signal_power / noise_power)
        else:
            original_snr = 100.0
    else:
        original_snr = 20.0
    
    print(f"  Original SNR: {original_snr:.2f} dB")
    
    # Apply denoising
    print(f"\nApplying noise reduction (strength: {strength})...")
    audio_denoised = denoise_audio(audio, sr, strength)
    
    # Calculate new SNR
    if len(audio_denoised) > noise_len:
        noise_sample = audio_denoised[:noise_len]
        signal_sample = audio_denoised[noise_len:]
        noise_power = np.mean(noise_sample ** 2)
        signal_power = np.mean(signal_sample ** 2)
        if noise_power > 0:
            new_snr = 10 * np.log10(signal_power / noise_power)
        else:
            new_snr = 100.0
    else:
        new_snr = original_snr
    
    improvement = new_snr - original_snr
    
    print(f"  ✓ Denoising complete")
    print(f"  New SNR: {new_snr:.2f} dB")
    print(f"  SNR improvement: {improvement:.2f} dB")
    
    # Save
    sf.write(output_path, audio_denoised, sr, subtype='PCM_16')
    print(f"\n✓ Saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Reduce background noise"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output audio file')
    parser.add_argument('--strength', type=float, default=0.8,
                       help='Denoising strength 0.0-1.0 (default: 0.8)')
    
    args = parser.parse_args()
    
    if args.strength < 0.0 or args.strength > 1.0:
        print("Error: strength must be between 0.0 and 1.0")
        return 1
    
    process_denoise(args.input, args.output, args.strength)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
