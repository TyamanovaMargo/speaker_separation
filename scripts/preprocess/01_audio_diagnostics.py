#!/usr/bin/env python3
"""
STEP 1: Audio Diagnostics

Analyzes audio quality and detects issues:
- Sample rate
- SNR (Signal-to-Noise Ratio)
- Clipping detection
- Hum/buzz detection
- Dynamic range

Usage:
    python scripts/preprocess/01_audio_diagnostics.py --input audio.wav
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import fft
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def analyze_audio(audio_path: str) -> dict:
    """
    Comprehensive audio quality analysis
    
    Returns:
        dict with metrics: snr, clipping, hum_detected, etc.
    """
    print(f"\n{'='*80}")
    print("AUDIO DIAGNOSTICS")
    print(f"{'='*80}\n")
    
    # Load audio
    print(f"Loading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    metrics = {
        'file': audio_path,
        'sample_rate': int(sr),
        'duration_seconds': float(len(audio) / sr),
        'samples': int(len(audio))
    }
    
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    
    # 1. Peak amplitude / Clipping detection
    peak_amplitude = np.max(np.abs(audio))
    metrics['peak_amplitude'] = float(peak_amplitude)
    metrics['is_clipped'] = bool(peak_amplitude > 0.95)
    
    print(f"\n1. Peak Amplitude: {peak_amplitude:.4f}")
    if metrics['is_clipped']:
        print("   ⚠️  CLIPPING DETECTED!")
    else:
        print("   ✓ No clipping")
    
    # 2. SNR Estimation
    noise_duration = int(0.5 * sr)
    if len(audio) > noise_duration:
        noise_sample = audio[:noise_duration]
        signal_sample = audio[noise_duration:]
        
        noise_power = np.mean(noise_sample ** 2)
        signal_power = np.mean(signal_sample ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100.0
    else:
        snr = 20.0
    
    metrics['snr_db'] = float(snr)
    
    print(f"\n2. Signal-to-Noise Ratio: {snr:.2f} dB")
    if snr < 5:
        print("   ⚠️  VERY LOW SNR - Extremely poor quality")
    elif snr < 10:
        print("   ⚠️  LOW SNR - Poor quality")
    elif snr < 20:
        print("   ⚙️  MODERATE SNR - Acceptable quality")
    else:
        print("   ✓ GOOD SNR - High quality")
    
    # 3. Hum/Buzz Detection
    hum_frequencies = [50, 60, 100, 120]
    detected_hums = []
    
    n_fft = min(8192, len(audio))
    freqs = fft.fftfreq(n_fft, 1/sr)
    fft_vals = np.abs(fft.fft(audio[:n_fft]))
    
    for hum_freq in hum_frequencies:
        freq_idx = np.argmin(np.abs(freqs - hum_freq))
        if freq_idx < len(fft_vals) // 2:
            local_region = fft_vals[max(0, freq_idx-5):min(len(fft_vals)//2, freq_idx+5)]
            if len(local_region) > 0:
                peak_energy = fft_vals[freq_idx]
                mean_energy = np.mean(local_region)
                if peak_energy > 3 * mean_energy:
                    detected_hums.append(hum_freq)
    
    metrics['hum_detected'] = bool(len(detected_hums) > 0)
    metrics['hum_frequencies'] = [int(f) for f in detected_hums]
    
    print(f"\n3. Hum/Buzz Detection:")
    if detected_hums:
        print(f"   ⚠️  HUM DETECTED at: {detected_hums} Hz")
    else:
        print("   ✓ No power line hum detected")
    
    # 4. Dynamic Range
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        dynamic_range = 20 * np.log10(peak_amplitude / rms)
    else:
        dynamic_range = 0.0
    
    metrics['dynamic_range_db'] = float(dynamic_range)
    metrics['rms'] = float(rms)
    
    print(f"\n4. Dynamic Range: {dynamic_range:.2f} dB")
    print(f"   RMS Level: {rms:.4f}")
    
    # 5. Zero Crossing Rate
    zcr = np.mean(librosa.zero_crossings(audio))
    metrics['zero_crossing_rate'] = float(zcr)
    
    print(f"\n5. Zero Crossing Rate: {zcr:.4f}")
    
    # Overall Quality Assessment
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}\n")
    
    quality_score = 0
    issues = []
    
    if snr >= 20:
        quality_score += 40
    elif snr >= 10:
        quality_score += 20
    else:
        issues.append("Low SNR")
    
    if not metrics['is_clipped']:
        quality_score += 30
    else:
        issues.append("Clipping detected")
    
    if not detected_hums:
        quality_score += 30
    else:
        issues.append("Power line hum")
    
    metrics['quality_score'] = int(quality_score)
    metrics['issues'] = issues
    
    if quality_score >= 80:
        print("Quality: GOOD ✓")
        print("Recommendation: Use 'good' quality mode")
    elif quality_score >= 50:
        print("Quality: MODERATE ⚙️")
        print("Recommendation: Use 'bad' quality mode (default)")
    else:
        print("Quality: POOR ⚠️")
        print("Recommendation: Use 'terrible' quality mode")
    
    if issues:
        print(f"\nIssues found: {', '.join(issues)}")
    
    print()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Analyze audio quality and detect issues"
    )
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Run diagnostics
    metrics = analyze_audio(args.input)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())