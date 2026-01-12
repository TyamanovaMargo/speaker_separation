#!/usr/bin/env python3
"""
ClearerVoice-Studio Speaker Separation - V2 (QUALITY FOCUSED)
==============================================================
Key improvements over v1:
1. Longer overlap (5s) with Hann window crossfade
2. Spectral gating to remove musical noise artifacts
3. Crosstalk reduction between speakers
4. VAD-based silence masking
5. Adaptive chunk boundaries at speech pauses
6. Multi-pass refinement for difficult separations
7. High-pass filtering to remove rumble

Performance on RTX A5000:
  - Base PyTorch: 1x
  - FP16: ~1.5-2x
  - torch.compile: ~2x  
  - TensorRT: ~3-5x
"""

import os
import sys
import argparse
import time
import numpy as np
import soundfile as sf
import torch
import librosa
from tqdm import tqdm
from scipy.signal import butter, sosfilt
from scipy.ndimage import uniform_filter, uniform_filter1d

from clearvoice import ClearVoice


# =============================================================================
# Optimization Levels
# =============================================================================

class OptLevel:
    NONE = 0        # Base PyTorch
    FP16 = 1        # Half precision
    COMPILE = 2     # torch.compile with inductor
    TENSORRT = 3    # TensorRT backend


OPT_NAMES = {
    0: "None (Base PyTorch)",
    1: "FP16 (Half Precision)", 
    2: "torch.compile (Inductor)",
    3: "TensorRT"
}


# =============================================================================
# Model Optimization Functions
# =============================================================================

def check_tensorrt():
    """Check if TensorRT is available"""
    try:
        import torch_tensorrt
        return True
    except ImportError:
        return False


def optimize_clearvoice(cv, opt_level=OptLevel.TENSORRT):
    """Apply optimizations to ClearVoice model"""
    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, using CPU")
        return cv
    
    try:
        for model_name, model in cv.models.items():
            model.eval()
            model.cuda()
            
            # FP16
            if opt_level >= OptLevel.FP16:
                try:
                    model.half()
                    print(f"      ✓ FP16 enabled")
                except Exception as e:
                    print(f"      ⚠ FP16 failed: {e}")
            
            # torch.compile
            if opt_level == OptLevel.COMPILE:
                try:
                    cv.models[model_name] = torch.compile(
                        model, mode="max-autotune", backend="inductor"
                    )
                    print(f"      ✓ torch.compile enabled")
                except Exception as e:
                    print(f"      ⚠ torch.compile failed: {e}")
            
            # TensorRT
            elif opt_level == OptLevel.TENSORRT:
                if check_tensorrt():
                    try:
                        import torch_tensorrt
                        cv.models[model_name] = torch.compile(
                            model,
                            backend="torch_tensorrt",
                            dynamic=False,
                            options={
                                "enabled_precisions": {torch.float16},
                                "optimization_level": 3,
                                "truncate_long_and_double": True,
                            }
                        )
                        print(f"      ✓ TensorRT enabled")
                    except Exception as e:
                        print(f"      ⚠ TensorRT failed: {e}, using torch.compile")
                        cv.models[model_name] = torch.compile(model, mode="max-autotune")
                else:
                    print(f"      ⚠ TensorRT not installed, using torch.compile")
                    cv.models[model_name] = torch.compile(model, mode="max-autotune")
                    
    except AttributeError:
        if hasattr(cv, 'model'):
            model = cv.model
            model.eval().cuda()
            if opt_level >= OptLevel.FP16:
                model.half()
            if opt_level >= OptLevel.COMPILE:
                cv.model = torch.compile(model, mode="max-autotune")
    
    return cv


# =============================================================================
# Audio Processing Core Functions
# =============================================================================

def compute_rms(audio):
    """Compute RMS energy of audio"""
    return np.sqrt(np.mean(audio.astype(np.float64)**2))


def compute_si_sdr(reference, estimate):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio
    Higher is better (in dB)
    """
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    
    # Zero-mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Compute SI-SDR
    dot = np.dot(reference, estimate)
    s_target = dot * reference / (np.dot(reference, reference) + 1e-8)
    e_noise = estimate - s_target
    
    si_sdr = 10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8) + 1e-8
    )
    
    return si_sdr


def load_audio(path, target_sr=16000):
    """Load and resample audio to target sample rate"""
    import subprocess
    import tempfile
    
    ext = os.path.splitext(path)[1].lower()
    needs_conversion = ext in ['.m4a', '.mp4', '.aac', '.ogg', '.opus', '.webm', '.wma']
    
    if needs_conversion:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-ar', str(target_sr),
                '-ac', '1',
                '-f', 'wav',
                tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            
            audio, sr = sf.read(tmp_path)
            audio = audio.astype(np.float32)
            
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            return audio, target_sr
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        audio, sr = librosa.load(path, sr=None, mono=True)
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio.astype(np.float32), target_sr


def save_audio_energy_matched(path, audio, sr, target_rms=None, headroom_db=-1.0):
    """
    Save audio with energy matching and headroom
    
    Args:
        path: Output file path
        audio: Audio array
        sr: Sample rate
        target_rms: Target RMS to match (None = no matching)
        headroom_db: Peak headroom in dB (default -1dB)
    """
    audio = audio.astype(np.float64)
    
    # Match target energy if specified
    if target_rms is not None and target_rms > 0:
        current_rms = compute_rms(audio)
        if current_rms > 1e-8:
            audio = audio * (target_rms / current_rms)
    
    # Apply headroom (soft limiting)
    peak = np.max(np.abs(audio))
    max_peak = 10 ** (headroom_db / 20)  # e.g., -1dB = 0.89
    
    if peak > max_peak:
        # Soft compression instead of hard clipping
        audio = np.tanh(audio / peak * 1.5) * max_peak
    
    sf.write(path, audio.astype(np.float32), sr, subtype='PCM_16')


# =============================================================================
# Improved Crossfade Merging
# =============================================================================

def crossfade_merge_hann(chunks, overlap_samples):
    """
    Merge audio chunks with Hann window crossfade
    Produces smoother transitions than cosine crossfade
    """
    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    
    # Create symmetric Hann window for overlap
    if overlap_samples > 0:
        window = np.hanning(overlap_samples * 2).astype(np.float32)
        fade_out = window[:overlap_samples]
        fade_in = window[overlap_samples:]
    
    result = chunks[0].copy()
    
    for chunk in chunks[1:]:
        if overlap_samples > 0 and len(result) >= overlap_samples and len(chunk) >= overlap_samples:
            # Blend in overlap region
            result[-overlap_samples:] = (
                result[-overlap_samples:] * fade_out + 
                chunk[:overlap_samples] * fade_in
            )
            # Append rest of chunk
            result = np.concatenate([result, chunk[overlap_samples:]])
        else:
            result = np.concatenate([result, chunk])
    
    return result


# =============================================================================
# Post-Processing for Cleaner Separation
# =============================================================================

def highpass_filter(audio, sr, cutoff_hz=80):
    """Remove low-frequency rumble below cutoff"""
    sos = butter(4, cutoff_hz, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def spectral_gate(audio, sr, threshold_db=-40, n_fft=2048):
    """
    Remove low-energy spectral components (musical noise artifacts)
    
    Args:
        audio: Input audio
        sr: Sample rate
        threshold_db: Threshold below which to attenuate
        n_fft: FFT size
    
    Returns:
        Cleaned audio
    """
    # STFT
    stft = librosa.stft(audio, n_fft=n_fft)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Compute adaptive threshold from magnitude statistics
    mag_db = librosa.amplitude_to_db(magnitude)
    threshold = np.percentile(mag_db, 30)
    threshold = max(threshold, threshold_db)
    
    # Create soft mask
    mask_db = mag_db - threshold
    mask = 1 / (1 + np.exp(-0.5 * mask_db))  # Sigmoid soft mask
    
    # Smooth mask to reduce musical noise
    mask = uniform_filter(mask.astype(np.float32), size=(3, 5))
    mask = np.clip(mask, 0, 1)
    
    # Reconstruct
    stft_clean = magnitude * mask * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, length=len(audio))
    
    return audio_clean.astype(np.float32)


def reduce_crosstalk(target, interference, sr, aggressiveness=0.3):
    """
    Reduce bleed-through from interference signal in target
    Uses spectral subtraction with flooring
    
    Args:
        target: Target speaker audio to clean
        interference: Other speaker's audio (source of bleed)
        sr: Sample rate
        aggressiveness: How much to subtract (0-1)
    
    Returns:
        Cleaned target audio
    """
    n_fft = 2048
    
    # Match lengths
    min_len = min(len(target), len(interference))
    target = target[:min_len]
    interference = interference[:min_len]
    
    # STFT
    target_stft = librosa.stft(target, n_fft=n_fft)
    interf_stft = librosa.stft(interference, n_fft=n_fft)
    
    target_mag = np.abs(target_stft)
    interf_mag = np.abs(interf_stft)
    target_phase = np.angle(target_stft)
    
    # Spectral subtraction with flooring to prevent musical noise
    floor = 0.1 * target_mag  # Keep at least 10% of original
    cleaned_mag = np.maximum(
        target_mag - aggressiveness * interf_mag,
        floor
    )
    
    # Reconstruct
    cleaned_stft = cleaned_mag * np.exp(1j * target_phase)
    cleaned = librosa.istft(cleaned_stft, length=min_len)
    
    return cleaned.astype(np.float32)


def apply_vad_mask(speakers, original, sr, threshold_db=-40):
    """
    Apply voice activity detection mask
    Silent regions in original should be silent in outputs
    
    Args:
        speakers: List of speaker audio arrays
        original: Original mixed audio
        sr: Sample rate
        threshold_db: Energy threshold for speech detection
    
    Returns:
        List of masked speaker arrays
    """
    # Compute frame-level energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    energy = librosa.feature.rms(y=original, frame_length=frame_length, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(energy + 1e-8)
    
    # Adaptive threshold
    noise_floor = np.percentile(energy_db, 10)
    threshold = max(noise_floor + 10, threshold_db)
    
    # Create frame-level mask
    is_speech = energy_db > threshold
    
    # Expand to sample level
    mask = np.repeat(is_speech.astype(np.float32), hop_length)
    mask = np.pad(mask, (0, max(0, len(original) - len(mask))))[:len(original)]
    
    # Smooth transitions (50ms ramps)
    smooth_samples = int(0.05 * sr)
    mask = uniform_filter1d(mask, size=smooth_samples)
    mask = np.clip(mask, 0, 1).astype(np.float32)
    
    # Apply to speakers
    cleaned = []
    for spk in speakers:
        spk_len = len(spk)
        mask_trimmed = mask[:spk_len] if len(mask) >= spk_len else np.pad(mask, (0, spk_len - len(mask)))
        spk_masked = spk * mask_trimmed[:len(spk)]
        cleaned.append(spk_masked)
    
    return cleaned


def clean_separated_speaker(audio, sr, other_speaker=None, 
                           highpass=True, gate=True, reduce_bleed=True):
    """
    Full post-processing pipeline for separated speaker
    
    Args:
        audio: Speaker audio to clean
        sr: Sample rate
        other_speaker: Other speaker's audio for crosstalk reduction
        highpass: Apply high-pass filter
        gate: Apply spectral gating
        reduce_bleed: Reduce crosstalk from other speaker
    
    Returns:
        Cleaned audio
    """
    # 1. High-pass filter (remove rumble below 80Hz)
    if highpass:
        audio = highpass_filter(audio, sr, cutoff_hz=80)
    
    # 2. Spectral gating to remove artifacts
    if gate:
        audio = spectral_gate(audio, sr, threshold_db=-45)
    
    # 3. Reduce bleed-through from other speaker
    if reduce_bleed and other_speaker is not None:
        audio = reduce_crosstalk(audio, other_speaker, sr, aggressiveness=0.25)
    
    return audio


# =============================================================================
# Adaptive Chunk Splitting
# =============================================================================

def find_optimal_split_points(audio, sr, target_chunk_sec=30, tolerance_sec=3):
    """
    Find chunk boundaries at natural speech pauses
    Avoids splitting mid-word/sentence
    
    Args:
        audio: Full audio array
        sr: Sample rate
        target_chunk_sec: Target chunk duration
        tolerance_sec: How far from target to search for pauses
    
    Returns:
        List of sample indices for chunk boundaries
    """
    # Compute frame energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find low-energy regions (pauses)
    threshold = np.percentile(energy, 25)
    is_pause = energy < threshold
    
    # Target chunk positions in samples
    target_samples = int(target_chunk_sec * sr)
    tolerance_samples = int(tolerance_sec * sr)
    
    split_points = [0]
    current_pos = 0
    
    while current_pos + target_samples < len(audio):
        # Search window around target position
        search_start = current_pos + target_samples - tolerance_samples
        search_end = current_pos + target_samples + tolerance_samples
        
        # Convert to frames
        frame_start = max(0, int(search_start / hop_length))
        frame_end = min(int(search_end / hop_length), len(is_pause))
        
        # Find pauses in search window
        pause_frames = np.where(is_pause[frame_start:frame_end])[0]
        
        if len(pause_frames) > 0:
            # Use middle of the pause region
            best_frame = frame_start + pause_frames[len(pause_frames) // 2]
            split_sample = best_frame * hop_length
        else:
            # No pause found, use target position
            split_sample = current_pos + target_samples
        
        split_points.append(int(split_sample))
        current_pos = int(split_sample)
    
    split_points.append(len(audio))
    
    return split_points


# =============================================================================
# Quality Metrics
# =============================================================================

def compute_quality_metrics_extended(original, speakers):
    """
    Compute comprehensive separation quality metrics
    
    Returns:
        dict with correlation, energy_ratio, si_sdr, energy_preservation
    """
    metrics = {}
    
    if len(speakers) < 2:
        return metrics
    
    spk1, spk2 = speakers[0], speakers[1]
    
    # Speaker correlation (lower = better separation)
    min_len = min(len(spk1), len(spk2))
    correlation = np.corrcoef(spk1[:min_len], spk2[:min_len])[0, 1]
    metrics['correlation'] = abs(correlation) if not np.isnan(correlation) else 1.0
    
    # Energy ratio between speakers
    energy1 = np.sum(spk1**2)
    energy2 = np.sum(spk2**2)
    metrics['energy_ratio'] = min(energy1, energy2) / max(energy1, energy2) if max(energy1, energy2) > 0 else 0
    
    # Energy preservation (combined output vs original)
    original_rms = compute_rms(original)
    combined = spk1[:len(original)] + spk2[:len(original)]
    combined_rms = compute_rms(combined)
    metrics['energy_preservation'] = combined_rms / original_rms if original_rms > 0 else 0
    
    # Individual speaker RMS
    metrics['spk1_rms'] = compute_rms(spk1)
    metrics['spk2_rms'] = compute_rms(spk2)
    metrics['original_rms'] = original_rms
    
    # SI-SDR of reconstruction (how well spk1+spk2 matches original)
    if len(combined) == len(original):
        metrics['reconstruction_si_sdr'] = compute_si_sdr(original, combined)
    
    # Quality rating based on correlation
    if metrics['correlation'] < 0.1:
        metrics['quality'] = "Excellent"
    elif metrics['correlation'] < 0.2:
        metrics['quality'] = "Very Good"
    elif metrics['correlation'] < 0.3:
        metrics['quality'] = "Good"
    elif metrics['correlation'] < 0.5:
        metrics['quality'] = "Moderate"
    else:
        metrics['quality'] = "Poor"
    
    return metrics


# =============================================================================
# Speaker Tracking
# =============================================================================

def track_speaker_consistency(prev_speakers, curr_speakers, sr=16000):
    """
    Track speaker identity across chunks to prevent speaker swapping
    Uses correlation on overlap region to determine best matching
    
    Returns:
        curr_speakers reordered to match prev_speakers
    """
    if prev_speakers is None or len(prev_speakers) < 2 or len(curr_speakers) < 2:
        return curr_speakers
    
    # Use last 0.5s of previous and first 0.5s of current for matching
    match_samples = min(sr // 2, len(prev_speakers[0]), len(curr_speakers[0]))
    
    if match_samples < sr // 10:  # Too short to match reliably
        return curr_speakers
    
    prev_end_0 = prev_speakers[0][-match_samples:]
    prev_end_1 = prev_speakers[1][-match_samples:]
    curr_start_0 = curr_speakers[0][:match_samples]
    curr_start_1 = curr_speakers[1][:match_samples]
    
    # Compute correlations for both orderings
    try:
        corr_same = (
            np.corrcoef(prev_end_0, curr_start_0)[0, 1] +
            np.corrcoef(prev_end_1, curr_start_1)[0, 1]
        )
        corr_swap = (
            np.corrcoef(prev_end_0, curr_start_1)[0, 1] +
            np.corrcoef(prev_end_1, curr_start_0)[0, 1]
        )
    except:
        return curr_speakers
    
    # Handle NaN correlations
    if np.isnan(corr_same):
        corr_same = 0
    if np.isnan(corr_swap):
        corr_swap = 0
    
    # Swap if needed
    if corr_swap > corr_same:
        return [curr_speakers[1], curr_speakers[0]]
    return curr_speakers


# =============================================================================
# Speaker Separator Class - V2
# =============================================================================

class SpeakerSeparatorV2:
    """
    Optimized speaker separator using ClearerVoice MossFormer2
    With enhanced post-processing for cleaner separation
    """
    
    def __init__(self, opt_level=OptLevel.TENSORRT):
        self.opt_level = opt_level
        self.separator = None
        self.enhancer = None
        self.sample_rate = 16000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_separator(self):
        """Load and optimize separation model"""
        if self.separator is None:
            print("📦 Loading MossFormer2_SS_16K (separation)...")
            self.separator = ClearVoice(
                task='speech_separation',
                model_names=['MossFormer2_SS_16K']
            )
            print(f"   ⚡ Applying {OPT_NAMES[self.opt_level]} optimization...")
            self.separator = optimize_clearvoice(self.separator, self.opt_level)
            self._warmup(self.separator)
        return self.separator
    
    def load_enhancer(self):
        """Load and optimize enhancement model"""
        if self.enhancer is None:
            print("📦 Loading MossFormer2_SE_48K (enhancement)...")
            self.enhancer = ClearVoice(
                task='speech_enhancement',
                model_names=['MossFormer2_SE_48K']
            )
            print(f"   ⚡ Applying {OPT_NAMES[self.opt_level]} optimization...")
            self.enhancer = optimize_clearvoice(self.enhancer, self.opt_level)
            self._warmup(self.enhancer, sr=48000)
        return self.enhancer
    
    def _warmup(self, cv, sr=16000, runs=3):
        """Warmup model for JIT compilation"""
        print("   🔥 Warmup...")
        warmup_audio = np.random.randn(sr).astype(np.float32) * 0.01
        warmup_path = '/tmp/warmup_sep.wav'
        sf.write(warmup_path, warmup_audio, sr)
        
        try:
            for _ in range(runs):
                cv(input_path=warmup_path, online_write=False)
        except:
            pass
        finally:
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("   ✓ Ready")
    
    def _separate_chunk(self, audio_chunk, chunk_rms, temp_dir='/tmp'):
        """
        Separate a single audio chunk with energy preservation
        
        Args:
            audio_chunk: Input audio array
            chunk_rms: Original RMS of this chunk (for energy matching)
            temp_dir: Temp directory for processing
            
        Returns:
            List of 2 speaker arrays, energy-matched
        """
        import tempfile
        
        chunk_id = id(audio_chunk)
        temp_input = os.path.join(temp_dir, f'chunk_{chunk_id}.wav')
        
        sf.write(temp_input, audio_chunk, self.sample_rate)
        
        try:
            with tempfile.TemporaryDirectory() as temp_output:
                self.separator(
                    input_path=temp_input,
                    online_write=True,
                    output_path=temp_output
                )
                
                speakers = []
                
                # Walk through all subdirectories to find wav files
                for root, dirs, files in os.walk(temp_output):
                    wav_files = sorted([f for f in files if f.endswith('.wav')])
                    for f in wav_files:
                        audio, _ = sf.read(os.path.join(root, f))
                        speakers.append(audio.astype(np.float32))
                
                # If no files found in subdirs, check root
                if not speakers:
                    wav_files = sorted([f for f in os.listdir(temp_output) if f.endswith('.wav')])
                    for f in wav_files:
                        filepath = os.path.join(temp_output, f)
                        if os.path.isfile(filepath):
                            audio, _ = sf.read(filepath)
                            speakers.append(audio.astype(np.float32))
                
                # Ensure we have 2 speakers
                while len(speakers) < 2:
                    speakers.append(np.zeros(len(audio_chunk), dtype=np.float32))
                
                # Match chunk length
                for i in range(len(speakers)):
                    if len(speakers[i]) != len(audio_chunk):
                        if len(speakers[i]) > len(audio_chunk):
                            speakers[i] = speakers[i][:len(audio_chunk)]
                        else:
                            speakers[i] = np.pad(speakers[i], (0, len(audio_chunk) - len(speakers[i])))
                
                # Energy preservation per chunk
                combined_rms = compute_rms(speakers[0] + speakers[1])
                
                if combined_rms > 1e-8 and chunk_rms > 1e-8:
                    scale_factor = chunk_rms / combined_rms
                    speakers[0] = speakers[0] * scale_factor
                    speakers[1] = speakers[1] * scale_factor
                
                return speakers[:2]
            
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _separate_chunk_with_refinement(self, audio_chunk, chunk_rms, max_passes=2):
        """
        Multi-pass separation for difficult cases
        Re-separates if initial separation has high correlation
        """
        speakers = self._separate_chunk(audio_chunk, chunk_rms)
        
        if max_passes <= 1:
            return speakers
        
        # Check separation quality
        min_len = min(len(speakers[0]), len(speakers[1]))
        try:
            corr = abs(np.corrcoef(speakers[0][:min_len], speakers[1][:min_len])[0, 1])
        except:
            corr = 0
        
        if corr > 0.35:
            # Poor separation - try refining each speaker
            refined = []
            for spk in speakers:
                spk_rms = compute_rms(spk)
                if spk_rms < 1e-8:
                    refined.append(spk)
                    continue
                    
                try:
                    re_sep = self._separate_chunk(spk, spk_rms)
                    # Take the more energetic output (the "real" speaker)
                    if compute_rms(re_sep[0]) > compute_rms(re_sep[1]):
                        refined.append(re_sep[0])
                    else:
                        refined.append(re_sep[1])
                except:
                    refined.append(spk)
            
            # Check if refinement helped
            try:
                new_corr = abs(np.corrcoef(refined[0][:min_len], refined[1][:min_len])[0, 1])
                if new_corr < corr:
                    speakers = refined
            except:
                pass
        
        return speakers
    
    def separate(self, input_path, output_dir, chunk_sec=30, overlap_sec=5,
                 enhance_first=False, output_sr=16000, preserve_energy=True,
                 post_process=True, adaptive_chunks=True, multi_pass=True):
        """
        Separate speakers from audio file
        
        Args:
            input_path: Input audio file path
            output_dir: Output directory
            chunk_sec: Chunk size in seconds for processing
            overlap_sec: Overlap between chunks in seconds (default: 5)
            enhance_first: Apply enhancement before separation
            output_sr: Output sample rate
            preserve_energy: Match output energy to input
            post_process: Apply post-processing (highpass, gating, crosstalk)
            adaptive_chunks: Find chunk boundaries at speech pauses
            multi_pass: Use multi-pass refinement for difficult separations
        
        Returns:
            List of output file paths
        """
        print(f"\n{'='*70}")
        print("🎤 Speaker Separation Pipeline V2 (QUALITY FOCUSED)")
        print(f"   Optimization: {OPT_NAMES[self.opt_level]}")
        print(f"   Energy preservation: {'ON' if preserve_energy else 'OFF'}")
        print(f"   Post-processing: {'ON' if post_process else 'OFF'}")
        print(f"   Adaptive chunks: {'ON' if adaptive_chunks else 'OFF'}")
        print(f"   Multi-pass refinement: {'ON' if multi_pass else 'OFF'}")
        print(f"{'='*70}\n")
        
        # GPU info
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🖥️  GPU: {gpu} ({mem:.1f} GB)")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        print(f"\n📂 Loading: {input_path}")
        audio, sr = load_audio(input_path, target_sr=self.sample_rate)
        duration = len(audio) / sr
        
        # Store original characteristics for energy matching
        original_rms = compute_rms(audio)
        original_peak = np.max(np.abs(audio))
        
        print(f"   Duration: {duration:.1f}s | Sample rate: {sr}Hz")
        print(f"   Original RMS: {original_rms:.4f} | Peak: {original_peak:.4f}")
        
        # Optional enhancement
        if enhance_first:
            print("\n🔧 Step 1: Enhancing audio...")
            self.load_enhancer()
            
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                self.enhancer(
                    input_path=input_path,
                    online_write=True,
                    output_path=temp_dir
                )
                
                enhanced_files = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]
                
                if enhanced_files:
                    temp_enhanced = os.path.join(temp_dir, enhanced_files[0])
                    audio, sr = load_audio(temp_enhanced, target_sr=self.sample_rate)
                    print(f"   ✓ Enhancement complete")
                else:
                    print(f"   ⚠ Enhancement produced no output, using original audio")
        
        # Load separator
        self.load_separator()
        
        # Determine chunk boundaries
        if adaptive_chunks and duration > chunk_sec:
            print(f"\n🔍 Finding optimal chunk boundaries...")
            split_points = find_optimal_split_points(audio, sr, chunk_sec, tolerance_sec=3)
            chunks = []
            chunk_rms_values = []
            for i in range(len(split_points) - 1):
                chunk = audio[split_points[i]:split_points[i+1]]
                chunks.append(chunk)
                chunk_rms_values.append(compute_rms(chunk))
            print(f"   Found {len(chunks)} chunks at natural boundaries")
        else:
            # Fixed chunking with overlap
            chunk_samples = chunk_sec * sr
            overlap_samples = overlap_sec * sr
            step_samples = chunk_samples - overlap_samples
            
            num_chunks = max(1, int(np.ceil((len(audio) - overlap_samples) / step_samples)))
            
            chunks = []
            chunk_rms_values = []
            for i in range(num_chunks):
                start = i * step_samples
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                
                # Pad last chunk if too short
                if len(chunk) < chunk_samples // 2:
                    chunk = np.pad(chunk, (0, chunk_samples // 2 - len(chunk)))
                
                chunks.append(chunk)
                chunk_rms_values.append(compute_rms(chunk))
        
        print(f"\n📊 Processing: {len(chunks)} chunks ({chunk_sec}s target, {overlap_sec}s overlap)")
        
        # Process chunks with speaker tracking
        speaker1_chunks = []
        speaker2_chunks = []
        prev_speakers = None
        
        start_time = time.time()
        
        for i, (chunk, chunk_rms) in enumerate(tqdm(
            zip(chunks, chunk_rms_values), 
            desc="Separating", 
            unit="chunk",
            total=len(chunks)
        )):
            try:
                if multi_pass:
                    speakers = self._separate_chunk_with_refinement(chunk, chunk_rms)
                else:
                    speakers = self._separate_chunk(chunk, chunk_rms)
                
                # Track speaker consistency across chunks
                speakers = track_speaker_consistency(prev_speakers, speakers, sr)
                prev_speakers = speakers
                
                speaker1_chunks.append(speakers[0])
                speaker2_chunks.append(speakers[1])
                
            except Exception as e:
                tqdm.write(f"⚠ Chunk {i+1} failed: {e}")
                speaker1_chunks.append(np.zeros(len(chunk), dtype=np.float32))
                speaker2_chunks.append(np.zeros(len(chunk), dtype=np.float32))
        
        elapsed = time.time() - start_time
        
        # Merge with improved crossfade
        print("\n🔗 Merging chunks with Hann window crossfade...")
        crossfade_samples = int(overlap_sec * sr * 0.5)
        
        speaker1 = crossfade_merge_hann(speaker1_chunks, crossfade_samples)
        speaker2 = crossfade_merge_hann(speaker2_chunks, crossfade_samples)
        
        # Trim to original length
        speaker1 = speaker1[:len(audio)]
        speaker2 = speaker2[:len(audio)]
        
        # Post-processing for cleaner separation
        if post_process:
            print("\n🧹 Post-processing for cleaner separation...")
            
            # Step 1: Clean each speaker
            print("   • High-pass filtering (removing rumble)...")
            speaker1 = highpass_filter(speaker1, sr, cutoff_hz=80)
            speaker2 = highpass_filter(speaker2, sr, cutoff_hz=80)
            
            print("   • Spectral gating (removing artifacts)...")
            speaker1 = spectral_gate(speaker1, sr, threshold_db=-45)
            speaker2 = spectral_gate(speaker2, sr, threshold_db=-45)
            
            print("   • Reducing crosstalk between speakers...")
            # Store copies before crosstalk reduction
            spk1_before = speaker1.copy()
            spk2_before = speaker2.copy()
            speaker1 = reduce_crosstalk(speaker1, spk2_before, sr, aggressiveness=0.25)
            speaker2 = reduce_crosstalk(speaker2, spk1_before, sr, aggressiveness=0.25)
            
            print("   • Applying VAD mask...")
            [speaker1, speaker2] = apply_vad_mask([speaker1, speaker2], audio, sr)
            
            print("   ✓ Post-processing complete")
        
        # Final energy matching
        if preserve_energy:
            print("\n⚖️  Matching output energy to input...")
            combined_rms = compute_rms(speaker1 + speaker2)
            
            if combined_rms > 1e-8:
                final_scale = original_rms / combined_rms
                speaker1 = speaker1 * final_scale
                speaker2 = speaker2 * final_scale
                
                new_combined_rms = compute_rms(speaker1 + speaker2)
                print(f"   Original RMS: {original_rms:.4f}")
                print(f"   Output RMS: {new_combined_rms:.4f}")
                print(f"   Energy ratio: {new_combined_rms/original_rms:.3f}x")
        
        # Resample output if needed
        if output_sr != sr:
            print(f"\n⬆️  Resampling to {output_sr}Hz...")
            speaker1 = librosa.resample(speaker1, orig_sr=sr, target_sr=output_sr)
            speaker2 = librosa.resample(speaker2, orig_sr=sr, target_sr=output_sr)
        
        # Save outputs
        print("\n💾 Saving results...")
        output_files = []
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for i, speaker_audio in enumerate([speaker1, speaker2], 1):
            filename = f"{base_name}_speaker{i}.wav"
            filepath = os.path.join(output_dir, filename)
            
            save_audio_energy_matched(
                filepath, 
                speaker_audio, 
                output_sr,
                target_rms=None,
                headroom_db=-1.0
            )
            output_files.append(filepath)
            
            spk_rms = compute_rms(speaker_audio)
            print(f"   ✓ {filename} ({len(speaker_audio)/output_sr:.1f}s, RMS: {spk_rms:.4f})")
        
        # Quality metrics
        metrics = compute_quality_metrics_extended(audio, [speaker1, speaker2])
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                      for k, v in metrics.items()}, f, indent=2)
        
        # Summary
        rtf = elapsed / duration
        
        print(f"\n{'='*70}")
        print("✅ SEPARATION COMPLETE")
        print(f"{'='*70}")
        
        print(f"\n📈 Performance:")
        print(f"   • Processing time: {elapsed:.1f}s")
        print(f"   • Real-time factor: {rtf:.2f}x {'(faster than real-time!)' if rtf < 1 else ''}")
        print(f"   • Speed: {duration/elapsed:.1f}x real-time")
        
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"   • Peak GPU memory: {max_mem:.2f} GB")
        
        print(f"\n📊 Quality Metrics:")
        print(f"   • Speaker correlation: {metrics.get('correlation', 0):.4f} (lower = better)")
        print(f"   • Energy ratio: {metrics.get('energy_ratio', 0):.3f}")
        print(f"   • Energy preservation: {metrics.get('energy_preservation', 0):.3f}x (target: 1.0)")
        if 'reconstruction_si_sdr' in metrics:
            print(f"   • Reconstruction SI-SDR: {metrics['reconstruction_si_sdr']:.2f} dB")
        print(f"   • Rating: {metrics.get('quality', 'N/A')}")
        
        print(f"\n📁 Output: {output_dir}/")
        for f in output_files:
            print(f"   • {os.path.basename(f)}")
        print(f"   • {base_name}_metrics.json")
        print()
        
        return output_files


def get_audio_duration(path):
    """Get audio duration"""
    import subprocess
    
    ext = os.path.splitext(path)[1].lower()
    needs_ffprobe = ext in ['.m4a', '.mp4', '.aac', '.ogg', '.opus', '.webm', '.wma']
    
    if needs_ffprobe:
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except:
            pass
        return 0
    else:
        try:
            info = sf.info(path)
            return info.duration
        except:
            return 0


def batch_process(input_dir, output_dir, separator, **kwargs):
    """Process all audio files in a directory"""
    
    audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.mp4', '.aac', '.opus', '.webm', '.wma'}
    files = [f for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in audio_exts]
    
    if not files:
        print(f"❌ No audio files found in: {input_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"📁 Batch Processing: {len(files)} files")
    print(f"{'='*70}\n")
    
    total_start = time.time()
    total_duration = 0
    results = {'success': 0, 'failed': 0}
    
    for i, filename in enumerate(files, 1):
        print(f"\n{'─'*70}")
        print(f"[{i}/{len(files)}] {filename}")
        print(f"{'─'*70}")
        
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(output_dir, base_name)
        
        try:
            duration = get_audio_duration(input_path)
            total_duration += duration
            
            separator.separate(input_path, file_output_dir, **kwargs)
            results['success'] += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results['failed'] += 1
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print("📊 BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"   ✓ Success: {results['success']}/{len(files)}")
    if results['failed'] > 0:
        print(f"   ✗ Failed: {results['failed']}/{len(files)}")
    print(f"   • Total duration: {total_duration/60:.1f} min")
    print(f"   • Total time: {total_elapsed/60:.1f} min")
    if total_elapsed > 0:
        print(f"   • Average speed: {total_duration/total_elapsed:.1f}x real-time")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Speaker Separation V2 - Quality Focused',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V2 Improvements:
  - Longer overlap (5s) with Hann window crossfade
  - Spectral gating to remove musical noise artifacts
  - Crosstalk reduction between speakers
  - VAD-based silence masking  
  - Adaptive chunk boundaries at speech pauses
  - Multi-pass refinement for difficult separations
  - High-pass filtering to remove rumble

Optimization Levels (--opt):
  0  Base PyTorch (slowest)
  1  FP16 (~1.5-2x faster)
  2  torch.compile (~2x faster)
  3  TensorRT (~3-5x faster) [DEFAULT]

Examples:
  # Full quality processing (all features enabled)
  python separate_tensorrt_v2.py -i audio.wav -o results/
  
  # Fast mode (disable post-processing)
  python separate_tensorrt_v2.py -i audio.wav -o results/ --no-post-process
  
  # With pre-enhancement for noisy audio
  python separate_tensorrt_v2.py -i audio.wav -o results/ --enhance-first
  
  # Batch processing
  python separate_tensorrt_v2.py --input-dir ./audio_files/ -o results/
"""
    )
    
    parser.add_argument('--input', '-i', help='Input audio file')
    parser.add_argument('--input-dir', help='Input directory for batch processing')
    parser.add_argument('--output', '-o', default='output/', help='Output directory')
    parser.add_argument('--opt', type=int, default=3, choices=[0, 1, 2, 3],
                        help='Optimization level (default: 3=TensorRT)')
    parser.add_argument('--chunk-sec', type=int, default=30,
                        help='Chunk size in seconds (default: 30)')
    parser.add_argument('--overlap-sec', type=int, default=5,
                        help='Overlap between chunks in seconds (default: 5)')
    parser.add_argument('--enhance-first', action='store_true',
                        help='Enhance audio before separation')
    parser.add_argument('--output-sr', type=int, default=16000,
                        help='Output sample rate (default: 16000)')
    parser.add_argument('--no-preserve-energy', action='store_true',
                        help='Disable energy preservation')
    parser.add_argument('--no-post-process', action='store_true',
                        help='Disable post-processing (faster but lower quality)')
    parser.add_argument('--no-adaptive-chunks', action='store_true',
                        help='Disable adaptive chunk boundaries')
    parser.add_argument('--no-multi-pass', action='store_true',
                        help='Disable multi-pass refinement')
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir required")
    
    # Create separator
    separator = SpeakerSeparatorV2(opt_level=args.opt)
    
    # Processing kwargs
    kwargs = {
        'chunk_sec': args.chunk_sec,
        'overlap_sec': args.overlap_sec,
        'enhance_first': args.enhance_first,
        'output_sr': args.output_sr,
        'preserve_energy': not args.no_preserve_energy,
        'post_process': not args.no_post_process,
        'adaptive_chunks': not args.no_adaptive_chunks,
        'multi_pass': not args.no_multi_pass,
    }
    
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"❌ Directory not found: {args.input_dir}")
            sys.exit(1)
        batch_process(args.input_dir, args.output, separator, **kwargs)
    else:
        if not os.path.exists(args.input):
            print(f"❌ File not found: {args.input}")
            sys.exit(1)
        separator.separate(args.input, args.output, **kwargs)


if __name__ == '__main__':
    main()
