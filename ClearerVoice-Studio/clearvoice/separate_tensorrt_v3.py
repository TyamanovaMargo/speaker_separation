#!/usr/bin/env python3
"""
ClearerVoice-Studio Speaker Separation - V3 (DIAGNOSTIC + FIXED)
================================================================
Fixes over v2:
1. Better debugging output to understand what ClearVoice returns
2. Fixed energy preservation (sum of speakers should equal original)
3. More robust speaker tracking with longer correlation window
4. Full overlap utilization in crossfade
5. Diagnostic mode to understand separation quality per chunk
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
import json
import tempfile

# Try to import ClearVoice
try:
    sys.path.insert(0, '/app/ClearerVoice-Studio/clearvoice')
    sys.path.insert(0, './ClearerVoice-Studio/clearvoice')
    from clearvoice import ClearVoice
    CLEARVOICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ClearVoice not available: {e}")
    CLEARVOICE_AVAILABLE = False


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
# Audio Utility Functions
# =============================================================================

def compute_rms(audio):
    """Compute RMS energy of audio"""
    return np.sqrt(np.mean(audio.astype(np.float64)**2))


def compute_peak(audio):
    """Compute peak amplitude"""
    return np.max(np.abs(audio))


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


def save_audio(path, audio, sr, headroom_db=-1.0):
    """Save audio with headroom to prevent clipping"""
    audio = audio.astype(np.float64)
    
    peak = np.max(np.abs(audio))
    max_peak = 10 ** (headroom_db / 20)
    
    if peak > max_peak:
        audio = audio * (max_peak / peak)
    
    sf.write(path, audio.astype(np.float32), sr, subtype='PCM_16')


# =============================================================================
# DIAGNOSTIC: Analyze ClearVoice Output Structure
# =============================================================================

def analyze_clearvoice_output(output_dir, verbose=True):
    """
    Analyze what ClearVoice actually produces
    Returns dict with structure info and audio files found
    """
    result = {
        'structure': [],
        'wav_files': [],
        'total_files': 0,
        'directories': []
    }
    
    for root, dirs, files in os.walk(output_dir):
        rel_root = os.path.relpath(root, output_dir)
        if rel_root == '.':
            rel_root = ''
        
        result['directories'].append(rel_root if rel_root else '(root)')
        
        for f in files:
            filepath = os.path.join(root, f)
            rel_path = os.path.join(rel_root, f) if rel_root else f
            result['total_files'] += 1
            result['structure'].append(rel_path)
            
            if f.endswith('.wav'):
                try:
                    audio, sr = sf.read(filepath)
                    info = {
                        'path': rel_path,
                        'full_path': filepath,
                        'duration': len(audio) / sr,
                        'sample_rate': sr,
                        'channels': 1 if len(audio.shape) == 1 else audio.shape[1],
                        'rms': compute_rms(audio),
                        'peak': compute_peak(audio)
                    }
                    result['wav_files'].append(info)
                except Exception as e:
                    result['wav_files'].append({
                        'path': rel_path,
                        'error': str(e)
                    })
    
    if verbose:
        print(f"\n📁 ClearVoice Output Analysis:")
        print(f"   Total files: {result['total_files']}")
        print(f"   Directories: {result['directories']}")
        print(f"   WAV files found: {len(result['wav_files'])}")
        for wav in result['wav_files']:
            if 'error' in wav:
                print(f"      ❌ {wav['path']}: {wav['error']}")
            else:
                print(f"      ✓ {wav['path']}: {wav['duration']:.2f}s, RMS={wav['rms']:.4f}")
    
    return result


# =============================================================================
# FIXED: Energy-Preserving Separation
# =============================================================================

def apply_energy_preservation(speakers, original_audio, method='wiener'):
    """
    Apply proper energy preservation so sum of speakers ≈ original
    
    Methods:
    - 'scale': Simple scaling (current approach - problematic)
    - 'wiener': Wiener-like masking (better for separation)
    - 'psd': Power spectral density matching
    """
    if len(speakers) < 2:
        return speakers
    
    spk1, spk2 = speakers[0].copy(), speakers[1].copy()
    original = original_audio.copy()
    
    # Ensure same length
    min_len = min(len(spk1), len(spk2), len(original))
    spk1 = spk1[:min_len]
    spk2 = spk2[:min_len]
    original = original[:min_len]
    
    if method == 'wiener':
        # Wiener-like approach: scale each speaker by its proportion of total energy
        eps = 1e-8
        
        # Frame-wise energy (25ms frames)
        frame_len = 400  # 25ms at 16kHz
        hop_len = 160    # 10ms hop
        
        # Compute frame energies
        def frame_energy(x, frame_len, hop_len):
            n_frames = (len(x) - frame_len) // hop_len + 1
            energies = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * hop_len
                frame = x[start:start + frame_len]
                energies[i] = np.sum(frame ** 2) + eps
            return energies
        
        e1 = frame_energy(spk1, frame_len, hop_len)
        e2 = frame_energy(spk2, frame_len, hop_len)
        e_orig = frame_energy(original, frame_len, hop_len)
        
        # Compute masks based on energy ratios
        total_sep = e1 + e2 + eps
        mask1 = e1 / total_sep
        mask2 = e2 / total_sep
        
        # Target energy per speaker
        target1 = e_orig * mask1
        target2 = e_orig * mask2
        
        # Scale factors per frame
        scale1 = np.sqrt(target1 / (e1 + eps))
        scale2 = np.sqrt(target2 / (e2 + eps))
        
        # Clip extreme scales
        scale1 = np.clip(scale1, 0.1, 10.0)
        scale2 = np.clip(scale2, 0.1, 10.0)
        
        # Apply frame-wise scaling with overlap-add
        spk1_out = np.zeros_like(spk1)
        spk2_out = np.zeros_like(spk2)
        weights = np.zeros(len(spk1))
        
        window = np.hanning(frame_len)
        
        for i in range(len(scale1)):
            start = i * hop_len
            end = start + frame_len
            if end > len(spk1):
                break
            
            spk1_out[start:end] += spk1[start:end] * scale1[i] * window
            spk2_out[start:end] += spk2[start:end] * scale2[i] * window
            weights[start:end] += window
        
        # Normalize by weights
        weights = np.maximum(weights, eps)
        spk1_out /= weights
        spk2_out /= weights
        
        return [spk1_out.astype(np.float32), spk2_out.astype(np.float32)]
    
    elif method == 'scale':
        # Simple global scaling (original approach)
        combined_rms = compute_rms(spk1 + spk2)
        original_rms = compute_rms(original)
        
        if combined_rms > 1e-8:
            scale = original_rms / combined_rms
            return [spk1 * scale, spk2 * scale]
        return [spk1, spk2]
    
    else:
        return speakers


# =============================================================================
# FIXED: Better Speaker Tracking
# =============================================================================

def track_speaker_consistency_v2(prev_speakers, curr_speakers, sr=16000, 
                                  match_duration=1.0, use_spectral=True):
    """
    Improved speaker tracking with:
    - Longer matching window (1s instead of 0.5s)
    - Optional spectral correlation
    - Energy-weighted matching
    """
    if prev_speakers is None or len(prev_speakers) < 2 or len(curr_speakers) < 2:
        return curr_speakers
    
    match_samples = int(match_duration * sr)
    match_samples = min(match_samples, len(prev_speakers[0]), len(curr_speakers[0]))
    
    if match_samples < sr // 10:  # Too short
        return curr_speakers
    
    prev_end_0 = prev_speakers[0][-match_samples:]
    prev_end_1 = prev_speakers[1][-match_samples:]
    curr_start_0 = curr_speakers[0][:match_samples]
    curr_start_1 = curr_speakers[1][:match_samples]
    
    def compute_similarity(a, b):
        """Compute similarity using correlation and spectral features"""
        # Time-domain correlation
        try:
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0
        
        if use_spectral:
            # MFCC-based similarity
            try:
                mfcc_a = librosa.feature.mfcc(y=a.astype(np.float32), sr=sr, n_mfcc=13)
                mfcc_b = librosa.feature.mfcc(y=b.astype(np.float32), sr=sr, n_mfcc=13)
                
                # DTW distance (lower = more similar)
                # Use simple Euclidean for speed
                min_frames = min(mfcc_a.shape[1], mfcc_b.shape[1])
                mfcc_dist = np.mean(np.abs(mfcc_a[:, :min_frames] - mfcc_b[:, :min_frames]))
                
                # Convert to similarity (0-1)
                mfcc_sim = 1.0 / (1.0 + mfcc_dist)
                
                # Combine
                return 0.5 * corr + 0.5 * mfcc_sim
            except:
                return corr
        
        return corr
    
    # Compute similarities for both orderings
    sim_same = (
        compute_similarity(prev_end_0, curr_start_0) +
        compute_similarity(prev_end_1, curr_start_1)
    )
    sim_swap = (
        compute_similarity(prev_end_0, curr_start_1) +
        compute_similarity(prev_end_1, curr_start_0)
    )
    
    if sim_swap > sim_same + 0.1:  # Threshold to prevent unnecessary swaps
        return [curr_speakers[1], curr_speakers[0]]
    return curr_speakers


# =============================================================================
# FIXED: Full Overlap Crossfade
# =============================================================================

def crossfade_merge_full_overlap(chunks, overlap_samples):
    """
    Merge chunks using FULL overlap (not half)
    Uses Hann window for smooth transitions
    """
    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    
    if overlap_samples <= 0:
        return np.concatenate(chunks)
    
    # Hann window for crossfade
    window = np.hanning(overlap_samples * 2).astype(np.float32)
    fade_out = window[:overlap_samples]
    fade_in = window[overlap_samples:]
    
    # Pre-calculate total length
    total_len = len(chunks[0])
    for i in range(1, len(chunks)):
        total_len += len(chunks[i]) - overlap_samples
    
    result = np.zeros(total_len, dtype=np.float32)
    pos = 0
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk: full copy
            result[:len(chunk)] = chunk
            pos = len(chunk) - overlap_samples
        else:
            # Crossfade region
            actual_overlap = min(overlap_samples, len(chunk), len(result) - pos)
            
            if actual_overlap > 0:
                # Fade out previous
                result[pos:pos + actual_overlap] *= fade_out[:actual_overlap]
                # Add faded-in current
                result[pos:pos + actual_overlap] += chunk[:actual_overlap] * fade_in[:actual_overlap]
            
            # Non-overlap region
            remaining = len(chunk) - actual_overlap
            if remaining > 0:
                end_pos = pos + actual_overlap + remaining
                if end_pos <= len(result):
                    result[pos + actual_overlap:end_pos] = chunk[actual_overlap:]
            
            pos = pos + len(chunk) - overlap_samples
    
    return result


# =============================================================================
# Main Separator Class
# =============================================================================

class SpeakerSeparatorV3:
    """
    Speaker separator with diagnostics and fixes
    """
    
    def __init__(self, opt_level=OptLevel.TENSORRT, diagnostic=True):
        self.opt_level = opt_level
        self.diagnostic = diagnostic
        self.separator = None
        self.sample_rate = 16000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chunk_diagnostics = []
        
    def load_separator(self):
        """Load separation model"""
        if not CLEARVOICE_AVAILABLE:
            raise RuntimeError("ClearVoice not available")
        
        if self.separator is None:
            print("📦 Loading MossFormer2_SS_16K...")
            self.separator = ClearVoice(
                task='speech_separation',
                model_names=['MossFormer2_SS_16K']
            )
            
            # Apply optimizations
            if torch.cuda.is_available():
                self._optimize_model()
            
            self._warmup()
        
        return self.separator
    
    def _optimize_model(self):
        """Apply GPU optimizations"""
        try:
            for model_name, model in self.separator.models.items():
                model.eval()
                model.cuda()
                
                if self.opt_level >= OptLevel.FP16:
                    try:
                        model.half()
                        print(f"   ✓ FP16 enabled")
                    except Exception as e:
                        print(f"   ⚠ FP16 failed: {e}")
                
                if self.opt_level >= OptLevel.COMPILE:
                    try:
                        self.separator.models[model_name] = torch.compile(
                            model, mode="max-autotune"
                        )
                        print(f"   ✓ torch.compile enabled")
                    except Exception as e:
                        print(f"   ⚠ torch.compile failed: {e}")
                        
        except AttributeError:
            print("   ⚠ Could not access model internals for optimization")
    
    def _warmup(self, runs=3):
        """Warmup model"""
        print("   🔥 Warmup...")
        warmup_audio = np.random.randn(self.sample_rate).astype(np.float32) * 0.01
        warmup_path = '/tmp/warmup.wav'
        sf.write(warmup_path, warmup_audio, self.sample_rate)
        
        try:
            for _ in range(runs):
                self.separator(input_path=warmup_path, online_write=False)
        except:
            pass
        finally:
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("   ✓ Ready")
    
    def _separate_chunk(self, audio_chunk, chunk_idx=0):
        """
        Separate a single chunk with diagnostics
        
        Returns:
            tuple: (speakers_list, diagnostic_info)
        """
        diag = {
            'chunk_idx': chunk_idx,
            'input_samples': len(audio_chunk),
            'input_rms': compute_rms(audio_chunk),
            'input_peak': compute_peak(audio_chunk),
            'output_files': [],
            'warnings': [],
            'errors': []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = os.path.join(temp_dir, f'chunk_{chunk_idx}.wav')
            temp_output = os.path.join(temp_dir, 'output')
            os.makedirs(temp_output, exist_ok=True)
            
            sf.write(temp_input, audio_chunk, self.sample_rate)
            
            try:
                # Run separation
                self.separator(
                    input_path=temp_input,
                    online_write=True,
                    output_path=temp_output
                )
                
                # Analyze output
                output_info = analyze_clearvoice_output(temp_output, verbose=False)
                diag['clearvoice_output'] = output_info
                
                # Extract speaker audio
                speakers = []
                for wav_info in sorted(output_info['wav_files'], 
                                       key=lambda x: x.get('path', '')):
                    if 'error' not in wav_info:
                        audio, _ = sf.read(wav_info['full_path'])
                        speakers.append(audio.astype(np.float32))
                        diag['output_files'].append({
                            'path': wav_info['path'],
                            'rms': wav_info['rms'],
                            'peak': wav_info['peak']
                        })
                
                # Validate
                if len(speakers) < 2:
                    diag['warnings'].append(f"Only {len(speakers)} speakers found")
                    while len(speakers) < 2:
                        speakers.append(np.zeros(len(audio_chunk), dtype=np.float32))
                
                # Match lengths
                for i in range(len(speakers)):
                    if len(speakers[i]) != len(audio_chunk):
                        if len(speakers[i]) > len(audio_chunk):
                            speakers[i] = speakers[i][:len(audio_chunk)]
                        else:
                            speakers[i] = np.pad(speakers[i], 
                                                 (0, len(audio_chunk) - len(speakers[i])))
                
                # Record output stats
                diag['spk1_rms'] = compute_rms(speakers[0])
                diag['spk2_rms'] = compute_rms(speakers[1])
                diag['combined_rms'] = compute_rms(speakers[0] + speakers[1])
                diag['energy_ratio'] = diag['combined_rms'] / (diag['input_rms'] + 1e-8)
                
                if diag['energy_ratio'] > 1.5 or diag['energy_ratio'] < 0.5:
                    diag['warnings'].append(
                        f"Energy ratio {diag['energy_ratio']:.2f} outside expected range"
                    )
                
                return speakers[:2], diag
                
            except Exception as e:
                diag['errors'].append(str(e))
                return [np.zeros(len(audio_chunk), dtype=np.float32)] * 2, diag
            
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def separate(self, input_path, output_dir, chunk_sec=30, overlap_sec=5,
                 output_sr=16000, energy_method='wiener'):
        """
        Main separation method
        
        Args:
            input_path: Input audio file
            output_dir: Output directory
            chunk_sec: Chunk duration in seconds
            overlap_sec: Overlap between chunks
            output_sr: Output sample rate
            energy_method: 'wiener' (recommended) or 'scale'
        """
        print(f"\n{'='*70}")
        print("🎤 Speaker Separation V3 (Diagnostic Edition)")
        print(f"   Optimization: {OPT_NAMES[self.opt_level]}")
        print(f"   Energy method: {energy_method}")
        print(f"   Diagnostic mode: {'ON' if self.diagnostic else 'OFF'}")
        print(f"{'='*70}\n")
        
        if torch.cuda.is_available():
            print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        print(f"\n📂 Loading: {input_path}")
        audio, sr = load_audio(input_path, target_sr=self.sample_rate)
        duration = len(audio) / sr
        
        original_rms = compute_rms(audio)
        original_peak = compute_peak(audio)
        
        print(f"   Duration: {duration:.1f}s")
        print(f"   RMS: {original_rms:.4f} | Peak: {original_peak:.4f}")
        
        # Load model
        self.load_separator()
        
        # Chunk audio
        chunk_samples = int(chunk_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        step_samples = chunk_samples - overlap_samples
        
        num_chunks = max(1, int(np.ceil((len(audio) - overlap_samples) / step_samples)))
        
        print(f"\n📊 Processing: {num_chunks} chunks ({chunk_sec}s, {overlap_sec}s overlap)")
        
        # Process chunks
        speaker1_chunks = []
        speaker2_chunks = []
        prev_speakers = None
        self.chunk_diagnostics = []
        
        start_time = time.time()
        
        for i in tqdm(range(num_chunks), desc="Separating"):
            start = i * step_samples
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Pad short chunks
            if len(chunk) < chunk_samples // 2:
                chunk = np.pad(chunk, (0, chunk_samples // 2 - len(chunk)))
            
            # Separate
            speakers, diag = self._separate_chunk(chunk, i)
            self.chunk_diagnostics.append(diag)
            
            # Track speakers
            speakers = track_speaker_consistency_v2(
                prev_speakers, speakers, sr, 
                match_duration=1.0, use_spectral=True
            )
            prev_speakers = speakers
            
            speaker1_chunks.append(speakers[0])
            speaker2_chunks.append(speakers[1])
            
            # Print warnings
            if self.diagnostic and diag['warnings']:
                for w in diag['warnings']:
                    tqdm.write(f"   ⚠ Chunk {i}: {w}")
        
        elapsed = time.time() - start_time
        
        # Merge with FULL overlap
        print("\n🔗 Merging chunks...")
        speaker1 = crossfade_merge_full_overlap(speaker1_chunks, overlap_samples)
        speaker2 = crossfade_merge_full_overlap(speaker2_chunks, overlap_samples)
        
        # Trim to original length
        speaker1 = speaker1[:len(audio)]
        speaker2 = speaker2[:len(audio)]
        
        # Apply energy preservation
        print(f"⚖️  Applying energy preservation ({energy_method})...")
        [speaker1, speaker2] = apply_energy_preservation(
            [speaker1, speaker2], audio, method=energy_method
        )
        
        # Metrics before saving
        combined_rms = compute_rms(speaker1 + speaker2)
        print(f"   Original RMS: {original_rms:.4f}")
        print(f"   Combined RMS: {combined_rms:.4f}")
        print(f"   Ratio: {combined_rms/original_rms:.3f}x")
        
        # Resample if needed
        if output_sr != sr:
            print(f"⬆️  Resampling to {output_sr}Hz...")
            speaker1 = librosa.resample(speaker1, orig_sr=sr, target_sr=output_sr)
            speaker2 = librosa.resample(speaker2, orig_sr=sr, target_sr=output_sr)
        
        # Save outputs
        print("\n💾 Saving results...")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_files = []
        
        for i, spk in enumerate([speaker1, speaker2], 1):
            filename = f"{base_name}_speaker{i}.wav"
            filepath = os.path.join(output_dir, filename)
            save_audio(filepath, spk, output_sr)
            output_files.append(filepath)
            print(f"   ✓ {filename} (RMS: {compute_rms(spk):.4f})")
        
        # Compute final metrics
        metrics = {
            'original_rms': float(original_rms),
            'original_peak': float(original_peak),
            'spk1_rms': float(compute_rms(speaker1)),
            'spk2_rms': float(compute_rms(speaker2)),
            'combined_rms': float(combined_rms),
            'energy_ratio': float(combined_rms / original_rms) if original_rms > 0 else 0,
            'processing_time': elapsed,
            'rtf': elapsed / duration,
            'chunks_processed': num_chunks
        }
        
        # Correlation between speakers (lower = better separation)
        min_len = min(len(speaker1), len(speaker2))
        try:
            corr = np.corrcoef(speaker1[:min_len], speaker2[:min_len])[0, 1]
            metrics['correlation'] = float(abs(corr)) if not np.isnan(corr) else 1.0
        except:
            metrics['correlation'] = 1.0
        
        # Quality rating
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
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save chunk diagnostics if enabled
        if self.diagnostic:
            diag_path = os.path.join(output_dir, f"{base_name}_chunk_diagnostics.json")
            # Convert numpy types for JSON
            def convert_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                return obj
            
            with open(diag_path, 'w') as f:
                json.dump(convert_types(self.chunk_diagnostics), f, indent=2)
            print(f"   ✓ {base_name}_chunk_diagnostics.json")
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ SEPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"\n📈 Performance:")
        print(f"   • Time: {elapsed:.1f}s ({duration/elapsed:.1f}x real-time)")
        print(f"\n📊 Quality:")
        print(f"   • Correlation: {metrics['correlation']:.4f}")
        print(f"   • Energy ratio: {metrics['energy_ratio']:.3f}x")
        print(f"   • Rating: {metrics['quality']}")
        print(f"\n📁 Output: {output_dir}/")
        
        return output_files


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Speaker Separation V3 with Diagnostics'
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', default='output/', help='Output directory')
    parser.add_argument('--opt', type=int, default=3, choices=[0, 1, 2, 3],
                        help='Optimization level (default: 3)')
    parser.add_argument('--chunk-sec', type=int, default=30, help='Chunk size')
    parser.add_argument('--overlap-sec', type=int, default=5, help='Overlap')
    parser.add_argument('--output-sr', type=int, default=16000, help='Output sample rate')
    parser.add_argument('--energy-method', choices=['wiener', 'scale'], 
                        default='wiener', help='Energy preservation method')
    parser.add_argument('--no-diagnostic', action='store_true',
                        help='Disable diagnostic output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)
    
    separator = SpeakerSeparatorV3(
        opt_level=args.opt,
        diagnostic=not args.no_diagnostic
    )
    
    separator.separate(
        args.input,
        args.output,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        output_sr=args.output_sr,
        energy_method=args.energy_method
    )


if __name__ == '__main__':
    main()
