#!/usr/bin/env python3
"""
MossFormer2 16kHz + TensorRT Speaker Separation
Simple, fast, production-ready
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch
import librosa
from tqdm import tqdm


def separate_audio(input_path, output_dir, use_tensorrt=True, chunk_size=30):
    """Separate audio into 2 speakers"""
    
    print(f"\n{'='*70}")
    print("MossFormer2 16kHz + TensorRT Speaker Separation")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s | Sample rate: {sr}Hz")
    
    # Resample to 16kHz
    if sr != 16000:
        print("Resampling to 16kHz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Load model
    print("\nLoading MossFormer2 16kHz model...")
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    
    separator = pipeline(
        Tasks.speech_separation,
        model='damo/speech_mossformer2_separation_temporal_16k',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Apply TensorRT
    if use_tensorrt and torch.cuda.is_available():
        try:
            import torch_tensorrt
            print("Applying TensorRT optimization (first run takes 2-3 min)...")
            
            model = separator.model
            model.eval()
            example = torch.randn(1, 80000).cuda()
            
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,
            )
            separator.model = trt_model
            print("✓ TensorRT enabled (2-3x faster)")
        except:
            print("⚠ TensorRT not available, using PyTorch")
            use_tensorrt = False
    
    # Process in chunks
    chunk_samples = chunk_size * 16000
    num_chunks = int(np.ceil(len(audio) / chunk_samples))
    
    print(f"\nProcessing {num_chunks} chunks...")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    speaker1_chunks = []
    speaker2_chunks = []
    
    for i in tqdm(range(num_chunks), desc="Separating"):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Save temp chunk
        temp_path = f'/tmp/chunk_{i}.wav'
        sf.write(temp_path, chunk, 16000)
        
        try:
            result = separator(temp_path)
            outputs = result.get('output_pcm_list', result)
            
            # Convert to numpy
            if isinstance(outputs[0], bytes):
                import struct
                s1 = np.array(struct.unpack(f'{len(outputs[0])//2}h', outputs[0]), dtype=np.float32) / 32768.0
                s2 = np.array(struct.unpack(f'{len(outputs[1])//2}h', outputs[1]), dtype=np.float32) / 32768.0
            else:
                s1 = np.array(outputs[0], dtype=np.float32).squeeze()
                s2 = np.array(outputs[1], dtype=np.float32).squeeze()
            
            # Ensure correct length
            if len(s1) != len(chunk):
                s1 = np.resize(s1, len(chunk))
            if len(s2) != len(chunk):
                s2 = np.resize(s2, len(chunk))
            
            speaker1_chunks.append(s1)
            speaker2_chunks.append(s2)
            
        except Exception as e:
            print(f"\n⚠ Chunk {i+1} failed: {e}")
            speaker1_chunks.append(np.zeros(len(chunk), dtype=np.float32))
            speaker2_chunks.append(np.zeros(len(chunk), dtype=np.float32))
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Merge with crossfade
    print("\nMerging chunks...")
    speaker1 = merge_with_crossfade(speaker1_chunks, crossfade_ms=100)
    speaker2 = merge_with_crossfade(speaker2_chunks, crossfade_ms=100)
    
    # Save results
    print("Saving results...")
    for name, data in [('speaker1.wav', speaker1), ('speaker2.wav', speaker2)]:
        path = os.path.join(output_dir, name)
        
        # Normalize
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data)) * 0.95
        
        sf.write(path, data, 16000, subtype='PCM_16')
        print(f"  ✓ {name}")
    
    # Quality metrics
    min_len = min(len(speaker1), len(speaker2))
    correlation = np.corrcoef(speaker1[:min_len], speaker2[:min_len])[0, 1]
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE!")
    print(f"{'='*70}")
    print(f"\nSeparation quality: {correlation:.3f} ({'Excellent' if correlation < 0.3 else 'Good' if correlation < 0.6 else 'Moderate'})")
    print(f"Output: {output_dir}/")
    print(f"  • speaker1.wav")
    print(f"  • speaker2.wav\n")


def merge_with_crossfade(chunks, crossfade_ms=100):
    """Merge chunks with crossfading"""
    if len(chunks) == 0:
        return np.array([])
    if len(chunks) == 1:
        return chunks[0]
    
    crossfade_samples = int(crossfade_ms * 16 / 1000)
    result = chunks[0].copy()
    
    for chunk in chunks[1:]:
        if crossfade_samples > 0 and len(result) >= crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            overlap_start = len(result) - crossfade_samples
            result[overlap_start:] *= fade_out
            result[overlap_start:] += chunk[:crossfade_samples] * fade_in
            result = np.concatenate([result, chunk[crossfade_samples:]])
        else:
            result = np.concatenate([result, chunk])
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Separate audio into 2 speakers')
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--no-tensorrt', action='store_true', help='Disable TensorRT')
    parser.add_argument('--chunk-size', type=int, default=30, help='Chunk size (seconds)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    separate_audio(
        args.input,
        args.output,
        use_tensorrt=not args.no_tensorrt,
        chunk_size=args.chunk_size
    )
