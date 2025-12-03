#!/usr/bin/env python3
"""
MossFormer2 GPU Benchmark & Scaling Analysis

This script helps you understand:
1. GPU memory usage per chunk
2. Processing speed
3. Optimal chunk size for your GPU
4. How to scale up (batch processing multiple files)
"""

import argparse
import os
import sys
import numpy as np
import soundfile as sf
import torch
import time
from datetime import datetime
import json

def benchmark_chunk_sizes(test_audio_path, output_dir="benchmark_results"):
    """Test different chunk sizes to find optimal GPU usage"""
    
    print(f"\n{'='*80}")
    print("MOSSFORMER2 GPU BENCHMARK")
    print(f"{'='*80}\n")
    
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import librosa
    
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Total GPU Memory: {gpu_memory_total:.2f} GB\n")
    else:
        print("⚠️  No GPU detected! Running on CPU (slow)")
        print()
    
    # Load test audio
    print("Loading test audio...")
    audio, sr = librosa.load(test_audio_path, sr=8000, mono=True, duration=60.0)  # Test with 60 seconds
    duration = len(audio) / sr
    print(f"  Test duration: {duration:.1f} seconds")
    print(f"  Sample rate: {sr} Hz\n")
    
    # Load model
    print("Loading MossFormer2 model...")
    separation = pipeline(
        Tasks.speech_separation,
        model='damo/speech_mossformer2_separation_temporal_8k',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("  ✓ Model loaded\n")
    
    # Test different chunk sizes
    chunk_sizes_to_test = [10, 15, 20, 30, 45, 60]  # seconds
    
    results = {
        'gpu_name': gpu_name if torch.cuda.is_available() else 'CPU',
        'gpu_memory_total_gb': gpu_memory_total if torch.cuda.is_available() else 0,
        'test_audio_duration': duration,
        'timestamp': datetime.now().isoformat(),
        'chunk_tests': []
    }
    
    print(f"{'='*80}")
    print("TESTING CHUNK SIZES")
    print(f"{'='*80}\n")
    
    for chunk_seconds in chunk_sizes_to_test:
        chunk_samples = int(chunk_seconds * sr)
        
        if chunk_samples > len(audio):
            print(f"⚠️  Skipping {chunk_seconds}s (longer than test audio)")
            continue
        
        print(f"\n--- Testing {chunk_seconds}s chunks ---")
        
        # Extract chunk
        test_chunk = audio[:chunk_samples]
        chunk_path = '/tmp/benchmark_chunk.wav'
        sf.write(chunk_path, test_chunk, sr)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated(0) / 1e9
        
        # Time the processing
        start_time = time.time()
        
        try:
            result = separation(chunk_path)
            
            elapsed_time = time.time() - start_time
            
            # Memory usage
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(0) / 1e9
                memory_peak = torch.cuda.max_memory_allocated(0) / 1e9
                memory_used = memory_peak - memory_before
            else:
                memory_used = 0
                memory_peak = 0
            
            # Calculate metrics
            processing_speed = chunk_seconds / elapsed_time  # realtime factor
            memory_per_second = memory_used / chunk_seconds if chunk_seconds > 0 else 0
            
            print(f"  ✓ Success")
            print(f"    Processing time: {elapsed_time:.2f}s")
            print(f"    Speed: {processing_speed:.2f}x realtime")
            print(f"    GPU memory used: {memory_used:.3f} GB")
            print(f"    GPU memory peak: {memory_peak:.3f} GB")
            print(f"    Memory per second: {memory_per_second:.3f} GB/s")
            
            results['chunk_tests'].append({
                'chunk_size_seconds': chunk_seconds,
                'processing_time_seconds': elapsed_time,
                'speed_realtime_factor': processing_speed,
                'gpu_memory_used_gb': memory_used,
                'gpu_memory_peak_gb': memory_peak,
                'memory_per_second_audio_gb': memory_per_second,
                'success': True
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OUT OF MEMORY")
                print(f"    Chunk size too large for GPU")
                results['chunk_tests'].append({
                    'chunk_size_seconds': chunk_seconds,
                    'success': False,
                    'error': 'OOM'
                })
            else:
                print(f"  ✗ Error: {e}")
                results['chunk_tests'].append({
                    'chunk_size_seconds': chunk_seconds,
                    'success': False,
                    'error': str(e)
                })
        
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save results
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    successful_tests = [t for t in results['chunk_tests'] if t.get('success', False)]
    
    if not successful_tests:
        print("⚠️  No successful tests! GPU may be too small or other issues.")
        return 1
    
    # Find optimal chunk size
    optimal = max(successful_tests, key=lambda x: x['chunk_size_seconds'])
    fastest = max(successful_tests, key=lambda x: x['speed_realtime_factor'])
    
    print(f"✓ Largest chunk that fits in GPU: {optimal['chunk_size_seconds']}s")
    print(f"  Memory used: {optimal['gpu_memory_used_gb']:.2f} GB")
    print(f"  Processing speed: {optimal['speed_realtime_factor']:.2f}x realtime")
    print()
    
    print(f"✓ Fastest processing: {fastest['chunk_size_seconds']}s chunks")
    print(f"  Speed: {fastest['speed_realtime_factor']:.2f}x realtime")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print()
    
    print("1. For Maximum Throughput:")
    print(f"   Use chunk size: {optimal['chunk_size_seconds']}s")
    print(f"   Expected speed: {optimal['speed_realtime_factor']:.1f}x realtime")
    print()
    
    print("2. For Long Files (>30 min):")
    safe_chunk = max(20, optimal['chunk_size_seconds'] - 10)
    print(f"   Use chunk size: {safe_chunk}s (safer for long processing)")
    print()
    
    print("3. Estimated Processing Times:")
    for duration_min in [5, 15, 30, 60]:
        duration_sec = duration_min * 60
        process_time = duration_sec / optimal['speed_realtime_factor']
        print(f"   {duration_min} min audio → ~{process_time/60:.1f} min processing")
    print()
    
    # Scaling recommendations
    print("4. SCALING UP - Batch Processing:")
    print(f"   Your GPU can handle {optimal['chunk_size_seconds']}s chunks")
    print(f"   Memory headroom: {gpu_memory_total - optimal['gpu_memory_peak_gb']:.2f} GB")
    print()
    
    if gpu_memory_total - optimal['gpu_memory_peak_gb'] > 5:
        print("   ✓ You have GOOD memory headroom!")
        print("   → Can safely process multiple files in parallel")
        print("   → Recommended: 2-3 parallel processes")
    elif gpu_memory_total - optimal['gpu_memory_peak_gb'] > 2:
        print("   ⚙️  You have MODERATE memory headroom")
        print("   → Stick with sequential processing")
        print("   → Or try 2 parallel processes carefully")
    else:
        print("   ⚠️  LIMITED memory headroom")
        print("   → Process files one at a time")
        print("   → Consider using smaller chunks for very long files")
    print()
    
    print(f"Results saved to: {results_file}")
    print()
    
    return 0


def estimate_processing_time(audio_file, chunk_size=30):
    """Estimate how long it will take to process a specific file"""
    
    print(f"\n{'='*80}")
    print("PROCESSING TIME ESTIMATOR")
    print(f"{'='*80}\n")
    
    import librosa
    
    # Load audio info (just metadata, not full audio)
    duration = librosa.get_duration(path=audio_file)
    
    print(f"Audio file: {audio_file}")
    print(f"Duration: {duration/60:.1f} minutes ({duration:.1f} seconds)")
    print(f"Chunk size: {chunk_size}s")
    print()
    
    # Use typical GPU speeds (adjust based on benchmark)
    # RTX A5000 typically processes at 6-10x realtime
    speed_estimates = {
        'conservative': 6.0,  # 6x realtime
        'typical': 8.0,       # 8x realtime
        'optimistic': 10.0    # 10x realtime
    }
    
    print("Estimated processing times:")
    for scenario, speed in speed_estimates.items():
        process_time = duration / speed
        print(f"  {scenario.capitalize()}: {process_time/60:.1f} minutes ({process_time:.0f} seconds)")
    print()
    
    # Chunk overhead
    num_chunks = int(np.ceil(duration / chunk_size))
    overhead_per_chunk = 0.5  # seconds overhead per chunk
    total_overhead = num_chunks * overhead_per_chunk
    
    print(f"Number of chunks: {num_chunks}")
    print(f"Estimated overhead: {total_overhead:.0f} seconds")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MossFormer2 GPU usage and get scaling recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (tests different chunk sizes)
  python benchmark_gpu.py --benchmark --audio test_audio.wav
  
  # Just estimate processing time for a specific file
  python benchmark_gpu.py --estimate --audio long_audio.wav --chunk_size 30
  
  # Both
  python benchmark_gpu.py --benchmark --estimate --audio test.wav
        """
    )
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Run GPU benchmark with different chunk sizes')
    parser.add_argument('--estimate', action='store_true',
                       help='Estimate processing time for audio file')
    parser.add_argument('--audio', required=True,
                       help='Audio file to test/estimate')
    parser.add_argument('--chunk_size', type=int, default=30,
                       help='Chunk size for estimation (default: 30)')
    parser.add_argument('--output_dir', default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return 1
    
    if not args.benchmark and not args.estimate:
        print("Error: Use --benchmark and/or --estimate")
        parser.print_help()
        return 1
    
    if args.benchmark:
        result = benchmark_chunk_sizes(args.audio, args.output_dir)
        if result != 0:
            return result
    
    if args.estimate:
        estimate_processing_time(args.audio, args.chunk_size)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
