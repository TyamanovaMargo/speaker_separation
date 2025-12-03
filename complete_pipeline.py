#!/usr/bin/env python3
"""
COMPLETE PIPELINE: Preprocessing + MossFormer2 Separation

Runs the full pipeline:
1. Preprocess audio (denoise, normalize, etc.)
2. Separate with MossFormer2
3. Save results

Usage:
    python complete_pipeline.py --input audio.wav --output_dir results/
"""

import argparse
import subprocess
import os
import sys


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*80}")
    print(description)
    print(f"{'='*80}\n")
    
    # Use list instead of shell=True to avoid parsing issues
    if isinstance(cmd, str):
        import shlex
        cmd = shlex.split(cmd)
    
    result = subprocess.run(cmd, shell=False)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed")
        return True
    else:
        print(f"\n✗ {description} failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Preprocessing + MossFormer2 separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Audio diagnostics
  2. Preprocessing (resample, denoise, normalize)
  3. MossFormer2 separation
  4. Save separated speakers

Examples:
  # Full pipeline
  python complete_pipeline.py \\
      --input /home/margo/Desktop/separation_voice_model/output/tafdenok.wav \\
      --output_dir results/tafdenok/

  # Skip preprocessing (if already preprocessed)
  python complete_pipeline.py \\
      --input preprocessed.wav \\
      --output_dir results/ \\
      --skip_preprocess
        """
    )
    
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing (use if already preprocessed)')
    parser.add_argument('--denoise_strength', type=float, default=0.8,
                       help='Denoising strength 0.0-1.0 (default: 0.8)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    preprocess_dir = os.path.join(args.output_dir, 'preprocessed')
    separated_dir = os.path.join(args.output_dir, 'separated')
    
    print(f"\n{'='*80}")
    print("COMPLETE PIPELINE: PREPROCESSING + MOSSFORMER2")
    print(f"{'='*80}\n")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Determine input for separation
    if args.skip_preprocess:
        print("⚠️  Skipping preprocessing (using input file directly)")
        separation_input = args.input
    else:
        # Step 1: Preprocessing
        success = run_command(
            [
                sys.executable,
                "scripts/preprocess/run_all.py",
                "--input", args.input,
                "--output_dir", preprocess_dir,
                "--denoise_strength", str(args.denoise_strength)
            ],
            "STEP 1: Preprocessing"
        )
        
        if not success:
            print("\n✗ Pipeline failed at preprocessing stage")
            return 1
        
        # Use preprocessed file for separation
        separation_input = os.path.join(preprocess_dir, 'preprocessed_final.wav')
        
        if not os.path.exists(separation_input):
            print(f"\n✗ Preprocessed file not found: {separation_input}")
            return 1
    
    # Step 2: MossFormer2 Separation
    success = run_command(
        [
            sys.executable,
            "scripts/separation/mossformer2_separate.py",
            "--input", separation_input,
            "--output_dir", separated_dir
        ],
        "STEP 2: MossFormer2 Separation"
    )
    
    if not success:
        print("\n✗ Pipeline failed at separation stage")
        return 1
    
    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}\n")
    
    print("Output structure:")
    print(f"  {args.output_dir}/")
    if not args.skip_preprocess:
        print(f"    ├── preprocessed/")
        print(f"    │   ├── diagnostics.json")
        print(f"    │   ├── step2_resampled.wav")
        print(f"    │   ├── step5_denoised.wav")
        print(f"    │   └── preprocessed_final.wav")
    print(f"    └── separated/")
    print(f"        ├── speaker1.wav")
    print(f"        └── speaker2.wav")
    print()
    
    print("✓ Your separated speakers are in:")
    print(f"  {separated_dir}/speaker1.wav")
    print(f"  {separated_dir}/speaker2.wav")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())