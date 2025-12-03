#!/usr/bin/env python3
"""
MASTER PREPROCESSING SCRIPT

Runs all preprocessing steps in sequence or individually.

Usage:
    # Run all steps
    python scripts/preprocess/run_all.py --input audio.wav --output_dir preprocessed/
    
    # Run specific steps
    python scripts/preprocess/run_all.py --input audio.wav --output_dir preprocessed/ --steps 1 2 3
    
    # Skip certain steps
    python scripts/preprocess/run_all.py --input audio.wav --output_dir preprocessed/ --skip 4
"""

import argparse
import subprocess
import os
import sys


STEPS = {
    1: {
        'name': 'Audio Diagnostics',
        'script': '01_audio_diagnostics.py',
        'description': 'Analyze audio quality'
    },
    2: {
        'name': 'Resample',
        'script': '02_resample.py',
        'description': 'Resample to 16kHz'
    },
    3: {
        'name': 'Declipping',
        'script': '03_declip.py',
        'description': 'Repair clipped audio'
    },
    4: {
        'name': 'Hum Removal',
        'script': '04_remove_hum.py',
        'description': 'Remove power line hum'
    },
    5: {
        'name': 'Noise Reduction',
        'script': '05_denoise.py',
        'description': 'Reduce background noise'
    },
    6: {
        'name': 'Normalization',
        'script': '06_normalize.py',
        'description': 'Normalize amplitude'
    }
}


def run_step(step_num: int, input_file: str, output_file: str, script_dir: str, extra_args: list = []):
    """Run a single preprocessing step"""
    
    step_info = STEPS[step_num]
    script_path = os.path.join(script_dir, step_info['script'])
    
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {step_info['name']}")
    print(f"{step_info['description']}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        script_path,
        '--input', input_file,
        '--output', output_file
    ] + extra_args
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n✗ Step {step_num} failed!")
        return False
    
    print(f"\n✓ Step {step_num} complete: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing steps in sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. Audio Diagnostics - Analyze quality
  2. Resample - Convert to 16kHz
  3. Declipping - Repair clipped audio
  4. Hum Removal - Remove power line interference
  5. Noise Reduction - Reduce background noise
  6. Normalization - Normalize amplitude

Examples:
  # Run all steps
  python run_all.py --input audio.wav --output_dir preprocessed/
  
  # Run specific steps only
  python run_all.py --input audio.wav --output_dir preprocessed/ --steps 1 2 5 6
  
  # Skip certain steps
  python run_all.py --input audio.wav --output_dir preprocessed/ --skip 3 4
        """
    )
    
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--steps', nargs='+', type=int, 
                       help='Run only these steps (e.g., --steps 1 2 3)')
    parser.add_argument('--skip', nargs='+', type=int,
                       help='Skip these steps (e.g., --skip 3 4)')
    parser.add_argument('--sr', type=int, default=16000, 
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--denoise_strength', type=float, default=0.8,
                       help='Denoising strength 0.0-1.0 (default: 0.8)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = list(STEPS.keys())
    
    if args.skip:
        steps_to_run = [s for s in steps_to_run if s not in args.skip]
    
    steps_to_run.sort()
    
    print(f"\n{'='*80}")
    print("PREPROCESSING PIPELINE")
    print(f"{'='*80}")
    print(f"\nInput: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Steps to run: {steps_to_run}")
    print()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Track current file
    current_file = args.input
    
    # Run each step
    for step_num in steps_to_run:
        step_info = STEPS[step_num]
        
        # Determine output filename
        if step_num == 1:
            # Step 1 (diagnostics) outputs JSON
            output_file = os.path.join(args.output_dir, 'diagnostics.json')
            extra_args = []
        elif step_num == 2:
            output_file = os.path.join(args.output_dir, 'step2_resampled.wav')
            extra_args = ['--sr', str(args.sr)]
        elif step_num == 3:
            output_file = os.path.join(args.output_dir, 'step3_declipped.wav')
            extra_args = []
        elif step_num == 4:
            output_file = os.path.join(args.output_dir, 'step4_dehum.wav')
            extra_args = []
        elif step_num == 5:
            output_file = os.path.join(args.output_dir, 'step5_denoised.wav')
            extra_args = ['--strength', str(args.denoise_strength)]
        elif step_num == 6:
            output_file = os.path.join(args.output_dir, 'step6_normalized.wav')
            extra_args = []
        
        # Run step
        success = run_step(step_num, current_file, output_file, script_dir, extra_args)
        
        if not success:
            print("\n✗ Pipeline failed!")
            return 1
        
        # Update current file for next step (except for step 1 which outputs JSON)
        if step_num != 1:
            current_file = output_file
    
    # Create final output link
    if 6 in steps_to_run:
        final_output = os.path.join(args.output_dir, 'preprocessed_final.wav')
        import shutil
        shutil.copy(current_file, final_output)
        print(f"\n{'='*80}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"\n✓ Final output: {final_output}")
    
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
