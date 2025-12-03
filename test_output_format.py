import numpy as np
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Load model
print("Loading model...")
separation = pipeline(
    Tasks.speech_separation,
    model='damo/speech_mossformer2_separation_temporal_8k',
    device='cuda'
)

# Use your preprocessed file (just first 5 seconds for testing)
import librosa
audio, sr = librosa.load('results/TAFPUR/preprocessed/preprocessed_final.wav', sr=8000, duration=5.0)

# Save test chunk
sf.write('/tmp/test_chunk.wav', audio, 8000)

# Run separation
print("\nRunning separation on test chunk...")
result = separation('/tmp/test_chunk.wav')

# Debug output
print("\n=== RESULT STRUCTURE ===")
print(f"Type: {type(result)}")
print(f"Keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

if isinstance(result, dict):
    for key, value in result.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            for i, item in enumerate(value[:2]):
                print(f"  Item {i} type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"  Item {i} shape: {item.shape}")
                if hasattr(item, '__len__'):
                    print(f"  Item {i} length: {len(item)}")
                # Try to convert to numpy
                try:
                    arr = np.array(item)
                    print(f"  Item {i} as numpy: shape={arr.shape}, dtype={arr.dtype}")
                    print(f"  Item {i} sample values: {arr.flatten()[:5]}")
                except Exception as e:
                    print(f"  Item {i} conversion failed: {e}")

print("\n=== ATTEMPTING TO EXTRACT AUDIO ===")
try:
    if 'output_pcm_list' in result:
        outputs = result['output_pcm_list']
        print(f"Found output_pcm_list with {len(outputs)} items")
        
        for i, output in enumerate(outputs):
            print(f"\nOutput {i}:")
            print(f"  Type: {type(output)}")
            
            # Try different conversion methods
            arr = np.array(output)
            print(f"  Direct np.array: shape={arr.shape}, dtype={arr.dtype}")
            
            if hasattr(output, 'numpy'):
                arr2 = output.numpy()
                print(f"  .numpy() method: shape={arr2.shape}, dtype={arr2.dtype}")
            
            if hasattr(output, 'cpu'):
                arr3 = output.cpu().numpy()
                print(f"  .cpu().numpy(): shape={arr3.shape}, dtype={arr3.dtype}")
                
                # This might be the correct one!
                print(f"  Sample values: {arr3.flatten()[:10]}")
                print(f"  RMS: {np.sqrt(np.mean(arr3**2)):.6f}")
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

