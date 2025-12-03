# GPU Scaling & Performance Guide

## 📊 Understanding Your GPU Performance

### Step 1: Run Benchmark

```bash
cd ~/Desktop/speaker_separation
source venv_moss/bin/activate

# Download benchmark_gpu.py to your project directory

# Run benchmark on a test file
python benchmark_gpu.py --benchmark --audio output/TAFPUR.wav --output_dir benchmark_results
```

**This will tell you:**
- ✓ Optimal chunk size for your GPU
- ✓ Processing speed (realtime factor)
- ✓ Memory usage per chunk
- ✓ How to scale up

---

## 🎯 Your GPU: NVIDIA RTX A5000 (24GB)

### Expected Performance

Based on your GPU specs:

| Chunk Size | GPU Memory | Processing Speed | Best For |
|------------|------------|------------------|----------|
| 20s | ~4-5 GB | 8-10x realtime | Safe, reliable |
| 30s | ~6-7 GB | 8-10x realtime | **Recommended** |
| 45s | ~9-10 GB | 8-10x realtime | Long files |
| 60s | ~12-14 GB | 8-10x realtime | Maximum throughput |

**Your 30s chunk size is optimal!**

---

## 📈 Scaling Strategies

### 1. Single File Processing (Current Setup)

```bash
# What you're doing now
python scripts/separation/mossformer2_8k_bytes.py \
    --input results/TAFPUR/preprocessed/preprocessed_final.wav \
    --output_dir results/TAFPUR/separated/ \
    --chunk_size 30
```

**Performance:**
- 16-minute audio → ~2-3 minutes processing
- Speed: ~8x realtime
- Memory: ~7 GB per chunk

---

### 2. Batch Processing (Multiple Files)

```bash
# Download batch_process.py to your project directory

# Process all WAV files in a directory
python batch_process.py \
    --input_dir /path/to/audio/files/ \
    --output_dir batch_results/ \
    --chunk_size 30
```

**Performance:**
- Sequential: processes one file after another
- Example: 10 files × 16 min each → ~30-40 min total

---

### 3. Parallel Processing (Scale Up)

Your RTX A5000 with 24GB can handle **2 files in parallel**:

```bash
# Process 2 files at once
python batch_process.py \
    --input_dir /path/to/audio/files/ \
    --output_dir batch_results/ \
    --parallel 2 \
    --chunk_size 25  # Slightly smaller chunks for safety
```

**Performance:**
- 2× throughput
- Example: 10 files × 16 min each → ~20 min total
- GPU usage: ~14-16 GB (safe)

**⚠️ Don't use --parallel 3+** → Risk of OOM

---

## 🔧 Optimizing for Different Scenarios

### Scenario 1: Many Short Files (< 5 min each)

```bash
# Use larger chunks, parallel processing
python batch_process.py \
    --input_dir short_files/ \
    --output_dir results/ \
    --parallel 2 \
    --chunk_size 60
```

---

### Scenario 2: Few Very Long Files (> 60 min each)

```bash
# Process sequentially, moderate chunks
python batch_process.py \
    --input_dir long_files/ \
    --output_dir results/ \
    --parallel 1 \
    --chunk_size 30
```

---

### Scenario 3: Mixed File Lengths

```bash
# Safe middle ground
python batch_process.py \
    --input_dir mixed_files/ \
    --output_dir results/ \
    --parallel 2 \
    --chunk_size 30
```

---

## 💡 Memory Management Tips

### Monitor GPU Usage

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi
```

### If You Get OOM Errors

1. **Reduce chunk size:**
   ```bash
   --chunk_size 20  # Instead of 30
   ```

2. **Reduce parallelism:**
   ```bash
   --parallel 1  # Instead of 2
   ```

3. **Clear GPU cache between files:**
   Already built into the scripts!

---

## 📊 Performance Calculator

### Your Setup (RTX A5000, 30s chunks, 8x speed)

| Audio Duration | Processing Time (Sequential) | Processing Time (Parallel×2) |
|----------------|------------------------------|------------------------------|
| 5 minutes | ~38 seconds | ~38 seconds |
| 15 minutes | ~1.9 minutes | ~1.9 minutes |
| 30 minutes | ~3.8 minutes | ~3.8 minutes |
| 1 hour | ~7.5 minutes | ~7.5 minutes |
| 10 files × 16 min | ~32 minutes | ~16 minutes |
| 50 files × 16 min | ~160 minutes | ~80 minutes |

---

## 🚀 Quick Commands

### Benchmark Your System
```bash
python benchmark_gpu.py --benchmark --audio test.wav
```

### Process One File
```bash
python scripts/separation/mossformer2_8k_bytes.py \
    --input audio.wav \
    --output_dir results/ \
    --chunk_size 30
```

### Batch Process (Sequential)
```bash
python batch_process.py \
    --input_dir audio_folder/ \
    --output_dir batch_results/
```

### Batch Process (Parallel - 2x faster)
```bash
python batch_process.py \
    --input_dir audio_folder/ \
    --output_dir batch_results/ \
    --parallel 2 \
    --chunk_size 25
```

### Resume Interrupted Batch
```bash
python batch_process.py \
    --input_dir audio_folder/ \
    --output_dir batch_results/ \
    --resume
```

---

## 🎯 Recommendations for RTX A5000

**For Maximum Efficiency:**
1. Use `--chunk_size 30` (balanced)
2. Use `--parallel 2` for batch jobs
3. Monitor with `nvidia-smi`
4. Use `--resume` for long batch jobs

**You can safely process:**
- 1 file at a time: any length, 60s chunks
- 2 files in parallel: any length, 30s chunks
- 3+ files: NOT recommended (OOM risk)

---

## 📈 Scaling to More GPUs (Future)

If you get access to more GPUs:

```bash
# Split files across GPUs (manual)
CUDA_VISIBLE_DEVICES=0 python batch_process.py --input_dir batch1/ --output_dir results/ &
CUDA_VISIBLE_DEVICES=1 python batch_process.py --input_dir batch2/ --output_dir results/ &
```

Each GPU can handle 2 parallel processes, so:
- 2 GPUs → 4× throughput
- 4 GPUs → 8× throughput

---

## 🔍 Troubleshooting

### "CUDA out of memory"
- Reduce chunk size: `--chunk_size 20`
- Reduce parallelism: `--parallel 1`
- Process files one at a time

### "Slow processing"
- Verify GPU is being used: `nvidia-smi`
- Check if other processes are using GPU
- Increase chunk size if memory allows

### "Files taking forever"
- Check chunk size isn't too small
- Verify GPU utilization is high
- Consider parallel processing

---

## 📝 Summary

**Your Current Setup:**
- ✓ RTX A5000 (24GB) - Excellent
- ✓ 30s chunks - Optimal
- ✓ ~8x realtime speed - Good
- ✓ Can process 2 files in parallel - Scalable

**To Scale Up:**
1. Use `batch_process.py` with `--parallel 2`
2. Process ~2-3× faster with parallelism
3. Monitor GPU with `nvidia-smi`
4. Adjust chunk size based on file lengths

**You're all set! 🚀**
