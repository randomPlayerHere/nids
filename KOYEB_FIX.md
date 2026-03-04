# Koyeb Deployment - Memory Optimization Fix

## Issue You Had
The app was getting stuck during inference with memory allocation warnings:
```
Allocation of 19398656 exceeds 10% of free system memory.
```

**Cause:** TensorFlow was trying to allocate too much memory at once on Koyeb's 512MB free tier.

## What We Fixed

### 1. **Reduced Workload Per Request** (`app.py`)
```python
MAX_ROWS_PER_REQUEST = 10000   # Was 30000
PREDICT_BATCH_SIZE = 64        # Was 128
MAX_DOWNLOAD_ROWS = 500        # Was 1500
MAX_DISPLAY_ROWS = 30          # Was 50
```

### 2. **Float32 Explicit Casting**
All arrays now use `.astype(np.float32)` to avoid float64 doubling memory.

### 3. **TFLite Support for ~4x Lower Memory**
The app now auto-detects and prefers TFLite model if available.

---

## Convert Model to TFLite (Recommended)

Run this locally before deploying:

```bash
# Basic conversion (no quantization)
python scripts/convert_to_tflite.py

# FP16 quantization (2x smaller, minimal accuracy loss) - RECOMMENDED
python scripts/convert_to_tflite.py --quantize fp16

# Dynamic quantization (good balance)
python scripts/convert_to_tflite.py --quantize dynamic

# INT8 quantization (4x smaller, may lose accuracy)
python scripts/convert_to_tflite.py --quantize int8
```

This creates `models/nids_dcnn_model.tflite`.

### Expected Size Reduction
| Format | Typical Size | Memory @ Runtime |
|--------|--------------|------------------|
| Keras H5 | ~10 MB | ~400 MB |
| TFLite FP32 | ~3 MB | ~100 MB |
| TFLite FP16 | ~1.5 MB | ~60 MB |
| TFLite INT8 | ~0.8 MB | ~40 MB |

---

## Deployment Steps

### Step 1: Convert Model (Local)
```bash
cd /home/kuper/projects/nids
python scripts/convert_to_tflite.py --quantize fp16
```

### Step 2: Push to GitHub
```bash
git add models/nids_dcnn_model.tflite app.py scripts/
git commit -m "Add TFLite model for low-memory deployment"
git push origin main
```

### Step 3: Redeploy on Koyeb
Koyeb will auto-rebuild. Check health endpoint:
```bash
curl https://<your-app>.koyeb.app/health
# Should show: {"status": "ready", "model_type": "tflite", ...}
```

---

## Additional Koyeb Settings

### Environment Variables (optional)
```
WEB_CONCURRENCY=1
TF_CPP_MIN_LOG_LEVEL=2
```

### Health Check
- Path: `/health`
- Interval: 30s
- Timeout: 10s

---

## Testing Memory Usage Locally

```bash
# Monitor memory during inference
python -c "
import psutil
import os
print(f'Before: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB')

# Load model
from app import load_assets
load_assets()
print(f'After load: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB')
"
```

---

## Troubleshooting

### Still getting 502/504?
1. Reduce `MAX_ROWS_PER_REQUEST` to 5000
2. Use FP16 or INT8 quantized TFLite model
3. Upgrade Koyeb instance to 1GB RAM

### TFLite not loading?
- Ensure `models/nids_dcnn_model.tflite` exists
- Check file was committed to git (not in .gitignore)

### Accuracy dropped after quantization?
- Use FP16 instead of INT8
- Or use no quantization (larger but exact)

---

## Rollback

If things break:
```bash
git revert HEAD
git push origin main
# App will fall back to Keras H5 model
```
