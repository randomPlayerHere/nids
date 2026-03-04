# Koyeb Deployment - Memory Optimization Fix

## Issue You Had
The app was getting stuck during inference with memory allocation warnings:
```
Allocation of 19398656 exceeds 10% of free system memory.
```

**Cause:** TensorFlow was trying to allocate too much memory at once on Koyeb's 512MB free tier.

## What I Fixed

### 1. **Changed to CPU-only TensorFlow** (`requirements.txt`)
- ❌ Old: `tensorflow` (500MB+)
- ✅ New: `tensorflow-cpu` (lighter, no GPU code)

### 2. **Added TensorFlow Memory Optimization** (`app.py`)
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.run_functions_eagerly(False)
```
This tells TensorFlow to grow memory gradually instead of pre-allocating everything.

### 3. **Batch Processing for Predictions** (`app.py`)
- ❌ Old: Predict all rows at once → big memory spike
- ✅ New: Process 128 rows per batch → distributed memory usage

### 4. **Reduced Result Set**
- ❌ Old: Return 100 rows of results
- ✅ New: Return 50 rows (still plenty for analysis)

## Deployment Steps

### Step 1: Push Updated Code
```bash
git add app.py requirements.txt
git commit -m "Optimize for Koyeb memory constraints"
git push origin main
```

### Step 2: Redeploy on Koyeb

**Option A: Auto-redeploy (if GitHub connected)**
- Koyeb will detect the push and auto-rebuild

**Option B: Manual redeploy**
1. Go to https://app.koyeb.com
2. Select your app
3. Click **"Redeploy"** → **"New deployment"**
4. Wait for build to complete

### Step 3: Test the Fix
```bash
curl https://<your-app>.koyeb.app/health
# Should return: {"status": "ready", "expected_features": 84}
```

Upload your test CSV and check if it completes now.

## Additional Optimizations (If Still Slow)

### A. Use Smaller Model Format (ONNX)
```bash
# Convert H5 to ONNX (smaller, faster)
pip install tf2onnx
python -m tf2onnx.convert --saved-model models/nids_dcnn_model.h5 --output_file models/model.onnx

# Then use onnx for inference (much faster on CPU)
```

### B. Increase Koyeb Resources
- Upgrade from free tier (512MB) to paid tier (1GB+)
- Cost: ~$0.50/hour

### C. Reduce CSV File Size
- Limit input to <10,000 rows
- Add validation in frontend to reject large files

## Monitoring

Check logs after redeploy:
```bash
# Via Koyeb CLI
koyeb service logs <app-name>

# Or via dashboard: Apps > Your App > Logs
```

You should see:
```
✅ Model and scaler loaded successfully
```

And when uploading a CSV:
```
INFO: POST /predict HTTP/1.1" 200 OK
```

**Not stuck on "Running ML Model..."?** → Fix worked! ✅

---

## Rollback (If Issues)

If things get worse:
```bash
git revert HEAD
git push origin main
# Koyeb will auto-redeploy previous version
```

## Questions?

- **Stuck still?** Check if `tensorflow-cpu` installed (may need to force rebuild)
- **Timeout?** Increase Koyeb timeout or upgrade resources
- **Slow?** Use ONNX model format or reduce batch size to 64
