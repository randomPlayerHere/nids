# Deploying NIDS on Koyeb

This guide walks you through deploying your Network Intrusion Detection System (FastAPI backend) on Koyeb.

## Prerequisites

- Koyeb account (https://app.koyeb.com)
- GitHub account with your project repository
- (Optional) Koyeb CLI installed locally

## Step 1: Prepare Your Project for Deployment

### 1.1 Create a `runtime.txt` (Python version specification)
```
python-3.11
```

### 1.2 Update `requirements.txt`
Koyeb doesn't support conda environments directly. **Convert your conda environment to pip requirements:**

```bash
# Export your conda environment to requirements
conda list --export | grep -v "^#" | grep -v "^@" > requirements_full.txt

# Or generate from pip
pip freeze > requirements.txt
```

**Note:** Keep the minimal version from your current `requirements.txt` and ensure these are included:
```
numpy
pandas
tensorflow
scikit-learn
joblib
fastapi
uvicorn[standard]
python-multipart
```

### 1.3 Create `.gitignore` (if not already present)
```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.env
```

### 1.4 Create a `Procfile`
Koyeb needs to know how to start your app:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

If Procfile isn't detected, specify the start command in Koyeb dashboard.

## Step 2: Prepare Model Files

### Important: Model Size & Storage
Koyeb has limitations on deployment package size. **Large models may not fit in the deployment.**

**Options:**
1. **Recommended: Use Koyeb Volumes or Object Storage**
   - Upload `models/` to AWS S3, Backblaze B2, or similar
   - Modify `app.py` to download models at startup
   
2. **If model files are small (<50MB total):**
   - Commit to GitHub with `git lfs` (Large File Storage)
   - Keep them in the repository

### Example: Download Model from S3 at Startup
```python
# Add to app.py startup
import boto3
import os

@app.on_event("startup")
def startup_download_model():
    s3 = boto3.client('s3')
    os.makedirs('models', exist_ok=True)
    s3.download_file('your-bucket', 'nids_dcnn_model.h5', 'models/nids_dcnn_model.h5')
    s3.download_file('your-bucket', 'cicids_scaler.pkl', 'models/cicids_scaler.pkl')
    load_assets()
```

## Step 3: Deploy via GitHub (Recommended)

### 3.1 Push to GitHub
```bash
git add .
git commit -m "Prepare for Koyeb deployment"
git push origin main
```

### 3.2 Connect to Koyeb
1. Go to https://app.koyeb.com
2. Click **"Create App"** → **"GitHub"**
3. Authorize Koyeb with GitHub
4. Select your repository and branch
5. Choose **"Docker"** or **"Buildpack"** deployment method

### 3.3 Configure Environment

**Service Settings:**
- **Runtime:** Python
- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Port:** `8000` (internal)

**Environment Variables** (if needed):
- Add any sensitive data here (API keys, S3 credentials, etc.)
- Set `MODEL_PATH` and `SCALER_PATH` if using object storage

### 3.4 Deploy
- Click **"Deploy"**
- Wait for the build to complete (~3-5 minutes)
- Your app will be live at `https://<app-name>-<account>.koyeb.app`

## Step 4: Deploy via Docker (Alternative)

### 4.1 Create `Dockerfile`
```dockerfile
# Use official Python runtime as base image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable
ENV PORT=8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Build and Test Locally
```bash
docker build -t nids-app .
docker run -p 8000:8000 nids-app
```

### 4.3 Push to Docker Hub
```bash
docker tag nids-app yourusername/nids-app:latest
docker push yourusername/nids-app:latest
```

### 4.4 Deploy to Koyeb
1. Go to Koyeb dashboard
2. Click **"Create App"** → **"Docker"**
3. Enter image: `yourusername/nids-app:latest`
4. Configure ports and environment variables
5. Deploy

## Step 5: Test Your Deployment

### Test API Endpoint
```bash
# Get API docs
curl https://<your-app>.koyeb.app/docs

# Test health check (if implemented)
curl https://<your-app>.koyeb.app/health

# Test inference with CSV upload
curl -X POST https://<your-app>.koyeb.app/predict \
  -F "file=@test.csv"
```

## Step 6: Optimize for Production

### 6.1 Handle Conda Specifics
If your code requires conda-specific packages:
- **Option A:** Use `conda env export > environment.yml` and recreate locally with pip
- **Option B:** Use a conda-based Docker image (larger, slower builds)

### 6.2 Set Resource Limits
In Koyeb dashboard:
- **Memory:** Start with 512MB (adjust based on model size)
- **CPU:** 500m (millicores) minimum

### 6.3 Enable Auto-scaling
- Set min/max instances for high traffic
- Configure health check endpoint

### 6.4 Monitor Logs
```bash
# View deployment logs in Koyeb dashboard
# Or use Koyeb CLI
koyeb service logs <app-name>
```

## Troubleshooting

### ❌ Build fails with "module not found"
- Check `requirements.txt` has all dependencies
- Remove conda-specific packages (e.g., `mkl`, `nomkl`)

### ❌ TensorFlow import error
- TensorFlow is large (~500MB). May need optimized build:
  ```
  tensorflow-lite  # Lightweight alternative
  # or
  tensorflow-cpu  # CPU-only (smaller than GPU version)
  ```

### ❌ Model files not found
- Upload to object storage
- Download at startup
- Or use smaller model format (ONNX, TFLite)

### ❌ Timeout on first request
- Koyeb may scale down idle instances
- Implement model caching to avoid reloading
- Use `@app.on_event("startup")` to pre-load

### ❌ Out of memory errors
- Reduce batch size in inference
- Use model quantization
- Split into microservices

## Cost Optimization

**Koyeb free tier includes:**
- 2 services (free tier)
- 512MB RAM per service
- 2 vCPU (shared)
- Limited bandwidth

**Upgrade when needed:**
- Production tier starts ~$0.50/hour per instance

## Additional Resources

- [Koyeb Docs](https://docs.koyeb.com)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Python Buildpack](https://docs.koyeb.com/docs/build/buildpacks#python)
