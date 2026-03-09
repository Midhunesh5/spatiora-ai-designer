# RunPod Deployment Guide: Stable Diffusion API (RTX 4090)

## 📋 Architecture Overview

- **Backend**: Flask + Gunicorn (single worker, no debug mode)
- **Base Model**: Hugging Face `runwayml/stable-diffusion-v1-5` (auto-downloads on first startup)
- **Pipelines**: Dual preloaded (interior + floorplan), each with LoRA weights
- **Concurrency**: Semaphore-limited to 2 concurrent GPU inferences
- **GPU**: RTX 4090 (24GB VRAM, no CPU offload needed)
- **Port**: 0.0.0.0:5000 exposed via RunPod HTTP proxy

---

## 🚀 Quick Start: RunPod Deployment

### Step 1: Prepare LoRA Weights

Upload your LoRA files to RunPod persistent volume:

```
models/
├── floor_plan_model/
│   └── lora_weights/
│       └── floor_pytorch_lora_weights.safetensors
└── interior_design/
    └── interior_gen_lora_weights/
        └── pytorch_lora_weights.safetensors
```

Or, add these paths to your `.gitignore` and exclude them from Docker build, then mount a persistent volume at `/app/models`.

### Step 2: Build Docker Image Locally (Optional)

If you want to test the image before deploying:

```bash
docker build -t my-sd-api:latest .
docker run --gpus all -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  my-sd-api:latest
```

### Step 3: Push to RunPod Container Registry

If you have a RunPod account and want to use a private registry:

1. Log in to RunPod registry:
   ```bash
   docker login -u your-runpod-username -p your-runpod-token registry.runpod.io
   ```

2. Build and push:
   ```bash
   docker build -t registry.runpod.io/your-namespace/sd-api:latest .
   docker push registry.runpod.io/your-namespace/sd-api:latest
   ```

### Step 4: Deploy on RunPod UI

1. **Create a GPU Pod**:
   - GPU: RTX 4090
   - Container Image: Use your pushed image OR upload the Dockerfile directly
   - Expose HTTP (on port 5000)

2. **Set Environment Variables** in the RunPod pod settings:
   ```
   FRONTEND_ORIGIN=*                          # For testing; set to your domain later
   HUGGINGFACE_HUB_TOKEN=(your HF token if private models)
   JWT_SECRET_KEY=your-secret-key-here
   MONGO_URI=your-mongodb-connection-string   # (optional, for user auth)
   ```

3. **Mount Persistent Volume** (optional but recommended):
   - Attach a persistent volume to `/app/models` to cache the base model and LoRA weights across pod restarts.
   - This avoids re-downloading the 5GB+ base model on every pod restart.

4. **Configure Network**:
   - Ensure port `5000` is exposed publicly via HTTP.
   - RunPod will assign a proxy URL like: `https://xxxx-xxxx.pods.runpod.io/`

### Step 5: Verify Deployment

Once the pod is running, check the logs:

```
[INFO] --- Starting Model Loading (two pipelines from Hugging Face) ---
[INFO] Downloading base model from Hugging Face...
[INFO] Loading interior pipeline...
[INFO] Loaded LoRA weights from models/interior_design/...
[INFO] Loading floorplan pipeline...
[INFO] Loaded LoRA weights from models/floor_plan_model/...
[INFO] --- Both pipelines loaded successfully ---
[INFO] Running warm-up inference to mitigate cold start...
100%|██████████| 1/1 [00:02<00:00,  2.00it/s]
100%|██████████| 1/1 [00:02<00:00,  2.00it/s]
[INFO] Warm-up completed.
[gunicorn] Listening at: http://0.0.0.0:5000
[gunicorn] Worker ready
```

✅ **If you see these logs, the deployment is successful!**

---

## 🧪 Test the Deployment

### 1. Check Server Status

```bash
curl https://xxxx-xxxx.pods.runpod.io/
```

Expected response:
```html
<h1>Flask Server is Running!</h1>...
```

### 2. Generate a Single Interior Image

```bash
curl -N -H "Content-Type: application/json" -X POST \
  https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{
    "prompt": "cozy modern bedroom",
    "style": "scandinavian",
    "tool_type": "interior",
    "num_images": 1
  }'
```

Expected: Streaming SSE chunks with base64 encoded image data.

### 3. Generate Multiple Images (Single Forward Pass)

```bash
curl -N -H "Content-Type: application/json" -X POST \
  https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{
    "prompt": "modern living room",
    "style": "minimalist",
    "tool_type": "interior",
    "num_images": 2
  }'
```

Generates 2 images in a **single forward pass** (not two separate passes).

### 4. Test Concurrency (Two Simultaneous Requests)

Open two browser tabs or use two curl processes:

**Terminal 1:**
```bash
curl -N -H "Content-Type: application/json" -X POST \
  https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{"prompt":"bedroom","style":"modern","tool_type":"interior","num_images":1}'
```

**Terminal 2 (start while Terminal 1 is generating):**
```bash
curl -N -H "Content-Type: application/json" -X POST \
  https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{"prompt":"floor plan","tool_type":"floorplan","num_images":1}'
```

✅ **Both should succeed without CUDA OOM.** The semaphore ensures exactly 2 concurrent inferences on GPU.

### 5. Confirm GPU Usage

Inside the pod, check:

```bash
nvidia-smi
```

Should show RTX 4090 with memory usage during inference.

Also, the backend logs will show:

```
[INFO] CUDA (GPU) is available. Using GPU for faster generation.
```

---

## 🔧 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FRONTEND_ORIGIN` | `*` | CORS allowed origins. Set to your Vercel domain for production. |
| `HUGGINGFACE_HUB_TOKEN` | (none) | HF token if models require authentication. |
| `JWT_SECRET_KEY` | `change-this-secret` | JWT signing key (change in production!). |
| `MONGO_URI` | (none) | MongoDB connection string for user auth (optional). |
| `PORT` | `5000` | Port to bind to (normally 5000 for RunPod). |

**To set in RunPod UI:**
- Pod Settings → Environment Variables → Add each variable.

---

## ⚙️ Model Download & Caching

### First Startup

1. Container starts.
2. Backend attempts to load `runwayml/stable-diffusion-v1-5` from Hugging Face.
3. Model **downloads automatically** to `/root/.cache/huggingface/hub/` (~5GB).
4. Model is used to initialize both pipelines.
5. LoRA weights are loaded from `/app/models/`.
6. Warm-up inference runs (1-2 minutes on RTX 4090).
7. Server ready to accept requests.

**Total cold start time: 5–10 minutes** (includes download, model load, warm-up).

### Subsequent Starts

If using a **persistent volume** at `/app/models`:

1. Re-attach the volume to a new pod.
2. The base model is cached in `/root/.cache/huggingface/hub/` (if you mount it separately).
3. Startup time: **1–2 minutes** (no re-download, just warm-up).

**Recommendation**: Mount both:
- `/app/models` for LoRA weights
- `/root/.cache/huggingface/hub/` for the base model cache

---

## 🎯 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **First Request Latency** | 30–60s | Model is preloaded + warmed up at startup. |
| **Subsequent Request** | 20–40s | Generation time depends on num_inference_steps (30 by default). |
| **Max Concurrent Users** | 2 | Enforced by semaphore; more requests queue. |
| **VRAM Usage** | ~20GB | Both pipelines + LoRA weights preloaded. |
| **Multi-Image (2 images)** | ~1.5x slower than 1 image | Single forward pass; no memory duplication. |

---

## 🚨 Troubleshooting

### Issue: Models load twice on startup

**Cause**: Debug mode enabled or Flask reloader active.

**Fix**: Ensure `debug=False` in backend.py and run via Gunicorn (not `python backend.py`).

### Issue: CUDA Out of Memory (OOM)

**Cause**: GPU not available or incorrect device.

**Fix**: 
- Confirm `torch.cuda.is_available() == True` in logs.
- Reduce `num_images` to 1 and retry.
- Check `/root/.cache/huggingface/hub/` isn't full (delete old model checkpoints if needed).

### Issue: LoRA not loading

**Cause**: Incorrect file path or missing files.

**Fix**:
- Verify files exist at:
  - `models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors`
  - `models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors`
- Check logs for: `[WARNING] LoRA path not found or load not supported`.

### Issue: Slow first request after startup

**Cause**: Warm-up inference did not complete or GPU still initializing.

**Fix**:
- Increase HEALTHCHECK grace period in Dockerfile (`--start-period=600s`).
- Check logs for: `[INFO] Warm-up completed.`
- Wait 1–2 minutes before sending first request.

### Issue: Endpoint returns 503 "Models not available"

**Cause**: Model loading failed at startup.

**Fix**:
- Check pod logs for errors during `load_models()`.
- Verify Hugging Face connection (persistent volume or internet access).
- Confirm GPU driver is loaded: `nvidia-smi`.

---

## 📦 Dependencies

The `requirements.txt` must include:

```
torch>=2.1.0
torchvision
torchaudio
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
safetensors>=0.3.1
Flask>=2.3.0
flask-cors>=4.0.0
flask-jwt-extended>=4.4.0
pymongo>=4.4.0
python-dotenv>=1.0.0
Pillow>=10.0.0
gunicorn>=21.0.0
xformers>=0.0.20  # (optional, for memory optimization)
```

---

## 🎬 Final Verification Checklist

Before live demo:

- [ ] Pod is running on RTX 4090
- [ ] Port 5000 is exposed publicly via HTTP
- [ ] `torch.cuda.is_available() == True` (confirmed in logs)
- [ ] Base model downloaded and cached
- [ ] Both LoRA weights loaded successfully
- [ ] Warm-up inference completed without errors
- [ ] Single interior + floorplan requests succeed
- [ ] Two concurrent requests succeed without OOM
- [ ] Multi-image generation uses single forward pass
- [ ] Response time is acceptable (~30–60s for first request, ~20–40s after)
- [ ] Endpoint accessible from external browser
- [ ] CORS configured correctly (FRONTEND_ORIGIN set appropriately)

✅ **If all above pass, ready for live demo!**

---

## 📞 Support

If issues arise:

1. Check pod logs in RunPod UI
2. Verify GPU driver: `nvidia-smi`
3. Test locally with Waitress: `waitress-serve --host=0.0.0.0 --port=5000 backend:app`
4. Confirm Python version: `python --version` (3.10+)
5. Validate Docker build: `docker build -t test:latest .`

---

## 🔐 Security Notes

Before production deployment:

- [ ] Change `JWT_SECRET_KEY` to a strong value
- [ ] Set `FRONTEND_ORIGIN` to your actual domain (not `*`)
- [ ] Configure `MONGO_URI` with a production MongoDB instance
- [ ] Use Hugging Face token (HUGGINGFACE_HUB_TOKEN) if models are private
- [ ] Enable HTTPS on the RunPod proxy (RunPod handles this by default)

---

**Happy deploying! 🚀**
