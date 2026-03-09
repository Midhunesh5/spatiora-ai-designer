# RunPod Deployment Checklist

## 📋 Pre-Deployment Verification (Local)

Before deploying to RunPod, verify everything works locally:

```bash
# 1. Activate Python environment
& D:\main_pwl\env\Scripts\Activate.ps1

# 2. Run startup test
python test_backend_startup.py

# 3. Start Waitress (Windows testing server, not Gunicorn)
waitress-serve --host=0.0.0.0 --port=5000 backend:app

# 4. In another terminal, test the endpoint
curl http://127.0.0.1:5000/

# 5. Monitor logs for:
#    - "CUDA (GPU) is available"
#    - "Loading interior pipeline..."
#    - "Loading floorplan pipeline..."
#    - "Both pipelines loaded successfully"
#    - "Running warm-up inference..."
#    - "Warm-up completed."
```

✅ If all logs appear and endpoint responds, proceed to RunPod.

---

## 🚀 RunPod Deployment Setup

### 1. **Prepare Models**

Upload LoRA weights to RunPod (via file upload or persistent volume):

```
models/
├── floor_plan_model/
│   └── lora_weights/
│       └── floor_pytorch_lora_weights.safetensors
└── interior_design/
    └── interior_gen_lora_weights/
        └── pytorch_lora_weights.safetensors
```

### 2. **Create RunPod GPU Pod**

- **GPU**: RTX 4090
- **Container Image**: Use your Dockerfile (or save as private image)
- **Expose Port**: 5000 (HTTP)

### 3. **Set Environment Variables**

In RunPod UI, set these:

```
FRONTEND_ORIGIN=*                          # Testing with all origins
HUGGINGFACE_HUB_TOKEN=(leave empty if not private)
JWT_SECRET_KEY=(generate a strong key, e.g., $(openssl rand -hex 32))
MONGO_URI=(leave empty for now, optional)
PORT=5000
```

### 4. **Container Startup Command**

The Dockerfile already specifies:
```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 300 backend:app
```

**Important**: Single worker only (`--workers 1`), no debug mode.

### 5. **Mount Persistent Volume (Recommended)**

- Attach persistent volume to `/app/models` (for LoRA weights)
- Optionally, mount `/root/.cache/huggingface/hub/` (to cache base model)

---

## 🔍 Post-Deployment Verification

Once pod is running:

### 1. **Check Pod Logs**

Look for these lines (in order):

```
[INFO] --- Starting Model Loading (two pipelines from Hugging Face) ---
[INFO] Loading interior pipeline...
[INFO] Downloading base model from Hugging Face...
[INFO] Loaded LoRA weights from models/interior_design/...
[INFO] Loading floorplan pipeline...
[INFO] Loaded LoRA weights from models/floor_plan_model/...
[INFO] --- Both pipelines loaded successfully ---
[INFO] Running warm-up inference to mitigate cold start...
100%|██████████| 1/1 [00:XX<00:00, X.XXit/s]
100%|██████████| 1/1 [00:XX<00:00, X.XXit/s]
[INFO] Warm-up completed.
[gunicorn] Listening at: http://0.0.0.0:5000 (3 workers)
[gunicorn] Worker ready
```

#### ✅ Key checkpoints:
- `Both pipelines loaded successfully` ← Confirms dual pipelines
- `Warm-up completed` ← Confirms warm-up ran
- `Worker ready` ← Gunicorn is listening

### 2. **Get Public URL**

RunPod displays: `https://xxxx-xxxx.pods.runpod.io/` (HTTP proxy URL)

### 3. **Test Endpoint from Browser**

Visit: `https://xxxx-xxxx.pods.runpod.io/`

Expected response:
```html
<h1>Flask Server is Running!</h1>...
```

### 4. **Test Generation (Single Image)**

```bash
curl -N -H "Content-Type: application/json" \
  -X POST https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{"prompt":"cozy bedroom","style":"modern","tool_type":"interior","num_images":1}'
```

Expected: Streaming SSE chunks with base64 image data (takes ~30–60s on RTX 4090).

### 5. **Test Concurrency (Two Simultaneous Requests)**

**Terminal 1:**
```bash
curl -N -H "Content-Type: application/json" \
  -X POST https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{"prompt":"bedroom","style":"scandinavian","tool_type":"interior","num_images":1}'
```

**Terminal 2 (start while Terminal 1 is generating):**
```bash
curl -N -H "Content-Type: application/json" \
  -X POST https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{"prompt":"floor plan","tool_type":"floorplan","num_images":1}'
```

✅ **Both should succeed without CUDA OOM.**

### 6. **Test Multi-Image Single Forward Pass**

```bash
curl -N -H "Content-Type: application/json" \
  -X POST https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -d '{"prompt":"modern living room","style":"minimalist","tool_type":"interior","num_images":2}'
```

✅ **Generates 2 images in ~1.5x time of 1 image (single forward pass, not two separate passes).**

### 7. **Confirm No Reruns on First Request**

Check logs after first generation:

```
[INFO] Streaming 1 image(s)...
```

**Key observation**: Should see ONLY ONE set of pipelines load + ONE warm-up. No reloads.

---

## 🚨 Common Deployment Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Pod stuck loading | HF model download in progress (normal) | Wait 5–10 minutes |
| CUDA not found | GPU not attached to pod | Restart pod; select RTX 4090 |
| Models load twice | Debug mode or Flask reloader | Verify Dockerfile CMD uses gunicorn |
| LoRA not found | Files not uploaded to `/app/models/` | Upload files to persistent volume |
| CUDA OOM on 2 concurrent requests | Semaphore misconfigured | Verify `Semaphore(2)` in backend.py |
| 503 "Models not available" | Model loading failed | Check pod logs for errors during startup |

---

## 📊 Performance Expectations (RTX 4090)

| Scenario | Time | Notes |
|----------|------|-------|
| **Cold Start (first pod boot)** | 5–10 min | Download base model (~5GB) + load pipelines + warm-up |
| **Warm Start (pod restart, persistent volume cached)** | 1–2 min | Just pipeline load + warm-up, no download |
| **First Generation Request** | 30–60s | Model already preloaded + warmed up |
| **Subsequent Requests** | 20–40s | Depends on num_inference_steps (30 by default) |
| **Multi-Image (2 images)** | ~1.5x slower | Single forward pass (more efficient than two passes) |
| **Concurrent User 1 + User 2** | Both ~30–70s | Semaphore queues; run sequentially on GPU |

---

## 🔐 Post-Launch Security Hardening

After deployment is stable:

1. **Change JWT Secret**:
   ```bash
   export JWT_SECRET_KEY=$(openssl rand -hex 32)
   # Update in RunPod pod environment
   ```

2. **Restrict CORS**:
   ```bash
   export FRONTEND_ORIGIN=https://your-vercel-domain.com
   # Update in RunPod pod environment
   ```

3. **Configure MongoDB** (if needed):
   ```bash
   export MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db
   ```

4. **Monitor GPU Usage**:
   ```bash
   nvidia-smi -l 1  # Update every 1 second
   ```

---

## ✅ Final Readiness Checklist

Before live demo, confirm:

- [ ] Pod running on RTX 4090 with HTTP exposed on port 5000
- [ ] Logs show `Both pipelines loaded successfully`
- [ ] Logs show `Warm-up completed`
- [ ] `curl https://xxxx-xxxx.pods.runpod.io/` responds with HTML
- [ ] Single interior image generation works (~30–60s)
- [ ] Single floorplan image generation works (~30–60s)
- [ ] Two concurrent requests both succeed (no OOM)
- [ ] Multi-image generation (2 images) runs in single forward pass
- [ ] No model reloads in logs (loads once only)
- [ ] GPU confirmed active: `nvidia-smi` shows RTX 4090
- [ ] Response headers include CORS headers (if configured)

✅ **If all above pass, ready for live demo!**

---

## 📞 Emergency Troubleshooting

If pod crashes mid-demo:

1. **Immediately check logs** in RunPod UI for the error
2. **Common crashes**:
   - Out of Memory: Reduce `num_images` to 1
   - CUDA Error: Restart pod (GPU may need reset)
   - Model loading error: Verify LoRA files uploaded
3. **Restart pod** (RunPod UI → Pod → Restart)
4. **Monitor for 10+ minutes** until "Worker ready" appears in logs

---

## 🎬 Demo Script

Once confirmed working:

```
1. Open https://xxxx-xxxx.pods.runpod.io in browser (should show "Flask Server is Running")
2. Open terminal:
   curl -N -H "Content-Type: application/json" -X POST \
     https://xxxx-xxxx.pods.runpod.io/generate-stream \
     -d '{"prompt":"modern kitchen","style":"minimalist","tool_type":"interior","num_images":1}'
3. Wait 30–60 seconds, image appears (base64 in SSE stream)
4. Show 2 concurrent requests running without OOM
5. Done! 🎉
```

---

**Deployment Guide Created: $(date)**

Good luck with your live demo! 🚀
