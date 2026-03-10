# 🚀 Vercel + RunPod Deployment Summary

Your project is ready for deployment! Here's what to do next:

---

## 📋 Deployment Checklist

### ✅ Backend Setup (RunPod) - Do This First

**Timeline: 15-30 minutes**

1. **Create RunPod Account**
   - Go to https://runpod.io
   - Sign up and add payment method

2. **Upload LoRA Weights**
   - Create Persistent Volume (50GB minimum)
   - Upload your model files:
     - `models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors`
     - `models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors`

3. **Create GPU Pod**
   - GPU: RTX 4090 (recommended)
   - Container: Use your `dockerfile` from the repo
   - Expose Port: 5000 (HTTP)

4. **Set Environment Variables**
   ```
   FRONTEND_ORIGIN=https://spatiora-frontend.vercel.app
   JWT_SECRET_KEY=generate-with-openssl-rand-hex-32
   MONGO_URI=(optional)
   PORT=5000
   ```

5. **Mount Persistent Volume**
   - Path: `/app/models`
   - This caches the base model for faster restarts

6. **Deploy and Wait**
   - Pod should fully start in 3-5 minutes
   - Check logs for: "Both pipelines loaded successfully"
   - **Copy your RunPod URL** (format: `https://xxxx-xxxx.pods.runpod.io`)

---

### ✅ Frontend Setup (Vercel) - Do This Next

**Timeline: 10 minutes**

1. **Update Backend URL in Backend**
   - Edit `frontend/config.js`
   - Replace line 3 with your RunPod URL:
     ```javascript
     window.BACKEND_URL = 'https://xxxx-xxxx.pods.runpod.io';  // Your actual URL
     ```

2. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "Update backend URL and prepare for Vercel deployment"
   git push origin main
   ```

3. **Create Vercel Account**
   - Go to https://vercel.com
   - Sign up with GitHub

4. **Import Project**
   - Click "Add New" → "Project"
   - Select your GitHub repository
   - Click "Import"

5. **Configure Project**
   - **Framework**: Select "Other" (static site)
   - **Root Directory**: `./` or leave blank (Vercel will auto-detect)
   - Click "Deploy"

6. **Get Your Frontend URL**
   - Vercel will assign a URL like: `https://spatiora-frontend.vercel.app`
   - **Save this URL** - you'll need it for configuration

---

### ✅ Connect Frontend to Backend

**Timeline: 5 minutes**

1. **Update RunPod CORS Settings**
   - Go back to RunPod pod settings
   - Update environment variable:
     ```
     FRONTEND_ORIGIN=https://your-vercel-url.vercel.app
     ```
   - Restart the pod

2. **Test the Connection**
   - Open your Vercel frontend URL
   - Navigate to "Interior" or "Floor Plan" section
   - Try generating an image
   - Images should stream from RunPod ✓

---

## 📁 Files Updated/Created

The following files have been created/updated to support your deployment:

| File | Purpose |
|------|---------|
| `VERCEL_RUNPOD_DEPLOYMENT.md` | Comprehensive deployment guide (detailed instructions for troubleshooting) |
| `QUICK_DEPLOYMENT.md` | Quick start guide (start here!) |
| `frontend/config.js` | Backend URL configuration (update this before deploying) |
| `package.json` | NPM package file (for future build tools) |
| `vercel.json` | Vercel configuration (deployment settings) |
| `inject-env.js` | Environment injection script (for build process) |

---

## 🔑 Important Configuration Values

You'll need to set these environment variables on RunPod:

```
FRONTEND_ORIGIN=https://spatiora-frontend.vercel.app
JWT_SECRET_KEY=your-secret-key-here
MONGO_URI=(optional for user auth)
PORT=5000
```

Generate a strong JWT secret:
```bash
# PowerShell:
[System.Guid]::NewGuid().ToString().Replace('-', '')

# Or use online: https://generate-random.org/
```

---

## 💰 Costs

- **Vercel Frontend**: FREE (with generous free tier)
- **RunPod Backend**: ~$0.29/hour for RTX 4090
  - **Stop the pod when not in use** to avoid unexpected charges
  - You only pay for running pods, not stopped ones

---

## 🧪 Quick Test Commands

### Test Backend is Running
```bash
curl https://xxxx-xxxx.pods.runpod.io/
```
Expected: HTML response with "Flask Server is Running!"

### Test Generation Endpoint
```bash
curl -N https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "modern bedroom",
    "style": "minimalist",
    "tool_type": "interior",
    "num_images": 1
  }'
```
Expected: Streaming JSON with base64 encoded images

---

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **CORS Error in browser** | Update `FRONTEND_ORIGIN` on RunPod, restart pod |
| **Blank page on Vercel** | Check console (F12), verify `config.js` has correct URL |
| **Images not generating** | Check RunPod pod is running, check logs for errors |
| **Slow generation** | Ensure GPU is RTX 4090, check no concurrent requests |
| **Pod keeps restarting** | Check VRAM usage, increase timeout to 300s |

---

## 📚 Next Steps

1. **For detailed help**: Read [VERCEL_RUNPOD_DEPLOYMENT.md](VERCEL_RUNPOD_DEPLOYMENT.md)
2. **For troubleshooting**: Check [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md)
3. **For quick reference**: Use [QUICK_DEPLOYMENT.md](QUICK_DEPLOYMENT.md)

---

## 🆘 Troubleshooting

### If deployment fails:

1. **Check that `frontend/config.js` has your RunPod URL**
   - Look for the line: `window.BACKEND_URL = 'https://...'`
   - Should match your actual RunPod endpoint

2. **Check Vercel build logs**
   - Vercel Dashboard → Deployments → Click failed build → View logs

3. **Check RunPod pod status**
   - Should show "RUNNING" status
   - Check logs for any error messages
   - Verify GPU usage in metrics

4. **Test CORS**
   - From browser console:
     ```javascript
     fetch('https://your-runpod-url.pods.runpod.io/')
     ```
   - Should NOT show CORS error if configured correctly

---

## 📞 Support Links

- **Vercel Help**: https://vercel.com/support
- **RunPod Help**: https://docs.runpod.io
- **Stable Diffusion Docs**: https://huggingface.co/runwayml/stable-diffusion-v1-5
- **Flask Docs**: https://flask.palletsprojects.com

---

**You're all set! Deploy with confidence! 🎉**

Last updated: March 10, 2026
