# Spatiora: Vercel + RunPod Deployment Guide

Complete guide to deploy the frontend on Vercel and backend on RunPod with proper communication setup.

---

## 🎯 Overview

- **Frontend**: Vercel (serverless, auto-deploys from GitHub)
- **Backend**: RunPod (GPU pod with RTX 4090 for AI inference)
- **Communication**: Frontend makes API calls to RunPod backend URL

---

# PART 1: FRONTEND DEPLOYMENT TO VERCEL

## Prerequisites

- GitHub account (with repository containing your frontend code)
- Vercel account (free tier works)

## Step 1: Prepare Frontend for Vercel

### 1.1 Create `package.json` in the root (if not exists)

```json
{
  "name": "spatiora-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint ."
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^4.3.0"
  }
}
```

OR if using plain HTML/JS (no build step):

```json
{
  "name": "spatiora-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "python -m http.server 3000"
  }
}
```

### 1.2 Update Your Frontend Code

Update the API base URL in your frontend code to use the RunPod backend:

**In `frontend/js/app.js`** (or wherever API calls are made):

```javascript
// Get the backend URL from environment variable or fallback
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 
                    (window.location.hostname === 'localhost' 
                      ? 'http://localhost:5000' 
                      : 'https://your-runpod-url.pods.runpod.io');

// Example API call function
async function generateImages(prompt, style, toolType, numImages) {
  try {
    const response = await fetch(`${BACKEND_URL}/generate-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      },
      body: JSON.stringify({
        prompt,
        style,
        tool_type: toolType,
        num_images: numImages
      })
    });

    // Handle streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.url) {
            updateUI(data); // Display the image
          }
        }
      }
    }
  } catch (error) {
    console.error('Generation failed:', error);
  }
}
```

### 1.3 Create a `.env.local` file (for local testing)

```
REACT_APP_BACKEND_URL=http://localhost:5000
```

### 1.4 Push Code to GitHub

```bash
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

---

## Step 2: Deploy to Vercel

### 2.1 Connect GitHub Repository

1. Go to [vercel.com](https://vercel.com)
2. Click **"Import Project"**
3. Select **"Import Git Repository"**
4. Paste your GitHub repo URL (e.g., `https://github.com/yourusername/spatiora`)
5. Click **"Continue"**

### 2.2 Configure Vercel Project

1. **Project Name**: `spatiora-frontend` (or your preference)
2. **Framework**: Select **"Other"** (or **"Next.js"** if using Next.js)
3. **Root Directory**: Leave as `.` or set to `frontend/` if that's where your files are
4. Click **"Continue"**

### 2.3 Set Environment Variables

On the "Environment Variables" step, add:

```
REACT_APP_BACKEND_URL = https://your-runpod-url.pods.runpod.io
```

(You'll get the actual RunPod URL after deploying the backend in Part 2)

### 2.4 Deploy

1. Click **"Deploy"**
2. Wait for build to complete (usually 1-2 minutes)
3. Your frontend is now live at `https://spatiora-frontend.vercel.app` (or your custom domain)

### 2.5 Update Backend URL After RunPod Deployment

Once you deploy the backend (Part 2), update the environment variable:

1. Go to Vercel Project Settings → Environment Variables
2. Update `REACT_APP_BACKEND_URL` to your actual RunPod URL
3. Redeploy: Click **"Deployments"** → Click the latest deployment → Click **"Redeploy"**

---

# PART 2: BACKEND DEPLOYMENT TO RUNPOD

## Prerequisites

- RunPod account
- Docker installed (for local testing, optional)

## Step 1: Prepare LoRA Weights

### Option A: Upload via RunPod UI (Easiest for first-time setup)

1. Create a **Persistent Volume** in RunPod:
   - Go to RunPod Dashboard → Cluster Storage
   - Create a new persistent volume (e.g., 50GB)
   - Name it: `spatiora-models`

2. Upload your LoRA files:
   ```
   models/
   ├── floor_plan_model/
   │   └── lora_weights/
   │       └── floor_pytorch_lora_weights.safetensors
   └── interior_design/
       └── interior_gen_lora_weights/
           └── pytorch_lora_weights.safetensors
   ```

### Option B: Include in Docker Image

If files are small (<1GB), add to Dockerfile:

```dockerfile
COPY models/ /app/models/
```

---

## Step 2: Create RunPod Pod

### 2.1 Create New Pod

1. Go to [runpod.io](https://runpod.io)
2. Click **"Secure Cloud"** → **"GPU Pods"**
3. Click **"New Pod"**
4. Search for and select a template or start from **"RunPod Official Docker Template"**

### 2.2 Configure Pod

**GPU Selection**:
- GPU Type: **RTX 4090** (or RTX 4080 if 4090 not available for cost reasons)
- GPU Count: 1
- Volume Size: 50GB

**Container Image**:
- Use your Dockerfile or provide the registry path:
  ```
  registry.runpod.io/your-namespace/spatiora:latest
  ```
  OR paste Dockerfile content directly in RunPod UI

**Port Configuration**:
- HTTP Port: `5000`
- Enable **"Expose HTTP Port"**

### 2.3 Set Environment Variables

In the RunPod pod settings, add these environment variables:

```
FRONTEND_ORIGIN=https://spatiora-frontend.vercel.app
HUGGINGFACE_HUB_TOKEN=(leave empty unless using private models)
JWT_SECRET_KEY=your-secret-key-generate-with-openssl-rand-hex-32
MONGO_URI=(optional: your MongoDB connection string if using auth)
PORT=5000
```

### 2.4 Mount Persistent Volume

1. Attach the persistent volume (`spatiora-models`) created in Step 1
2. Mount path: `/app/models`
3. This caches the base model and LoRA weights across pod restarts

### 2.5 Deploy Pod

1. Click **"Let's Go!"** or **"Deploy"**
2. Wait for pod to start (2-5 minutes depending on image size)
3. Check logs for:
   ```
   [INFO] --- Starting Model Loading (two pipelines from Hugging Face) ---
   [INFO] Loading interior pipeline...
   [INFO] Loaded LoRA weights from models/interior_design/...
   [INFO] Loading floorplan pipeline...
   [INFO] Loaded LoRA weights from models/floor_plan_model/...
   [INFO] --- Both pipelines loaded successfully ---
   [INFO] Running warm-up inference...
   [INFO] Warm-up completed.
   [gunicorn] Worker ready
   ```

---

## Step 3: Get RunPod Backend URL

Once the pod is running:

1. Go to your RunPod dashboard
2. Find your pod in the list
3. Copy the **HTTP endpoint** URL
   - Format: `https://xxxx-xxxx.pods.runpod.io`

---

## Step 4: Test Backend Deployment

### 4.1 Test Server Status

```bash
curl https://xxxx-xxxx.pods.runpod.io/
```

Expected:
```html
<h1>Flask Server is Running!</h1>...
```

### 4.2 Test Interior Generation

```bash
curl -N https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cozy modern bedroom",
    "style": "scandinavian",
    "tool_type": "interior",
    "num_images": 1
  }'
```

### 4.3 Test Floor Plan Generation

```bash
curl -N https://xxxx-xxxx.pods.runpod.io/generate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "2-bedroom apartment with open kitchen",
    "tool_type": "floorplan",
    "num_images": 1
  }'
```

---

# PART 3: CONNECT FRONTEND TO BACKEND

## Step 1: Update Vercel Environment Variables

1. Go to Vercel Project Settings
2. **Environment Variables** section
3. Add or update:
   ```
   REACT_APP_BACKEND_URL=https://xxxx-xxxx.pods.runpod.io
   ```
4. Redeploy the frontend

## Step 2: Update RunPod CORS Settings

Ensure backend allows your Vercel domain:

In `backend.py`, update line 32:

```python
_allowed_origin = os.environ.get("FRONTEND_ORIGIN", "*")  # Set to your Vercel URL in production
```

RunPod environment variable:
```
FRONTEND_ORIGIN=https://spatiora-frontend.vercel.app
```

## Step 3: Test Full Pipeline

1. Open your Vercel frontend: `https://spatiora-frontend.vercel.app`
2. Go to the generation page
3. Enter a prompt and generate an image
4. Should stream images from RunPod backend

---

# 🔧 TROUBLESHOOTING

## Frontend Issues

### Issue: CORS Error in Browser Console
```
Access to XMLHttpRequest at 'https://xxx.pods.runpod.io' blocked by CORS policy
```

**Solution**:
- Verify `FRONTEND_ORIGIN` environment variable is set correctly on RunPod
- Check that `FRONTEND_ORIGIN` matches your Vercel URL exactly
- Restart the RunPod pod after updating environment variables

### Issue: Backend URL is not updating
```javascript
// Make sure the environment variable is correctly passed to the client
console.log(process.env.REACT_APP_BACKEND_URL);
```

**Solution**:
- Vercel only injects environment variables starting with `REACT_APP_`
- Rebuild/redeploy on Vercel after updating env vars

---

## Backend Issues

### Issue: Models not loading on RunPod

**Check**:
1. Verify persistent volume is mounted at `/app/models`
2. Check pod logs for GPU availability
3. Ensure sufficient disk space (50GB minimum)

**Logs to check**:
```
[INFO] Loading interior pipeline...
[INFO] Downloaded 4.18GB to cache
[INFO] Loaded LoRA weights from models/interior_design/...
```

### Issue: Slow generation on RunPod

**Check**:
1. GPU type is RTX 4090 (not CPU-only)
2. Pod has sufficient VRAM (24GB for RTX 4090)
3. No concurrent requests hitting the semaphore limit
4. Model is cached (second request should be faster)

### Issue: Pod keeps stopping/restarting

**Check**:
1. VRAM usage: Monitor in RunPod logs
2. Timeout: Increase Gunicorn timeout:
   ```
   gunicorn --timeout 300 backend:app
   ```
3. Persistent volume issues: Restart pod

---

# 📊 Performance Tips

## Frontend (Vercel)
- Use lazy loading for images
- Compress initial bundle
- Cache images in browser

## Backend (RunPod)
- Use persistent volume to cache base model
- Keep inference_semaphore at 2 (prevents OOM)
- Monitor GPU memory usage
- Use float16 on GPU (already done in code)

---

# 🚀 Production Checklist

- [ ] Update `JWT_SECRET_KEY` to a strong random value
- [ ] Set `FRONTEND_ORIGIN` to your Vercel domain (not `*`)
- [ ] Enable MongoDB (optional) for user auth
- [ ] Set up HTTPS for all endpoints (Vercel/RunPod handle this)
- [ ] Monitor RunPod pod costs (RTX 4090 ≈ $0.29/hour)
- [ ] Test generation pipeline end-to-end
- [ ] Set up error logging/monitoring
- [ ] Create user documentation

---

# 📝 Environment Variables Reference

## Vercel (Frontend)
```
REACT_APP_BACKEND_URL=https://xxxx-xxxx.pods.runpod.io
```

## RunPod (Backend)
```
FRONTEND_ORIGIN=https://spatiora-frontend.vercel.app
HUGGINGFACE_HUB_TOKEN=(optional)
JWT_SECRET_KEY=your-strong-secret-key
MONGO_URI=(optional)
PORT=5000
```

---

# 🆘 Getting Help

- **Vercel Issues**: Check Vercel docs or ask in their community
- **RunPod Issues**: Check RunPod documentation or community forums
- **CORS Issues**: Verify domain settings on both platforms
- **Model Errors**: Check backend logs for detailed error messages

---

**Deployment Date**: [Date]
**Frontend URL**: https://spatiora-frontend.vercel.app
**Backend URL**: https://xxxx-xxxx.pods.runpod.io
