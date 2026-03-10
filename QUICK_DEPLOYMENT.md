# Quick Setup for Vercel + RunPod Deployment

This document provides quick steps to get your Spatiora frontend on Vercel and backend on RunPod.

## TL;DR - Quick Start

### 1. Deploy Backend to RunPod First (5-10 min)

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for detailed steps, but briefly:

1. Create a RunPod account and GPU pod
2. Select RTX 4090, upload Dockerfile
3. Upload LoRA weights to persistent volume
4. Set environment variables (see RUNPOD_DEPLOYMENT.md)
5. **Copy your RunPod URL** when pod is running (looks like `https://xxxx-xxxx.pods.runpod.io`)

### 2. Update Frontend Config

Update the file `frontend/config.js`:

Replace:
```javascript
window.BACKEND_URL = 'https://your-runpod-url.pods.runpod.io';
```

With your actual RunPod URL:
```javascript
window.BACKEND_URL = 'https://xxxx-xxxx.pods.runpod.io';  // Your actual RunPod URL
```

### 3. Deploy Frontend to Vercel (5 min)

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for Vercel deployment"
   git push origin main
   ```

2. Go to [vercel.com](https://vercel.com)
3. Click **Import Project** → Select your GitHub repo
4. Select **Root Directory**: `frontend` (or `.` if frontend files are at root)
5. Click **Deploy**
6. Your frontend is now live! 🎉

---

## After Deployment

### Test Your Setup

1. Open your Vercel URL (you'll get it after deployment)
2. Go to the generation page
3. Try generating an interior design or floor plan
4. Images should stream from your RunPod backend

### If Something Breaks

**Frontend shows blank page or errors?**
- Check browser console (F12) for errors
- Check that `frontend/config.js` has the correct RunPod URL
- Redeploy on Vercel if you updated the URL

**Images not generating?**
- Check RunPod pod is still running (costs $0.29/hour for RTX 4090)
- Check RunPod pod logs for errors
- Verify CORS is enabled (set `FRONTEND_ORIGIN` in RunPod env vars)

**CORS errors in browser?**
- Update `FRONTEND_ORIGIN` environment variable on RunPod to your Vercel URL:
  ```
  FRONTEND_ORIGIN=https://your-project.vercel.app
  ```
- Restart RunPod pod

---

## Important Notes

- **RunPod pods cost money** when running (~$0.29/hour for RTX 4090)
- **Stop the pod** when not in use to save costs
- **Keep the backend URL updated** if you restart RunPod (you might get a different URL)
- **For production**, set `JWT_SECRET_KEY` to a strong random value

---

## Full Documentation

For detailed deployment guides, see:
- [VERCEL_RUNPOD_DEPLOYMENT.md](VERCEL_RUNPOD_DEPLOYMENT.md) - Complete deployment guide
- [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) - RunPod-specific details
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Pre-deployment checklist

---

**Need help?**

1. **Vercel Issues**: Check [Vercel Docs](https://vercel.com/docs)
2. **RunPod Issues**: Check [RunPod Docs](https://docs.runpod.io)
3. **Frontend Issues**: Check browser console for errors
4. **Backend Issues**: Check RunPod pod logs
