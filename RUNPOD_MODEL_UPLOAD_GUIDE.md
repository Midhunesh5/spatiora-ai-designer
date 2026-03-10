# Guide: Uploading Models to RunPod via JupyterLab

This guide walks you through creating the correct folder structure and uploading your LoRA weights using RunPod's JupyterLab interface.

---

## 📁 Folder Structure You Need to Create

Based on your `backend.py`, create **exactly** this structure in `/app/models/`:

```
/app/models/
├── floor_plan_model/
│   └── lora_weights/
│       └── floor_pytorch_lora_weights.safetensors
├── interior_design/
│   └── interior_gen_lora_weights/
│       └── pytorch_lora_weights.safetensors
└── sd-v1-5-fp16/
    └── (Will be auto-downloaded from Hugging Face - don't upload)
```

**Important**: The base model (`sd-v1-5-fp16`) will be **automatically downloaded** from Hugging Face on first run. You only need to upload the **two LoRA weight files**.

---

## 🚀 Step-by-Step Instructions

### Step 1: Open JupyterLab on RunPod

1. Go to your RunPod pod dashboard
2. Click **"Jupyter"** or **"JupyterLab"** button
3. Wait for JupyterLab to load (opens in a new tab)

---

### Step 2: Open Terminal in JupyterLab

1. In JupyterLab, click **"File"** → **"New"** → **"Terminal"**
2. A terminal window opens at the bottom

---

### Step 3: Create the Folder Structure

Copy and paste **all of these commands** into the JupyterLab terminal:

```bash
# Create the main models directory
mkdir -p /app/models

# Create floor plan model folder structure
mkdir -p /app/models/floor_plan_model/lora_weights

# Create interior design model folder structure
mkdir -p /app/models/interior_design/interior_gen_lora_weights

# Verify the structure was created
tree /app/models/
```

**Expected output** after running `tree /app/models/`:
```
/app/models/
├── floor_plan_model
│   └── lora_weights
└── interior_design
    └── interior_gen_lora_weights

4 directories, 0 files
```

*(If `tree` command not found, use: `ls -la /app/models/`)*

---

### Step 4: Upload LoRA Weight Files

You have **two LoRA files to upload**:

1. **`floor_pytorch_lora_weights.safetensors`** - Goes to `/app/models/floor_plan_model/lora_weights/`
2. **`pytorch_lora_weights.safetensors`** (interior) - Goes to `/app/models/interior_design/interior_gen_lora_weights/`

#### Method A: Upload via JupyterLab UI (Easiest)

1. In the left sidebar, navigate to: **File Browser** → scroll down to find `/app/models/`
2. Double-click to open `/app/models/`
3. Open `floor_plan_model/lora_weights/`
4. Click **Upload** button (⬆️ icon) on the toolbar
5. Select your `floor_pytorch_lora_weights.safetensors` file
6. Wait for upload to complete ✓

7. Navigate back to `/app/models/`
8. Open `interior_design/interior_gen_lora_weights/`
9. Click **Upload** button again
10. Select your `pytorch_lora_weights.safetensors` file
11. Wait for upload to complete ✓

#### Method B: Upload via Terminal (Alternative)

If files are on your local machine and you want faster upload:

```bash
# First, download your files to the RunPod pod (if they're somewhere else)
# Option: Use scp from your local machine or a web link

# If you have a download link:
wget -O /app/models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors https://your-file-url

wget -O /app/models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors https://your-file-url
```

---

### Step 5: Verify Files Were Uploaded

Run this command in JupyterLab terminal to verify:

```bash
# List all files in the models directory
find /app/models/ -type f

# Show detailed file sizes
du -sh /app/models/*/*
```

**Expected output**:
```
/app/models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors
/app/models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors
```

**Check file sizes** (should be between 50MB - 500MB each):
```bash
ls -lh /app/models/floor_plan_model/lora_weights/
ls -lh /app/models/interior_design/interior_gen_lora_weights/
```

---

## ✅ Verification Checklist

Run this command to verify everything is in place:

```bash
# Quick verification script
echo "=== Checking Model Structure ===" && \
echo "Floor Plan LoRA:" && ls -lh /app/models/floor_plan_model/lora_weights/ && \
echo "" && \
echo "Interior Design LoRA:" && ls -lh /app/models/interior_design/interior_gen_lora_weights/ && \
echo "" && \
echo "✓ Everything looks good!" || echo "✗ Missing files encountered"
```

**All three conditions should be met**:
- ✅ `/app/models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors` exists
- ✅ `/app/models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensores` exists
- ✅ Base model files can be auto-downloaded when backend starts

---

## 🚨 Common Issues

### Issue: "Permission Denied" when creating folders

**Solution**: Add `sudo` prefix:
```bash
sudo mkdir -p /app/models/floor_plan_model/lora_weights
sudo mkdir -p /app/models/interior_design/interior_gen_lora_weights
```

### Issue: Files uploaded to wrong location

**Solution**: Use this command to find where they are:
```bash
find /app -name "*.safetensors" 2>/dev/null
```

Then move them to correct location:
```bash
mv /path/to/wrong/location/file.safetensors /app/models/floor_plan_model/lora_weights/
```

### Issue: Upload is very slow

**Solution**:
- Check your internet connection
- Use terminal upload method instead (can be faster)
- Upload one file at a time

### Issue: "No space left on device"

**Solution**:
- Check available storage:
  ```bash
  df -h /app
  ```
- LoRA files should be ~100-500MB total
- Base model is ~5GB (will download automatically)
- Ensure you have at least 20GB free space

---

## 📝 What Happens Next

Once files are uploaded:

1. **Mount Persistent Volume** (in RunPod settings)
   - Attach to `/app/models`
   - This persists files across pod restarts and saves re-uploading

2. **Update backend.py** (already configured)
   - Lines 15-16 point to correct paths
   - Auto-detects LoRA files on startup

3. **Backend Will Auto-Download**
   - Base model: `runwayml/stable-diffusion-v1-5`
   - First startup: ~10 minutes (downloads 5GB)
   - Subsequent starts: ~2 minutes (cached)

4. **First Inference Will Be Warm-up**
   - Backend loads both pipelines
   - Runs test inference
   - Then ready to serve requests

---

## 🎯 Quick Command Reference

**Copy entire block and paste into JupyterLab terminal**:

```bash
# Create all folders at once
mkdir -p /app/models/floor_plan_model/lora_weights
mkdir -p /app/models/interior_design/interior_gen_lora_weights

# Verify structure
echo "Created folders:" && ls -la /app/models/*/

# After uploading files, verify
echo "Checking uploaded files..." && \
ls -lh /app/models/floor_plan_model/lora_weights/ && \
ls -lh /app/models/interior_design/interior_gen_lora_weights/
```

---

## 📞 Still Having Issues?

**Check these paths in backend.py** (lines 15-16):

```python
FLOORPLAN_LORA_PATH = "models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors"
INTERIOR_LORA_PATH = "models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors"
```

These paths are **relative to `/app/`**, so files should be:
- `/app/models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors`
- `/app/models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors`

If your files have **different names**, either:
1. Rename them to match the paths above, OR
2. Update the paths in `backend.py` to match your filenames

---

## ✨ Pro Tips

1. **Mount persistent volume BEFORE restarting pod**
   - Prevents re-uploading files
   - Caches downloaded base model

2. **Test backend after upload**
   - Navigate to RunPod pod's HTTP endpoint
   - Check logs for "Loaded LoRA weights from..."
   - Run a test generation request

3. **Monitor disk space during first run**
   - Base model download (~5GB) happens on first startup
   - Make sure you have ~20GB free space total

4. **If uploading large files**
   - Use persistent volume to store files between restarts
   - Don't rely on ephemeral storage for model files

---

You're all set! Once files are uploaded and persistent volume is mounted, your RunPod backend is ready to go! 🚀
