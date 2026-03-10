# Quick Reference: JupyterLab Model Upload

**Copy-paste this into JupyterLab terminal → Done! ✓**

---

## 🎯 One Command Block (Copy All, Paste Once)

```bash
# Create all necessary folders
mkdir -p /app/models/floor_plan_model/lora_weights
mkdir -p /app/models/interior_design/interior_gen_lora_weights

# Verify folders created
echo "=== FOLDER STRUCTURE ===" && \
ls -la /app/models/ && \
echo "" && echo "✓ Folders ready for file upload!"
```

---

## 📂 Folder Structure (Visual)

```
/app/models/                                    ⬅️ Main directory
├── floor_plan_model/
│   └── lora_weights/                          ⬅️ Upload floor_pytorch_lora_weights.safetensors here
│
└── interior_design/
    └── interior_gen_lora_weights/             ⬅️ Upload pytorch_lora_weights.safetensors here
```

---

## 📤 Upload Files via JupyterLab UI

**Two files you need to upload:**

| File Name | Upload To |
|-----------|-----------|
| `floor_pytorch_lora_weights.safetensors` | `/app/models/floor_plan_model/lora_weights/` |
| `pytorch_lora_weights.safetensors` (interior) | `/app/models/interior_design/interior_gen_lora_weights/` |

**Steps**:
1. Left sidebar → File Browser
2. Navigate to folder (e.g., `/app/models/floor_plan_model/lora_weights/`)
3. Click Upload button ⬆️
4. Select file from your computer
5. Done! ✓

---

## ✅ Verify After Upload

Paste this to check everything is in place:

```bash
echo "=== CHECKING UPLOADS ===" && \
echo "Floor Plan File:" && ls -lh /app/models/floor_plan_model/lora_weights/ && \
echo "" && \
echo "Interior Design File:" && ls -lh /app/models/interior_design/interior_gen_lora_weights/ && \
echo "" && echo "✓ All files uploaded!"
```

**You should see**:
- `floor_pytorch_lora_weights.safetensors` (50-500 MB)
- `pytorch_lora_weights.safetensors` (50-500 MB)

---

## 🚀 Then What?

1. ✅ Mount Persistent Volume at `/app/models` (RunPod settings)
2. ✅ Set environment variables (see DEPLOYMENT_SUMMARY.md)
3. ✅ Start the pod
4. ✅ Backend auto-downloads base model
5. ✅ Ready to use! 

---

## ⚡ Terminal Tips

- **Stuck?** Copy the folder creation command above → paste in terminal → press Enter
- **Can't find file?** Use: `find /app -name "*.safetensors"`
- **Need to delete wrong files?** Use: `rm /path/to/file`
- **Need more space?** Check with: `df -h`

**That's it! Model upload takes 5-10 minutes total. ✨**
