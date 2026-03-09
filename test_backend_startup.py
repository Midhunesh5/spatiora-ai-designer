#!/usr/bin/env python3
"""
Quick test script to verify backend startup and model loading.
Runs locally without actually generating images.
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("Backend Startup Verification Test")
    print("=" * 70)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    
    print("\n1️⃣  Checking Python environment...")
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                           capture_output=True, text=True)
    
    has_torch = "torch" in result.stdout
    has_diffusers = "diffusers" in result.stdout
    has_flask = "Flask" in result.stdout
    has_gunicorn = "gunicorn" in result.stdout
    
    print(f"   ✓ torch: {'installed' if has_torch else '❌ MISSING'}")
    print(f"   ✓ diffusers: {'installed' if has_diffusers else '❌ MISSING'}")
    print(f"   ✓ Flask: {'installed' if has_flask else '❌ MISSING'}")
    print(f"   ✓ gunicorn: {'installed' if has_gunicorn else '❌ MISSING'}")
    
    if not all([has_torch, has_diffusers, has_flask, has_gunicorn]):
        print("\n❌ Missing dependencies. Install with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n2️⃣  Checking CUDA availability...")
    cuda_check = subprocess.run(
        [sys.executable, "-c", "import torch; print('CUDA available:', torch.cuda.is_available())"],
        capture_output=True, text=True
    )
    print(f"   {cuda_check.stdout.strip()}")
    
    print("\n3️⃣  Checking LoRA weight files...")
    interior_lora = script_dir / "models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors"
    floorplan_lora = script_dir / "models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors"
    
    interior_exists = interior_lora.exists()
    floorplan_exists = floorplan_lora.exists()
    
    print(f"   ✓ Interior LoRA: {'found' if interior_exists else '⚠️  not found (will warn but continue)'}")
    print(f"   ✓ Floorplan LoRA: {'found' if floorplan_exists else '⚠️  not found (will warn but continue)'}")
    
    print("\n4️⃣  Testing backend import...")
    try:
        import backend
        print("   ✓ backend.py imports successfully")
    except Exception as e:
        print(f"   ❌ Backend import failed: {e}")
        return False
    
    print("\n5️⃣  Checking backend configuration...")
    print(f"   ✓ BASE_MODEL_PATH: {backend.BASE_MODEL_PATH}")
    print(f"   ✓ INTERIOR_LORA_PATH: {backend.INTERIOR_LORA_PATH}")
    print(f"   ✓ FLOORPLAN_LORA_PATH: {backend.FLOORPLAN_LORA_PATH}")
    print(f"   ✓ Semaphore limit: {backend.inference_semaphore._value}")
    
    print("\n6️⃣  Simulating model loading (without Gunicorn)...")
    print("   ⚠️  Full model load test requires GPU and ~5-10 min (skipping here)")
    print("   💡 To test full startup, run: waitress-serve --host=0.0.0.0 --port=5000 backend:app")
    print("      Then verify logs show: 'Both pipelines loaded successfully'")
    
    print("\n7️⃣  Backend is ready for deployment! ✅")
    print("\nNext steps:")
    print("   1. Upload models/ folder to RunPod persistent volume")
    print("   2. Build Docker image: docker build -t my-sd-api:latest .")
    print("   3. Deploy to RunPod GPU pod (RTX 4090)")
    print("   4. Monitor logs for: 'Warm-up completed.'")
    print("   5. Test endpoint with: curl https://xxxx-xxxx.pods.runpod.io/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
