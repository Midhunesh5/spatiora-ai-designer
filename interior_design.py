from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
from pathlib import Path
import time

# --- Configuration ---
BASE_MODEL_PATH = "models/sd-v1-4-local1"
LORA_WEIGHTS_PATH = "models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors"
OUTPUT_DIR = "outputs"
SEED = 1024

PROMPT = (
    "photorealistic interior design of a modern full house, minimalistic, bright colors, "
    "large windows, indoor plants, cozy atmosphere with abundant natural light, "
    "predominantly white color scheme, 8k, high quality, sharp focus"
)
NEGATIVE_PROMPT = "low quality, blurry, worst quality, ugly, deformed, watermark, text, signature"

# --- Setup ---

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Enable cuDNN benchmark for speed
torch.backends.cudnn.benchmark = True

# Automatically select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pipeline with safetensors (torch_dtype set to float16 for GPU)
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None  # Disable safety checker if not needed
).to(device)

# Use DPMSolverMultistepScheduler for better image quality
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable memory-efficient attention
if device == "cuda":
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory efficient attention enabled.")
    except (ImportError, ModuleNotFoundError):
        print("xformers not installed. Falling back to sequential CPU offload.")
        pipe.enable_sequential_cpu_offload()

# Optional: Load LoRA weights (if you have them)
if os.path.exists(LORA_WEIGHTS_PATH):
    pipe.load_lora_weights(LORA_WEIGHTS_PATH)
    print(f"✅ LoRA weights loaded from {LORA_WEIGHTS_PATH}")
else:
    print(f"⚠️ LoRA weights not found at {LORA_WEIGHTS_PATH}, continuing without them.")

# Set seed for reproducibility
generator = torch.Generator(device).manual_seed(SEED)

# Generate image
with torch.autocast(device):
    image = pipe(
        PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

# Save output
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_filename = f"output_{SEED}_{timestamp}.png"
output_path = os.path.join(OUTPUT_DIR, output_filename)
image.save(output_path)
print(f"✅ High-quality image generated and saved as {output_path}")
