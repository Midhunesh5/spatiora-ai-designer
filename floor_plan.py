from diffusers import StableDiffusionPipeline
import torch
import sys
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_MODEL_PATH = "models/sd-v1-4-local1"
LORA_PATH = "models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_IMAGE_PATH = "floorplan_generated.png"
SEED = 42  # Use a specific seed for reproducible results
# --- Prompt Engineering ---
# This prompt is concise to avoid truncation (max 77 tokens) and focuses on key elements.
PROMPT = (
    "floorplan, architectural blueprint, 2d floor plan, top-down view, "
    "modern 2BHK apartment, minimalist, black and white, technical drawing, "
    "labeled rooms, clean lines."
)
NEGATIVE_PROMPT = (
    "photorealistic, 3d render, exterior, building, house, photo, color, "
    "shadows, textures, furniture, people, watermark, text, blurry, jpg artifacts, "
    "messy lines, disproportionate, isometric, perspective view"
)

def load_sketch_pipeline():
    """Loads and configures the diffusion pipeline for sketch generation."""
    print("Loading standard pipeline for sketching...")
    sketch_pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16, safety_checker=None
    )
    if not os.path.exists(LORA_PATH):
        print(f"FATAL: LoRA weights not found at '{LORA_PATH}'")
        sys.exit(1)

    print("Loading LoRA weights and optimizing pipeline...")
    sketch_pipe.load_lora_weights(LORA_PATH)

    try:
        import xformers
        sketch_pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        print("xformers not available. For faster performance, install xformers.")
    sketch_pipe.enable_model_cpu_offload()
    print("Sketch pipeline loaded successfully!")
    return sketch_pipe

def main():
    sketch_pipe = load_sketch_pipeline()
    print(f"\n--- Starting floor plan generation with seed: {SEED} ---")

    # Use a generator for reproducible results
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    try:
        generated_sketch = sketch_pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=60,  # Increased for more detail
            guidance_scale=8.5,  # Increased for stricter prompt adherence
            generator=generator
        ).images[0]
        generated_sketch.save(OUTPUT_IMAGE_PATH)
        print(f"Saved generated sketch to: {os.path.abspath(OUTPUT_IMAGE_PATH)}")
    except Exception as e:
        print(f"Error during generation: {e}")
    print("\n--- Generation complete. ---")

if __name__ == "__main__":
    main()
