import base64
from io import BytesIO
import json
import torch
import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import logging

# --- Configuration ---
BASE_MODEL_PATH = "E:/main_pwl/models/sd-v1-4-local1"
FLOORPLAN_LORA_PATH = "E:/main_pwl/models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors"
INTERIOR_LORA_PATH = "E:/main_pwl/models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors"

# --- App Setup ---
app = Flask(__name__)
CORS(app) # This enables Cross-Origin Resource Sharing for your frontend
logging.basicConfig(level=logging.INFO)

# --- Global Variables for Pipelines ---
base_pipe = None
floorplan_pipe = None

def load_models():
    """Loads Stable Diffusion models into memory."""
    global base_pipe, floorplan_pipe
    
    # Check for GPU and set the device
    if torch.cuda.is_available():
        device = "cuda"
        # Use float16 for faster inference and less memory usage on compatible GPUs
        torch_dtype = torch.float16 
        logging.info("CUDA (GPU) is available. Using GPU for faster generation.")
    else:
        device = "cpu"
        torch_dtype = torch.float32 # CPU works with float32
        logging.warning("CUDA (GPU) not available. Using CPU. Generation will be slower.")

    logging.info("--- Starting Model Loading ---")
    try:
        # 1. Load the base Stable Diffusion pipeline for interior design
        # Note: The FutureWarning about CLIPFeatureExtractor is from the transformers library,
        # used internally by diffusers. It's generally safe to ignore if functionality is not affected,
        # but keeping libraries updated is recommended.
        logging.info("Loading Base Stable Diffusion Model for Interior Design...")
        base_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch_dtype,
            safety_checker=None, # Disabling the safety checker can speed up inference
            requires_safety_checker=False
        )
        base_pipe = base_pipe.to(device)

        # --- Performance Optimizations for base_pipe ---
        base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)
        try:
            base_pipe.enable_xformers_memory_efficient_attention()
            logging.info("...xformers enabled for Interior Design pipeline.")
        except (ImportError, ModuleNotFoundError):
            logging.warning("xformers not available. Enabling model CPU offload as a fallback for VRAM optimization.")
            # This is a crucial fallback for systems without xformers or with limited VRAM.
            # It keeps the model on the CPU and only moves parts to the GPU when needed.
            base_pipe.enable_model_cpu_offload()
        logging.info("...Base Model Loaded Successfully.")
        
        # Load LoRA weights for the interior design pipeline
        if os.path.exists(INTERIOR_LORA_PATH):
            base_pipe.load_lora_weights(INTERIOR_LORA_PATH)
            logging.info("...Interior Design LoRA weights loaded successfully.")
        else:
            logging.warning(f"Interior Design LoRA not found at {INTERIOR_LORA_PATH}. The model will work, but results may be suboptimal.")

        # 2. Load a separate pipeline for floor plans and apply the LoRA
        logging.info("Loading Floor Plan Model...")
        floorplan_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        # Load the LoRA weights specifically for the floorplan pipeline
        floorplan_pipe.load_lora_weights(FLOORPLAN_LORA_PATH)

        # --- Performance Optimizations for floorplan_pipe ---
        floorplan_pipe.scheduler = DPMSolverMultistepScheduler.from_config(floorplan_pipe.scheduler.config)
        try:
            floorplan_pipe.enable_xformers_memory_efficient_attention()
            logging.info("...xformers enabled for Floor Plan pipeline.")
        except (ImportError, ModuleNotFoundError):
            logging.warning("xformers not available. Enabling model CPU offload as a fallback for VRAM optimization.")
            # This is a crucial fallback for systems without xformers or with limited VRAM.
            # It keeps the model on the CPU and only moves parts to the GPU when needed.
            floorplan_pipe.enable_model_cpu_offload()
        floorplan_pipe = floorplan_pipe.to(device)
        logging.info("...Floor Plan Model with LoRA Loaded Successfully.")


        logging.info("--- All Models Loaded ---")

    except Exception as e:
        logging.error(f"--- Error loading model: {e} ---")
        logging.error("Please ensure the model paths are correct and you have downloaded all necessary files.")
        # Exit if models cannot be loaded, as the app won't function
        exit(1)

# Load models when the application starts
load_models()

# --- Helper Functions ---
def _create_full_prompt(prompt: str, style: str, tool_type: str = 'interior') -> str:
    """Creates a detailed prompt for the model based on the tool type."""
    # Enhance the prompt for better results based on tool type
    if tool_type == 'floorplan':
        # This prompt is optimized for the floor plan LoRA
        return f"floorplan, architectural blueprint, 2d floor plan, top-down view, {prompt}, minimalist, black and white, technical drawing, labeled rooms, clean lines."
    else: # Default to interior
        return f"A beautiful, wide-angle shot of a {style} style room, focusing on the overall ambiance. The room features {prompt}. photorealistic, 8k, high quality, detailed, interior design, professional photograph"

def _get_negative_prompt(user_negative_prompt: str, tool_type: str) -> str:
    """Creates a full negative prompt by combining a base prompt with a user-provided one."""
    base_negative_prompt = "blurry, low-quality, ugly, watermark, signature, deformed"
    if tool_type == 'floorplan':
        # For floorplans, we want to avoid photorealistic elements
        base_negative_prompt = "3d, photo, realistic, color, shadows, furniture"
    
    if user_negative_prompt:
        return f"{base_negative_prompt}, {user_negative_prompt}"
    return base_negative_prompt

def _get_generation_params(tool_type: str) -> dict:
    """Returns a dictionary of generation parameters based on the tool type."""
    if tool_type == 'floorplan':
        params = {
            "num_inference_steps": 50,
            "guidance_scale": 12.0
        }
        logging.info(f"Using floor plan specific parameters: {params}")
    else:  # Default for 'interior'
        params = {
            "num_inference_steps": 25,
            "guidance_scale": 7.5
        }
    return params

def _convert_images_to_urls(images: list[Image.Image]) -> list[dict]:
    """Converts a list of PIL images to a list of data URLs."""
    image_urls = []
    for i, image in enumerate(images):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"
        image_urls.append({"id": i, "url": data_url, "title": f"Generated Image {i+1}"})

    return image_urls

# --- Image Generation Functions ---
def generate_interior_design_images(prompt: str, style: str, num_images: int = 1, negative_prompt: str = '') -> list[dict]:
    """Generates interior design images using the base Stable Diffusion model."""
    if not base_pipe:
        raise RuntimeError("Base model is not loaded. Cannot generate images.")
    
    tool_type = 'interior'
    full_prompt = _create_full_prompt(prompt, style, tool_type)
    full_negative_prompt = _get_negative_prompt(negative_prompt, tool_type)

    # --- Get generation parameters ---
    gen_params = _get_generation_params(tool_type)

    images = base_pipe(
        full_prompt,
        negative_prompt=full_negative_prompt,
        num_images_per_prompt=num_images,
        **gen_params
    ).images

    # Convert images to Base64 data URLs
    return _convert_images_to_urls(images)

def generate_floorplan_images(prompt: str, num_images: int = 1, negative_prompt: str = '') -> list[dict]:
    """Generates floor plan images using the base Stable Diffusion model."""
    if not floorplan_pipe:
        raise RuntimeError("Floor plan model is not loaded. Cannot generate images.")
    
    tool_type = 'floorplan'
    # Style is ignored for floorplans, so we pass an empty string.
    full_prompt = _create_full_prompt(prompt, style="", tool_type=tool_type)
    full_negative_prompt = _get_negative_prompt(negative_prompt, tool_type)
    gen_params = _get_generation_params(tool_type)
    
    images = floorplan_pipe(
        full_prompt,
        negative_prompt=full_negative_prompt,
        num_images_per_prompt=num_images,
        **gen_params
    ).images

    return _convert_images_to_urls(images)
 
def generate_interior_design_stream(prompt: str, style: str, num_images: int = 1, negative_prompt: str = ''):
    """Yields generated interior design images one by one for streaming."""
    if not base_pipe:
        logging.error("Base model is not loaded. Cannot generate images.")
        raise RuntimeError("Base model is not loaded. Cannot generate images.")
    
    tool_type = 'interior'
    full_prompt = _create_full_prompt(prompt, style, tool_type)
    full_negative_prompt = _get_negative_prompt(negative_prompt, tool_type)

    # --- Get generation parameters ---
    gen_params = _get_generation_params(tool_type)

    # To stream, we generate one image at a time in a loop.
    for i in range(num_images):
        try:
            logging.info(f"Streaming image {i+1}/{num_images} for prompt: '{prompt}'")
            image = base_pipe(full_prompt, negative_prompt=full_negative_prompt, num_images_per_prompt=1, **gen_params).images[0]
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_str}"
            
            # Format the data for Server-Sent Events (SSE)
            data_to_send = {"id": i, "url": data_url, "title": f"Generated Image {i+1}"}
            yield f"data: {json.dumps(data_to_send)}\n\n"
        except Exception as e:
            logging.error(f"Error generating image {i+1}: {e}")
            error_data = {"error": f"Failed to generate image {i+1}", "details": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

def generate_floorplan_stream(prompt: str, num_images: int = 1, negative_prompt: str = ''):
    """Yields generated floor plan images one by one for streaming."""
    if not floorplan_pipe:
        logging.error("Floor plan model is not loaded. Cannot generate images.")
        raise RuntimeError("Floor plan model is not loaded. Cannot generate images.")

    tool_type = 'floorplan'
    full_prompt = _create_full_prompt(prompt, style="", tool_type=tool_type)
    full_negative_prompt = _get_negative_prompt(negative_prompt, tool_type)
    gen_params = _get_generation_params(tool_type)

    for i in range(num_images):
        try:
            logging.info(f"Streaming image {i+1}/{num_images} for prompt: '{prompt}'")
            image = floorplan_pipe(full_prompt, negative_prompt=full_negative_prompt, num_images_per_prompt=1, **gen_params).images[0]
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_str}"
            
            data_to_send = {"id": i, "url": data_url, "title": f"Generated Image {i+1}"}
            yield f"data: {json.dumps(data_to_send)}\n\n"
        except Exception as e:
            logging.error(f"Error generating image {i+1}: {e}")
            error_data = {"error": f"Failed to generate image {i+1}", "details": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

# --- API Endpoints ---
@app.route('/')
def index():
    """A simple endpoint to check if the server is running."""
    return "<h1>Flask Server is Running!</h1><p>The AI models have been loaded. You can now send POST requests to the API endpoints.</p>"

@app.route('/generate-interior-design', methods=['POST'])
def generate_interior_design_endpoint() -> Response:
    """Handles interior design and floor plan generation requests."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"success": False, "error": "Invalid request. 'prompt' is required."}), 400

        if base_pipe is None or floorplan_pipe is None:
            return jsonify({"success": False, "error": "AI model is not available on the server."}), 503

        tool_type = data.get('tool_type', 'interior').lower().strip()
        num_images = data.get('num_images', 1)
        negative_prompt = data.get('negative_prompt', '')
        if not isinstance(num_images, int) or not 1 <= num_images <= 4:
            return jsonify({"success": False, "error": "Number of images must be an integer between 1 and 4."}), 400

        if tool_type == 'floorplan':
            logging.info(f"Generating {num_images} floor plan(s) with prompt: '{data['prompt']}'")
            generated_images = generate_floorplan_images(data['prompt'], num_images=num_images, negative_prompt=negative_prompt)
        else: # Default to interior
            style = data.get('style')
            if not style:
                return jsonify({"success": False, "error": "'style' is required for interior design generation."}), 400
            
            logging.info(f"Generating {num_images} interior image(s) with prompt: '{data['prompt']}' and style: '{style}'")
            generated_images = generate_interior_design_images(data['prompt'], style, num_images=num_images, negative_prompt=negative_prompt)
    
        return jsonify({"success": True, "images": generated_images})
    except RuntimeError as e:
        logging.error(f"Runtime error during image generation: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        # This will catch JSON parsing errors and other unexpected issues.
        logging.error(f"An unexpected error occurred in generate_interior_design_endpoint: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

@app.route('/generate-interior-design-stream', methods=['POST'])
def generate_interior_design_stream_endpoint() -> Response:
    """Handles interior design and floor plan generation requests via streaming."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"success": False, "error": "Invalid request. 'prompt' is required."}), 400

        if base_pipe is None or floorplan_pipe is None:
            return jsonify({"success": False, "error": "AI model is not available on the server."}), 503

        prompt = data['prompt']
        tool_type = data.get('tool_type', 'interior').lower().strip()
        negative_prompt = data.get('negative_prompt', '')
        num_images = data.get('num_images', 1)
        if not isinstance(num_images, int) or not 1 <= num_images <= 4:
            return jsonify({"success": False, "error": "Number of images must be an integer between 1 and 4."}), 400

        if tool_type == 'floorplan':
            logging.info(f"Streaming {num_images} floor plan(s) with prompt: '{prompt}'")
            stream = generate_floorplan_stream(prompt, num_images=num_images, negative_prompt=negative_prompt)
        else: # Default to interior
            style = data.get('style')
            if not style:
                return jsonify({"success": False, "error": "'style' is required for interior design generation."}), 400
            logging.info(f"Streaming {num_images} interior image(s) with prompt: '{prompt}' and style: '{style}'")
            stream = generate_interior_design_stream(prompt, style, num_images=num_images, negative_prompt=negative_prompt)
        
        return Response(stream, mimetype='text/event-stream')

    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_interior_design_stream_endpoint: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Runs the Flask app on http://127.0.0.1:5000
    app.run(debug=True, port=5000)