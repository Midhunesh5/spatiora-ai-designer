import base64
from io import BytesIO
import json
import torch
import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import logging
import threading

# --- Configuration ---
# Base model is loaded from Hugging Face Hub (auto-downloads and caches on first startup)
BASE_MODEL_PATH = "models/sd-v1-5-fp16"
# LoRA weight paths (local to RunPod container workspace)
FLOORPLAN_LORA_PATH = "models/floor_plan_model/lora_weights/floor_pytorch_lora_weights.safetensors"
INTERIOR_LORA_PATH = "models/interior_design/interior_gen_lora_weights/pytorch_lora_weights.safetensors"

# --- App Setup ---
app = Flask(__name__)
# Configure CORS: allow everything by default for testing; override via FRONTEND_ORIGIN env var
_allowed_origin = os.environ.get("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": _allowed_origin}}, supports_credentials=True)
app.config["CORS_HEADERS"] = "Content-Type,Authorization"
logging.basicConfig(level=logging.INFO)
# Load environment variables from .env (if present)
load_dotenv()
# --- JWT Configuration ---
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-this-secret")
jwt = JWTManager(app)

# --- MongoDB Configuration ---
# Set environment variable MONGO_URI or MONGODB_URI to your MongoDB Atlas connection string.
MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
if not MONGO_URI:
    logging.warning("MONGO_URI not set. Database features will be disabled until configured.")
    mongo_client = None
    db = None
else:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client.get_database("spatiora_db")
    users_coll = db.get_collection("users")
    creations_coll = db.get_collection("creations")
 
# --- Global Pipeline Variables ---
# Preload two separate pipelines to avoid dynamic LoRA switching at runtime
pipe_interior = None
pipe_floorplan = None
# Semaphore to allow up to 2 concurrent GPU inferences (keeps concurrency bounded)
inference_semaphore = threading.Semaphore(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
def _configure_pipeline(base_path: str, lora_path: str, torch_dtype, device: str):
    """Helper to load a pipeline and apply LoRA weights (if available)."""
    try:
        p = StableDiffusionPipeline.from_pretrained(
            base_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        p.scheduler = DPMSolverMultistepScheduler.from_config(
            p.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )

        try:
            p.enable_attention_slicing()
        except Exception:
            logging.debug("attention slicing not available")
        try:
            p.enable_vae_tiling()
        except Exception:
            logging.debug("VAE tiling not available")
        try:
            p.enable_xformers_memory_efficient_attention()
        except Exception:
            logging.debug("xformers not available; RTX 4090 has sufficient VRAM, no CPU offload needed")

        # Move to device
        p = p.to(device)

        # Attempt to load LoRA weights if present
        if lora_path and os.path.exists(lora_path) and hasattr(p, 'load_lora_weights'):
            try:
                p.load_lora_weights(lora_path)
                logging.info(f"Loaded LoRA weights from {lora_path}")
            except Exception as e:
                logging.warning(f"Failed to load LoRA {lora_path}: {e}")
        else:
            logging.warning(f"LoRA path not found or load not supported: {lora_path}")

        return p
    except Exception as e:
        logging.error(f"Failed to configure pipeline from {base_path}: {e}")
        raise


def load_models():
    """Loads two Stable Diffusion pipelines (interior + floorplan) into memory once at startup.
    
    The base model is downloaded from Hugging Face Hub (runwayml/stable-diffusion-v1-5) on first 
    startup and cached locally. If authentication is required, set HUGGINGFACE_HUB_TOKEN env var.
    Both pipelines are loaded with their respective LoRA weights and a warm-up inference runs.
    
    Returns tuple (interior_pipeline, floorplan_pipeline, device)
    """
    global pipe_interior, pipe_floorplan, DEVICE

    # Check for GPU and set the device
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        logging.info("CUDA (GPU) is available. Using GPU for faster generation.")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        logging.warning("CUDA (GPU) not available. Using CPU. Generation will be slower.")
    DEVICE = device

    logging.info("--- Starting Model Loading (two pipelines from Hugging Face) ---")
    try:
        # Load interior pipeline and apply interior LoRA
        logging.info("Loading interior pipeline...")
        pipe_interior = _configure_pipeline(BASE_MODEL_PATH, INTERIOR_LORA_PATH, torch_dtype, device)

        # Load floorplan pipeline and apply floorplan LoRA
        logging.info("Loading floorplan pipeline...")
        pipe_floorplan = _configure_pipeline(BASE_MODEL_PATH, FLOORPLAN_LORA_PATH, torch_dtype, device)

        logging.info("--- Both pipelines loaded successfully ---")

        # Warm-up inference to initialize CUDA kernels and caches
        try:
            logging.info("Running warm-up inference to mitigate cold start...")
            warmup_prompt = "warmup"
            gen_params = {"num_inference_steps": 1, "guidance_scale": 1.0, "num_images_per_prompt": 1}
            # Use the semaphore to ensure warmup uses the same concurrency control
            inference_semaphore.acquire()
            try:
                # Run warm-up on interior then floorplan (sequentially)
                pipe_interior(warmup_prompt, **gen_params)
                pipe_floorplan(warmup_prompt, **gen_params)
            finally:
                inference_semaphore.release()
            logging.info("Warm-up completed.")
        except Exception as e:
            logging.warning(f"Warm-up inference failed: {e}")

        return pipe_interior, pipe_floorplan, device
    except Exception as e:
        logging.error(f"--- Error loading models: {e} ---")
        logging.error("Please ensure the model paths are correct and you have downloaded all necessary files.")
        raise

# Load models when the application starts
load_models()

# Runtime LoRA switching is disabled. Two pipelines are preloaded: `pipe_interior` and `pipe_floorplan`.
def _switch_lora(tool_type: str):
    """Deprecated: runtime switching is not supported. This function exists for compatibility."""
    logging.debug("_switch_lora called but runtime switching is disabled; using preloaded pipelines.")

# --- Helper Functions ---
def _create_full_prompt(prompt: str, style: str, tool_type: str = 'interior') -> str:
    """Creates a detailed prompt for the model based on the tool type."""
    # Enhance the prompt for better results based on tool type
    if tool_type == 'floorplan':
        # This prompt is optimized for the floor plan LoRA
        return f"floorplan, architectural blueprint, 2d floor plan, top-down view, {prompt}, minimalist, black and white, technical drawing, clean lines, unlabeled, no text, no annotations."
    else: # Default to interior
        return f"{style} bedroom interior, {prompt}, photorealistic, high quality, detailed, no text, no watermark."

def _get_negative_prompt(user_negative_prompt: str, tool_type: str) -> str:
    """Creates a full negative prompt by combining a base prompt with a user-provided one."""
    base_negative_prompt = "blurry, low-quality, ugly, watermark, signature, deformed, text, letters, words, numbers, logo, ui, banner, caption, subtitle, label"
    if tool_type == 'floorplan':
        # For floorplans, we want to avoid photorealistic elements
        base_negative_prompt = "3d, photo, realistic, color, shadows, furniture, text, letters, words, numbers, logo, watermark, label"
    
    if user_negative_prompt:
        return f"{base_negative_prompt}, {user_negative_prompt}"
    return base_negative_prompt

def _get_generation_params(tool_type: str) -> dict:
    """Returns a dictionary of generation parameters based on the tool type."""
    if tool_type == 'floorplan':
        params = {
            "num_inference_steps": 30,
            "guidance_scale": 13.0
        }
        logging.info(f"Using floor plan specific parameters: {params}")
    else:  # Default for 'interior'
        params = {
            "num_inference_steps": 30,
            "guidance_scale": 11.0
        }
    if DEVICE == "cuda":
        if tool_type == 'interior':
            params.update({"width": 768, "height": 512})
        else:
            params.update({"width": 640, "height": 640})
    else:
        params.update({"width": 512, "height": 512})
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

# --- API Endpoints ---
@app.route('/')
def index():
    """A simple endpoint to check if the server is running."""
    return "<h1>Flask Server is Running!</h1><p>The AI models have been loaded. You can now send POST requests to the API endpoints.</p>"

@app.route('/generate-stream', methods=['POST'])
def generate_stream_endpoint() -> Response:
    """Handles all generation requests via streaming."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"success": False, "error": "Invalid request. 'prompt' is required."}), 400

        if pipe_interior is None or pipe_floorplan is None:
            return jsonify({"success": False, "error": "AI models are not available on the server."}), 503

        prompt = data['prompt']
        tool_type = data.get('tool_type', 'interior').lower().strip()
        negative_prompt = data.get('negative_prompt', '')
        num_images = data.get('num_images', 1)
        seed = data.get('seed')
        if not isinstance(num_images, int) or not 1 <= num_images <= 4:
            return jsonify({"success": False, "error": "Number of images must be an integer between 1 and 4."}), 400

        style = (data.get('style', '') or '').strip()

        if tool_type == 'floorplan':
            logging.info(f"Streaming {num_images} floor plan(s) with prompt: '{prompt}'")
        else: # Default to interior
            if not style:
                return jsonify({"success": False, "error": "'style' is required for interior design generation."}), 400
            logging.info(f"Streaming {num_images} interior image(s) with prompt: '{prompt}' and style: '{style}'")

        stream = generate_images_stream(prompt, style, tool_type, num_images, negative_prompt, seed=seed)
        return Response(stream, mimetype='text/event-stream')

    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_stream_endpoint: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

def generate_images_stream(prompt: str, style: str, tool_type: str, num_images: int, negative_prompt: str, steps: int = None, guidance: float = None, seed: int | None = None):
    """Yields generated images one by one for streaming, handling different tool types.

    Accepts optional `steps` and `guidance` to allow callers to control generation
    parameters for fair comparisons (e.g., benchmarks).
    """
    # Concurrency controlled via semaphore (inference_semaphore)
    if pipe_interior is None and pipe_floorplan is None:
        logging.error("Pipelines are not loaded. Cannot generate images.")
        raise RuntimeError("Pipelines are not loaded. Cannot generate images.")

    # Select preloaded pipeline
    try:
        if tool_type == 'floorplan':
            selected_pipe = pipe_floorplan
        else:
            selected_pipe = pipe_interior
        if selected_pipe is None:
            raise RuntimeError('Requested model pipeline is not loaded.')
    except Exception as e:
        logging.error(f"Failed to select pipeline: {e}")
        error_data = {"error": "Requested model pipeline is not available.", "details": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
        return

    full_prompt = _create_full_prompt(prompt, style, tool_type)
    full_negative_prompt = _get_negative_prompt(negative_prompt, tool_type)
    gen_params = _get_generation_params(tool_type)

    # Allow explicit override of steps/guidance when provided by the caller
    if steps is not None:
        gen_params['num_inference_steps'] = steps
    if guidance is not None:
        gen_params['guidance_scale'] = guidance

    if seed is None and tool_type == 'interior':
        seed = 3626764240

    try:
        logging.info(f"Streaming {num_images} image(s) for prompt: '{prompt}' using pipeline for '{tool_type}'")
        generator = None
        if seed is not None:
            # Use a single generator for a single forward pass
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Run the forward pass inside the semaphore to bound concurrent GPU usage
        inference_semaphore.acquire()
        try:
            outputs = selected_pipe(
                full_prompt,
                negative_prompt=full_negative_prompt,
                num_images_per_prompt=num_images,
                generator=generator,
                **gen_params
            ).images
        finally:
            inference_semaphore.release()

        for i, image in enumerate(outputs):
            try:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_url = f"data:image/png;base64,{img_str}"
                data_to_send = {"id": i, "url": data_url, "title": f"Generated Image {i+1}"}
                yield f"data: {json.dumps(data_to_send)}\n\n"
            except Exception as e:
                logging.error(f"Error processing generated image {i+1}: {e}")
                error_data = {"error": f"Failed to process image {i+1}", "details": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        error_data = {"error": "Failed to generate images.", "details": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"

# --- Auth and Creation Endpoints ---
from datetime import datetime


@app.route('/signup', methods=['POST'])
def signup():
    if db is None:
        return jsonify({"success": False, "error": "Database not configured."}), 500
    data = request.get_json() or {}
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')
    if not email or not password or not username:
        return jsonify({"success": False, "error": "'email', 'username' and 'password' are required."}), 400

    existing = users_coll.find_one({"email": email})
    if existing:
        return jsonify({"success": False, "error": "Email already registered."}), 400

    hashed = generate_password_hash(password)
    user_doc = {
        "email": email,
        "username": username,
        "password": hashed,
        "created_at": datetime.utcnow()
    }
    res = users_coll.insert_one(user_doc)
    user_id = str(res.inserted_id)
    access_token = create_access_token(identity=user_id)
    return jsonify({"success": True, "access_token": access_token, "user": {"id": user_id, "email": email, "username": username}}), 201


@app.route('/login', methods=['POST'])
def login():
    if db is None:
        return jsonify({"success": False, "error": "Database not configured."}), 500
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"success": False, "error": "'email' and 'password' are required."}), 400

    user = users_coll.find_one({"email": email})
    if not user or not check_password_hash(user.get('password', ''), password):
        return jsonify({"success": False, "error": "Invalid credentials."}), 401

    user_id = str(user.get('_id'))
    access_token = create_access_token(identity=user_id)
    return jsonify({"success": True, "access_token": access_token, "user": {"id": user_id, "email": user.get('email'), "username": user.get('username')}}), 200


@app.route('/update_profile', methods=['POST'])
@jwt_required()
def update_profile():
    if db is None:
        return jsonify({"success": False, "error": "Database not configured."}), 500
    user_id = get_jwt_identity()
    try:
        owner_oid = ObjectId(user_id)
    except Exception:
        return jsonify({"success": False, "error": "Invalid user identity."}), 400

    data = request.get_json() or {}
    new_username = data.get('username')
    new_email = data.get('email')
    new_password = data.get('password')

    update_fields = {}
    if new_username:
        update_fields['username'] = new_username
    if new_email:
        # Check uniqueness
        existing = users_coll.find_one({"email": new_email})
        if existing and str(existing.get('_id')) != user_id:
            return jsonify({"success": False, "error": "Email already in use."}), 400
        update_fields['email'] = new_email
    if new_password:
        update_fields['password'] = generate_password_hash(new_password)

    if not update_fields:
        return jsonify({"success": False, "error": "No changes provided."}), 400

    users_coll.update_one({"_id": owner_oid}, {"$set": update_fields})
    user = users_coll.find_one({"_id": owner_oid})
    user_response = {"id": str(user.get('_id')), "email": user.get('email'), "username": user.get('username')}
    return jsonify({"success": True, "user": user_response}), 200


@app.route('/save_creation', methods=['POST'])
@jwt_required()
def save_creation():
    if db is None:
        return jsonify({"success": False, "error": "Database not configured."}), 500
    user_id = get_jwt_identity()
    try:
        owner_oid = ObjectId(user_id)
    except Exception:
        return jsonify({"success": False, "error": "Invalid user identity."}), 400

    data = request.get_json() or {}
    title = data.get('title') or 'Untitled'
    image_data = data.get('image_data')  # Expect full data URL or base64 string
    metadata = data.get('metadata', {})
    if not image_data:
        return jsonify({"success": False, "error": "'image_data' is required."}), 400

    creation_doc = {
        "owner_id": owner_oid,
        "title": title,
        "image_data": image_data,
        "metadata": metadata,
        "created_at": datetime.utcnow()
    }
    res = creations_coll.insert_one(creation_doc)
    return jsonify({"success": True, "creation_id": str(res.inserted_id)}), 201


@app.route('/my_creations', methods=['GET'])
@jwt_required()
def my_creations():
    if db is None:
        return jsonify({"success": False, "error": "Database not configured."}), 500
    user_id = get_jwt_identity()
    try:
        owner_oid = ObjectId(user_id)
    except Exception:
        return jsonify({"success": False, "error": "Invalid user identity."}), 400

    docs = list(creations_coll.find({"owner_id": owner_oid}).sort("created_at", -1))
    results = []
    for d in docs:
        results.append({
            "id": str(d.get('_id')),
            "title": d.get('title'),
            "image_data": d.get('image_data'),
            "metadata": d.get('metadata', {}),
            "created_at": d.get('created_at')
        })

    return jsonify({"success": True, "creations": results}), 200

# --- Run the App ---
if __name__ == '__main__':
    # Use Gunicorn in production: gunicorn -w 1 -b 0.0.0.0:5000 backend:app
    # This fallback is for local testing only; debug must be False
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)