# RunPod GPU Pod optimization: PyTorch + CUDA 12.1
FROM runpod/pytorch:2.1-cuda12.1-devel-ubuntu22.04

# Prevent buffering for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (no cache to reduce size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY backend.py .
COPY frontend/ ./frontend/

# Create directories for LoRA weights (will be mounted from RunPod persistent volume)
RUN mkdir -p /app/models/floor_plan_model/lora_weights /app/models/interior_design/interior_gen_lora_weights

# Set environment variables
ENV FLASK_APP=backend.py
ENV PYTHONUNBUFFERED=1

# Health check: confirm the app is running (with longer grace period for model download)
HEALTHCHECK --interval=30s --timeout=60s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Expose port 5000 for RunPod HTTP proxy
EXPOSE 5000

# Production entrypoint: Gunicorn with single worker (no debug mode)
# Base model downloads from HuggingFace automatically on first startup
# --timeout 300: Generous timeout for long inference tasks
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "backend:app"]
