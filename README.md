# Spatiora AI Designer

Spatiora is a web-based application that uses AI to generate interior designs and architectural floor plans based on user descriptions. It leverages Stable Diffusion with specialized LoRA models to produce high-quality visual concepts.

![image](https://user-images.githubusercontent.com/12345/your-image-link-here.png) <!-- Replace with a screenshot of your app -->

## Features

- **Interior Design Generator**: Create photorealistic images of interior spaces from a text prompt and a style choice (e.g., Modern, Minimalist, Rustic).
- **Floor Plan Generator**: Generate 2D architectural blueprints from a description of a layout (e.g., "a 2-bedroom apartment with an open kitchen").
- **Web-Based Interface**: An intuitive, single-page application built with HTML, TailwindCSS, and React.
- **Streaming Results**: Images are streamed to the user as they are generated for a better user experience.
- **Efficient Backend**: A Flask-based backend that dynamically loads AI models to conserve memory.

## Project Structure

```
spatiora-ai-designer/
├── models/             # (Not in Git) AI models are downloaded here.
├── backend.py          # Flask server and AI logic.
├── index.html          # Frontend single-page application.
├── requirements.txt    # Python dependencies.
└── README.md           # This file.
```

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Midhunesh5/spatiora-ai-designer.git
cd spatiora-ai-designer
```

### 2. Set Up a Python Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Download the AI Models

The AI models are not included in this repository due to their large size. You must download them manually.

1.  Create a `models` directory in the root of the project.
2.  **Base Model**: Download the Stable Diffusion v1.4 model files and place them in `models/sd-v1-4-local1/`. You can get them from Hugging Face.
3.  **Interior Design LoRA**: Download `pytorch_lora_weights.safetensors` for interior design and place it in `models/interior_design/interior_gen_lora_weights/`.
4.  **Floor Plan LoRA**: Download `floor_pytorch_lora_weights.safetensors` for floor plans and place it in `models/floor_plan_model/lora_weights/`.

Your final `models` directory should look like this:
```
models/
├── sd-v1-4-local1/
│   ├── feature_extractor/
│   ├── safety_checker/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   └── model_index.json
├── interior_design/interior_gen_lora_weights/
│   └── pytorch_lora_weights.safetensors
└── floor_plan_model/lora_weights/
    └── floor_pytorch_lora_weights.safetensors
```

### 4. Run the Application

Open two separate terminals.

1.  **Run the Backend**:
    ```bash
    python backend.py
    ```
2.  **Open the Frontend**:
    Open the `index.html` file in your web browser.

The application will be running and accessible.
