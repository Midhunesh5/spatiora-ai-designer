# Spatiora AI Designer

Spatiora is a web-based application that uses AI to generate interior designs and architectural floor plans from user descriptions. It leverages a Stable Diffusion base model enhanced with specialized LoRA (Low-Rank Adaptation) models to produce high-quality, stylized visual concepts.

![Spatiora AI Screenshot](https://user-images.githubusercontent.com/12345/your-image-link-here.png) <!-- TODO: Replace with a screenshot of your app -->

## Features

- **Interior Design Generator**: Create photorealistic images of interior spaces from a text prompt and a style choice (e.g., Modern, Minimalist, Rustic).
- **Floor Plan Generator**: Generate 2D architectural blueprints from a description of a layout (e.g., "a 2-bedroom apartment with an open kitchen").
- **Intuitive Web Interface**: A responsive, single-page application built with React and TailwindCSS for a modern user experience.
- **Streaming Results**: Images are streamed to the user as they are generated for a better user experience.
- **Memory-Efficient Backend**: A Flask backend that dynamically loads and unloads LoRA models on-demand, significantly reducing VRAM/RAM usage.
- **GPU & CPU Support**: Automatically detects and uses a CUDA-enabled GPU for fast generation, with a fallback to CPU.

## Project Structure

```
spatiora-ai-designer/
├── models/
│   ├── sd-v1-4-local1/           # Base Stable Diffusion model
│   ├── floor_plan_model/         # Floor plan LoRA model
│   └── interior_design/          # Interior design LoRA model
├── outputs/                      # Directory for images from standalone scripts
├── backend.py                    # Flask server with API endpoints
├── floor_plan.py                 # Standalone script for testing floor plan generation
├── interior_design.py            # Standalone script for testing interior design generation
├── index.html                    # The main frontend file (HTML, CSS, React JSX)
└── README.md                     # Project documentation
```

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.8+
- A CUDA-compatible GPU is highly recommended for reasonable generation speeds.
- `git` for cloning the repository.

### 1. Clone the Repository

```bash
git clone https://github.com/Midhunesh5/spatiora-ai-designer.git
cd spatiora-ai-designer
