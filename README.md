# Capstone Project: Virtual Interior Redesign

A complete end-to-end AI-powered pipeline for virtual interior redesign, transforming furnished room photos into realistically rendered, re-staged spaces with new furniture layouts and styles.

## 🎯 Overview

This project delivers a comprehensive three-stage pipeline:
1. **Furniture Removal** - Intelligent segmentation and inpainting to create empty room images
2. **Furniture Selection & Layout Planning** - AI-driven furniture selection within budget constraints and optimal placement
3. **Photorealistic Rendering** - Structure-aware, style-controlled final rendering using ControlNet and Stable Diffusion

---

## 🏗️ Architecture

### Stage 1: Furniture Removal (Segmentation + Inpainting)

**Input:** Indoor image with existing furniture

**Process:**
- **Segmentation:** SegFormer (transformer-based semantic segmentation model fine-tuned on ADE20K indoor scenes) identifies furniture objects
- **Masking:** Furniture regions are converted into precise binary masks
- **Inpainting:** Stable Diffusion Inpainting removes furniture and reconstructs the background

**Output:** Empty-room image that preserves room structure and lighting

**Key File:** `stage1_clutter removal/rcsd.py`

---

### Stage 2: Furniture Selection and Layout Planning

**Input:** 
- Empty room image (from Stage 1)
- User preferences: budget, room type, style preference
- 3D-Future dataset (furniture models and metadata)

**Process:**
- **Selection:** Optimization module filters and selects suitable furniture items within budget
- **Layout Planning:** Vision-Language Model (VLM) proposes furniture layout aligned with room geometry
- **Draft Visualization:** OpenCV generates draft image with initial furniture placements

**Output:** 
- `composed_room.jpg` - Draft layout with furniture
- `selection.json` - Selected furniture list

**Key Files:** 
- `stage2_furniture selection/run.py` - Main entry point
- `stage2_furniture selection/furniture_select/select.py` - Selection logic
- `stage2_furniture selection/furniture_place/place.py` - Layout generation

---

### Stage 3: Room Rendering (Structure- and Style-Guided)

**Input:**
- Empty room image (from Stage 1)
- Draft furniture layout (from Stage 2)

**Process:**
- **Preprocessing:** Unify resolution and color balance
- **Enhanced Mask Generation:** 
  - LAB color space for lighting robustness
  - Otsu adaptive thresholding
  - Connected component analysis for noise removal
- **Structure Extraction:** Canny edge detection captures room geometry
- **Structure Conditioning:** ControlNet guides Stable Diffusion to preserve layout while applying target style
- **Harmonization:** Optional img2img refinement for enhanced detail

**Output:**
- `furnished_room.png` - Main rendered result
- `edge_map.png` - Canny edge detection visualization
- `mask.png` - Furniture mask for debugging
- `furnished_room_harmonized.png` - Enhanced version with improved details

**Key File:** `stage3_room rendering/furnishing.py`

**Recent Improvements:**
- LAB color space for better lighting invariance
- Otsu adaptive thresholding (vs. fixed threshold)
- Region constraints (ignores walls/ceiling)
- Connected component analysis removes noise
- Dual Gaussian blur for smoother edges
- Fixed img2img pipeline bug

---

## 📁 Repository Structure

```
GenAI-Virtual-Staging/
├── README.md                          # This file
│
├── stage1_clutter removal/            # Stage 1: Furniture removal
│   └── rcsd.py                        # Segmentation + inpainting script
│
├── stage2_furniture selection/        # Stage 2: Furniture selection & layout
│   ├── run.py                         # Main entry point
│   ├── requirements.txt               # Python dependencies
│   ├── data/                          # Furniture database
│   │   ├── model_infos.json          # Furniture metadata
│   │   └── modern_images/            # 2445 furniture images
│   ├── furniture_select/             # Selection module
│   │   ├── select.py                 # Selection logic
│   │   └── stage.ipynb               # Interactive notebook
│   ├── furniture_place/              # Layout module
│   │   └── generate_views.py        # Layout generation
│   └── inputs/                       # User inputs
│       ├── empty_room.jpg            # Empty room image
│       └── furniture.json            # Configuration
│
├── stage3_room rendering/            # Stage 3: Photorealistic rendering
│   ├── furnishing.py                 # Main rendering script 
│   └── Sample Data/                  # Example inputs
│       ├── empty_room.png            # Empty room sample
│       └── crude_image.png           # Draft layout sample
│
└── front_end/                        # Web interface (optional)
    ├── api_server.py                 # FastAPI backend
    ├── worker_server.py              # Task processing worker
    ├── app.py                        # Flask app (legacy)
    ├── requirements.txt              # Backend dependencies
    ├── package.json                  # Frontend dependencies
    └── src/                          # React components
```

---

## 🚀 Getting Started

### Prerequisites

- **OS:** macOS, Linux, or Windows
- **Python:** 3.9–3.11
- **GPU:** NVIDIA GPU with CUDA (highly recommended for Stage 1 & 3)
- **RAM:** 16GB+ recommended
- **Storage:** ~10GB for models

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rachel560lu/GenAI-Virtual-Staging.git
cd GenAI-Virtual-Staging

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies (choose based on stage)
# For Stage 2 (no GPU required)
pip install -r "stage2_furniture selection/requirements.txt"

# For Stage 1 & 3 (GPU required)
pip install torch torchvision diffusers transformers accelerate pillow opencv-python numpy
```

---

## 💻 Setup Instructions

### Stage 1: Furniture Removal

**Requirements:** 
- CUDA GPU
- Python 3.9–3.11
- PyTorch with CUDA support

**Installation:**
```bash
cd "stage1_clutter removal"

# Install dependencies
pip install torch torchvision diffusers transformers accelerate pillow opencv-python numpy
```

**Usage:**
```bash
# Place your room image at: input/room.jpeg
python rcsd.py
```

**Configuration** (edit `rcsd.py`):
```python
input_image_path = "input/room.jpeg"     # Input image path
output_dir = "output"                     # Output directory
max_iterations = 3                        # Number of iterations
```

**Output:** 
- `output/final_empty_room.png` - Main cleaned room image
- `output/mask_iter_N.png` - Masks for each iteration
- `output/image_iter_N.png` - Intermediate results

---

### Stage 2: Furniture Selection and Layout Planning

**Requirements:** 
- Python 3.9–3.11
- No GPU required

**Installation:**
```bash
cd "stage2_furniture selection"

# Install dependencies
pip install -r requirements.txt
```

**Usage:**
```bash
# Place empty room image at: inputs/empty_room.jpg
python run.py
```

**Configuration** (edit `run.py`):
```python
budget_cny = 6000.0              # Budget in CNY
style = "modern"                 # Style: modern, classic, etc.
room_type = "living room"        # Room type
room_size_m = (6.0, 5.0)        # Room dimensions in meters
```

**Input:**
- `inputs/empty_room.jpg` - Empty room image (typically from Stage 1)

**Output:** 
- `furniture_place/composed_room.jpg` - Draft layout with furniture
- `furniture_select/selection.json` - Selected furniture list
- `furniture_place/layout_results.json` - Layout coordinates
- `furniture_place/message.json` - Processing messages

---

### Stage 3: Photorealistic Rendering

**Requirements:** 
- CUDA GPU
- Python 3.9–3.11
- PyTorch with CUDA support

**Installation:**
```bash
cd "stage3_room rendering"

# Install dependencies
pip install torch torchvision diffusers transformers accelerate pillow opencv-python numpy
```

**Usage:**
```bash
# Place input files in the same directory:
# - empty_room.png (from Stage 1)
# - crude_image.png (from Stage 2)
python furnishing.py
```

**Configuration** (edit `furnishing.py`):
```python
# Mask generation threshold (line 84)
min_th = 60                      # Minimum threshold (50-80)

# Rendering quality (line 159)
num_inference_steps = 25         # More steps = better quality
guidance_scale = 4.5             # Higher = more prompt adherence
```

**Input Files:**
- `empty_room.png` - Empty room image (from Stage 1)
- `crude_image.png` - Draft furniture layout (from Stage 2)

**Output Files:**
- `furnished_room.png` - Main rendered result
- `edge_map.png` - Canny edge detection visualization (for debugging)
- `mask.png` - Furniture mask (for debugging)
- `furnished_room_harmonized.png` - Enhanced version with improved details

---

### Web Interface (Optional)

**Requirements:**
- Redis server
- Node.js and npm
- All dependencies from Stages 1-3

**Installation:**
```bash
cd front_end

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

**Usage:**
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start API Server
cd front_end
python api_server.py

# Terminal 3: Start Worker
cd front_end
python worker_server.py

# Terminal 4: Start Frontend
cd front_end
npm run dev
```

**Access:** Open browser at `http://localhost:3000`

---


## 📊 Data and Models

### Datasets
- **ADE20K:** Indoor scene segmentation dataset
- **3D-Future:** 2445+ furniture models with metadata and images

### AI Models
- **SegFormer:** Semantic segmentation (ADE20K fine-tuned)
- **Stable Diffusion v1.5:** Inpainting and rendering
- **ControlNet (Canny):** Structure-guided generation
- **Vision-Language Model:** Layout planning

### Model Downloads
Models are automatically downloaded on first run:
- SegFormer: `nvidia/segformer-b3-finetuned-ade-512-512`
- Stable Diffusion: `runwayml/stable-diffusion-v1-5`
- ControlNet: `lllyasviel/sd-controlnet-canny`

**Storage Required:** ~5-10 GB

---

## 🔧 Configuration

### Stage 2 Parameters (`run.py`)

```python
budget_cny = 6000.0              # Total budget in CNY
style = "modern"                 # Furniture style
room_type = "living room"        # Room type
room_size_m = (6.0, 5.0)        # Width x Height in meters
```

### Stage 3 Parameters (`furnishing.py`)

```python
# Mask generation (line 84)
min_th = 60                      # Minimum threshold (50-80)

# Region constraints (line 90)
mask_fg[:h//3, :] = 0           # Ignore top 1/3

# Noise removal (line 100)
min_area = (h*w)//400           # Minimum area threshold

# Rendering quality (line 159)
num_inference_steps = 25         # More steps = better quality
guidance_scale = 4.5             # Higher = more prompt adherence
```

---

## 🎨 Features

- ✅ **Intelligent Furniture Removal** - Preserves room structure
- ✅ **Budget-Aware Selection** - Optimizes within constraints
- ✅ **Style-Guided Layout** - Matches user preferences
- ✅ **Photorealistic Rendering** - ControlNet-guided generation
- ✅ **Enhanced Mask Generation** - LAB color space + Otsu thresholding
- ✅ **Web Interface** - User-friendly React frontend
- ✅ **Async Processing** - Redis-based task queue
- ✅ **Comprehensive Testing** - Detailed guides and examples

---

## ⚠️ Limitations

- **Inpainting Quality:** Depends on mask accuracy and background complexity
- **GPU Requirement:** Stages 1 & 3 require NVIDIA GPU with CUDA
- **Processing Time:** 2-5 minutes per image on GPU, much slower on CPU



---



**Hardware Tested:**
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CPU: Apple M1 Pro
- RAM: 16GB

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 Recent Updates

### Version 2.0 (November 2025)
- ✨ Enhanced Stage 3 mask generation with LAB color space
- 🐛 Fixed img2img pipeline initialization bug
- 📊 Added connected component analysis for noise removal
- 🎨 Improved edge blending with dual Gaussian blur
- 🌐 Added React-based web interface
- ⚡ Implemented Redis-based async processing


---

## 🙏 Acknowledgements

- **ADE20K Dataset:** MIT Scene Parsing Benchmark
- **SegFormer:** NVIDIA Research
- **Stable Diffusion:** Stability AI & RunwayML
- **ControlNet:** Lvmin Zhang et al.
- **3D-Future Dataset:** Furniture model providers
- **Diffusers Library:** Hugging Face

---

## 📄 License

Please see the repository license or add one if missing. Ensure compliance with the licenses of third-party models and datasets used.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the project maintainer： rachel560lv@gmail.com.

---

## 🔗 Links

- **Repository:** https://github.com/Rachel560lu/GenAI-Virtual-Staging

---

**Built with ❤️ for virtual interior design**
