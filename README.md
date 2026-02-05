# Hunyuan 3D 2.1 – Image to Textured 3D (Batch Script)

This repository contains a **custom batch / single-image script** based on the official  
**Hunyuan 3D 2.1 demo** by Tencent.

It converts one or more images into **3D meshes and textured 3D models (GLB / OBJ)** using the
official **shape** and **paint** pipelines, with additional quality-of-life improvements.

⚠️ **This repository does NOT include models or weights.**

---

## What this script does

- Processes **multiple images** from an `input/` folder (batch mode)
- Or processes **a single image** via command line (`--path`)
- Optional **automatic background removal** using U²-Net
- Generates:
  - a 3D shape mesh
  - a textured 3D model
- Creates **one output folder per image**
- Runs **fully headless** (no Blender, no `bpy` dependency)

---

## Requirements

### Hardware
- NVIDIA GPU (RTX class recommended)
- **24 GB VRAM recommended** (tested on RTX 3090)

### Software
- Python **3.10+**
- CUDA-enabled PyTorch
- Linux environment recommended

---

## Project structure

Your working directory must look like this:

```text
hy3d/
├─ models/
│  └─ Hunyuan3D-2.1/          # Shape model weights
│
├─ Hunyuan3D-2.1-main/        # Official GitHub code
│  ├─ hy3dshape/
│  ├─ hy3dpaint/
│  │  └─ ckpt/
│  │     └─ RealESRGAN_x4plus.pth
│  ├─ input/
│  ├─ output/
│  └─ z3v_official.py.py       #script
```

---

## Setup

### 1) Clone the official Hunyuan repository

```bash
git clone https://github.com/Tencent/Hunyuan3D-2.1.git
```

From this repository you must have:
- `hy3dshape/`
- `hy3dpaint/`

Place them inside the hy3d folder with name Hunyuan3D-2.1-main.

In the folder Hunyuan3D-2.1-main, insert the script from this repository.

---

### 2) Download the shape model (mandatory)

Download the **Hunyuan 3D 2.1 shape model weights** and place them here:

```text
models/Hunyuan3D-2.1/
```

The script loads the model **from a local path**, not from the Hugging Face hub.

Make sure the folder is complete and not empty.

---

### 3) Download RealESRGAN (required for paint)

The texture generation step requires **RealESRGAN_x4plus.pth**.

Download the file: 

https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

and place it here:

```text
Hunyuan3D-2.1-main/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
```

If this file is missing, the paint step may fail.

---

### 4) Install Python dependencies

Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required packages:

```bash
pip install torch torchvision numpy pillow tqdm trimesh
```

Additional dependencies may be required depending on your CUDA / PyTorch setup.
See the requirements file.
---

## Usage

### Batch mode (process all images in `input/`)

Put images in the `input/` folder.

Supported formats:
- PNG
- JPG / JPEG
- WEBP
- BMP
- TIFF

Example:

```text
input/
├─ object_01.png
├─ object_02.jpg
```

Run:

```bash
python z3v_official.py
```

---

### Single image mode

Process a single image directly:

```bash
python z3v_official.py --path you/path/image.png
```

---

### Background removal (optional)

Enable background removal:

```bash
python z3v_official.py --remove-bg
```

Force background removal even if the image already has an alpha channel:

```bash
python z3v_official.py --remove-bg --force-remove
```

> For backward compatibility, `--rimuovi-sfondo` (italian) is also supported as an alias.

if you have compatibility issues or errors, and want to make sure you're using the env you created, use the full path:
```bash
/home/user/hy3d/Hunyuan3D-2.1-main/path/to/venv/bin/python z3v_official.py
```
---

## Output

For each input image, the script generates:

```text
output/
└─ image_name/
   ├─ image_name_shape.glb
   ├─ image_name_textured.glb   (or .obj)
   └─ image_name_bgremoved.png  (if enabled)
```

### Note
The paint pipeline may output an **OBJ file with a `.glb` extension**.  
The script automatically detects this and renames the file correctly.

---

## Configuration

You can edit these values at the top of the script:

```python
MAX_NUM_VIEW = 6
RESOLUTION = 512
```

### Recommended values (RTX 3090)

```text
MAX_NUM_VIEW = 6-8
RESOLUTION = 512
```

Higher values may cause **out-of-memory (OOM)** errors or GPU instability.

---

## Notes

- The script bypasses Blender (`bpy`) and runs fully headless
- All credits for models and core pipelines go to the **Hunyuan 3D team**
- This repository provides **glue code and batch automation only**

---

## License

This script is released under the **MIT License**.

Original Hunyuan 3D code and models are subject to their respective licenses.
