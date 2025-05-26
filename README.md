# Pseudo-Labeling YOLO for Video with Minimal Annotations

This repository implements an iterative pseudo-labeling pipeline for object detection in video, using YOLO11 and a custom annotation GUI. It minimizes manual labeling by generating pseudo-labels on video frames, correcting them via a GUI, and retraining the model in a loop.

## Repository Structure

- `predict.py`  
  Performs inference on a single image using a YOLO checkpoint. Visualizes the image with the bounding-boxes.

- `video.py`  
  Performs inference on a video file using a YOLO checkpoint. Visualizes the video with the bounding-boxes.

- `yolo_annotation_editor.py`  
  A Pygame-based GUI that:
  1. Loads a video (or image sequence) frame by frame.  
  2. Runs YOLO prediction on each frame.  
  3. Allows manual correction (add/move/delete boxes).  
  4. Saves images and YOLO-format labels for retraining.

- `setup.py`  
  Automates environment setup and dependency installation (GPU-enabled PyTorch, OpenCV, Ultralytics YOLO11, etc.). Supports both conda and pip installations.

- `requirements.txt`  
  Lists conda packages needed for conda-based installation.

- `requirements_pip.txt`  
  Lists pip packages needed for pip-based installation.

## Installation

Choose one of the following methods:

### 1. Automated Setup (Recommended)

#### Using Conda (Default)
```bash
# Run installer script with conda
python setup.py

# Or explicitly specify conda
python setup.py --conda

# Activate environment
conda activate yolo_env_project
```

#### Using Pip with Virtual Environment
```bash
# Install with pip in a virtual environment
python setup.py --pip

# Activate virtual environment (Windows)
yolo_env_project\Scripts\activate

# Activate virtual environment (Linux/macOS)
source yolo_env_project/bin/activate
```

### 2. Manual Installation

#### Conda Environment Import
```bash
# Create an environment using the provided requirements file
conda create -n yolo_env_project -f requirements.txt

# Activate the environment
conda activate yolo_env_project
```

#### Manual Pip Installation
```bash
# Create virtual environment (Windows)
py -3.11 -m venv yolo_env_project
# OR (if py launcher not available)
python -m venv yolo_env_project

# Create virtual environment (Linux/macOS)
python3.11 -m venv yolo_env_project

# Activate virtual environment
# Windows:
yolo_env_project\Scripts\activate
# Linux/macOS:
source yolo_env_project/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python pygame tqdm ultralytics numpy pillow pyyaml
```

## Usage Examples

### Image Inference
```bash
python predict.py \
  --model_path path/to/best.pt \
  --image_path path/to/image.png \
```

### Video Inference & Frame Saving
```bash
python video.py \
  --model_path path/to/best.pt \
  --video_path path/to/video.mp4 \
  --save_video
```
Press **space** during playback to stop the video at the current frame.

### Annotation Editor
```bash
python yolo_annotation_editor.py
```

**Keybindings:**
- `d` / `a`: Next / Previous frame
- `b`: Enter "draw" mode to add a new box
- `e`: Enter "edit" mode to select, move or resize boxes
- `s`: Save current frame and its label file
- `x`: Delete the currently selected box
- `0-9`: Change class of selected box
- `q`: Quit the editor

---

## Downloading the Model

To download the pretrained YOLO checkpoint:
1. Click the **Releases** tab at the top of this repository.
2. Select the latest release.
3. Under the **Assets** section, click on **`best.pt`** to download the checkpoint file to your machine.
---

