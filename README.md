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
  Automates environment setup and dependency installation (GPU-enabled PyTorch, OpenCV, Ultralytics YOLO11, etc.).

- `requirements.txt`  
  Lists Python packages needed.

## Installation

Choose one of the following methods:

### 1. Conda (recommended)
```bash
# Run installer script
python setup.py

# Activate environment
conda activate yolo_env_project
```

### 2. Conda environment import
  ```bash
   # Create an environment using the provided requirements file
   conda create -n yolo_env_project -f requirements.txt

   # Activate the environment
   conda activate yolo_env_project
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

- **Web:** Click [here](https://github.com/talalony/Vision_Project/releases/download/V1.0/best.pt) to download **`best.pt`** immediately.  
- **CLI:** Run:
  ```bash
  wget https://github.com/talalony/Vision_Project/releases/download/V1.0/best.pt -O best.pt
  
---

