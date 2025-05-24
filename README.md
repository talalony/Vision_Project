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
  Automates environment setup and dependency installation (GPU-enabled PyTorch, OpenCV, Ultralytics YOLO11, etc.). Simply run:
  ```bash
  python setup.py
  ```
  Then activate the created environment with:
  ```bash
  conda activate yolo_env_project
  ```

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
Press **S** during playback to save the current frame.

### Annotation GUI
```bash
python yolo_annotation_editor.py \
  --data_path path/to/video_or_images \
  --input_type video     # or 'images' for a folder of frames
  --model_path orig_best.pt \
  --conf_thresh 0.5 \
  --out_img_dir output/images \
  --out_lbl_dir output/labels
```
---
