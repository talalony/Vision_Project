# Pseudo-Labeling YOLO for Video with Minimal Annotations

This repository implements an iterative pseudo-labeling pipeline for object detection in video, using YOLOv8 and a custom annotation GUI. It minimizes manual labeling by generating pseudo-labels on video frames, correcting them via a GUI, and retraining the model in a loop.

## Repository Structure

- `predict.py`  
  Performs inference on a single image using a YOLO checkpoint. Outputs bounding boxes and confidence scores.

- `video.py`  
  Streams inference on a video file. Includes a **save frame** feature—press **S** during playback to dump the current frame to disk.

- `yolo_annotation_editor.py`  
  A Pygame-based GUI that:
  1. Loads a video (or image sequence) frame by frame.  
  2. Runs YOLO prediction on each frame.  
  3. Allows manual correction (add/move/delete boxes).  
  4. Saves images and YOLO-format labels for retraining.

- `setup.py`  
  Automates environment setup and dependency installation (GPU-enabled PyTorch, OpenCV, Ultralytics YOLOv8, etc.). Simply run:
  ```bash
  python setup.py
  ```
  Then activate the created environment with:
  ```bash
  conda activate yolo_env
  ```

- `requirements.txt`  
  Lists Python packages needed for non-conda installations (e.g. via `pip install -r requirements.txt`).

- `install_conda.py` (alias of `setup.py`)  
  Conda-based installer that creates `yolo_env` with Python 3.10, CUDA-enabled PyTorch, and other dependencies.

## Installation

Choose one of the following methods:

### 1. Conda (recommended)
```bash
# Run installer script
python setup.py

# Activate environment
conda activate yolo_env
```

### 2. Virtual environment + pip
```bash
# Create venv and install
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
```

## Usage Examples

### Image Inference
```bash
python predict.py \
  --model_path orig_best.pt \
  --image_path path/to/image.png \
  --conf_thresh 0.5
```

### Video Inference & Frame Saving
```bash
python video.py \
  --model_path orig_best.pt \
  --video_path path/to/video.mp4 \
  --save_dir saved_frames
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

## Report
A LaTeX report (`report.tex`) is included that documents:
1. Exploratory Data Analysis  
2. Experiments and Training Metrics  
3. Discussion and Conclusions

Compile with:
```bash
pdflatex report.tex
pdflatex report.tex
```

---

If you encounter any issues, please ensure your paths and environment variables (CUDA, Conda) are correctly configured. Happy annotating!
