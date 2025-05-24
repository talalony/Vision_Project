"""
yolo_annotation_editor.py

YOLO Annotation Editor GUI
---------------------------

A graphical tool for YOLO-based pseudo-labeling with interactive frame-by-frame editing
of object bounding boxes on videos or image sequences.

Description:
    - Load either a video file or a directory of image frames.
    - Run an Ultralytics YOLO model to generate initial detections.
    - Navigate through frames and manually correct:
        - Draw new boxes
        - Move or resize existing boxes via draggable corners
        - Delete boxes
        - Reclassify boxes using number keys
    - Save corrected frames (for videos) and write YOLO-format .txt label files.
    - Optionally extract all frames from a video into an image directory.

Keybindings:
    - d / a      : Next / Previous frame
    - b          : Enter “draw” mode to add a new box
    - e          : Enter “edit” mode to select, move or resize boxes
    - s          : Save current frame and its label file
    - x          : Delete the currently selected box
    - 0-9        : Change class of selected box
    - q          : Quit the editor

:author: Tal Aloni
:date:   2025-05-16
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import cv2
import pygame
import torch
from ultralytics import YOLO

# Constants for drawing and UI
BOX_THICK = 2           # thickness of rectangle border
CORNER_SIZE = 10        # radius for corner handles
FONT_SIZE = 20          # font size for labels and HUD
FPS = 30                # frames per second for UI loop


def get_default_device():
    """
    Return the default torch device: GPU/MPS if available, else CPU.
    """
    return 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class YOLOTask:
    """
    Wrapper around a YOLO model for running inference on frames.
    """
    def __init__(self, model_path: str, conf_thresh: float = 0.5, device: str = None):
        """
        Initialize the YOLO model.

        Args:
            model_path: Path to the .pt YOLO weights file.
            conf_thresh: Confidence threshold for detections.
            device: Torch device string, e.g. 'cuda:0', 'mps' or 'cpu'. Auto-detected if None.
        """
        if device is None:
            device = get_default_device()
        # Load model and set to evaluation mode
        self.model = YOLO(model_path).eval().to(device)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """
        Run detection on a single image frame.

        Args:
            frame: A BGR image (numpy array) from OpenCV.

        Returns:
            A list of detection dicts, each containing:
                'box': [x1, y1, x2, y2],
                'cls': class index,
                'conf': confidence score,
                'edited': False (flag for manual edits)
        """
        res = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        dets = []
        # Convert each detection box to integers and gather metadata
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            dets.append({
                'box': [x1, y1, x2, y2],
                'cls': int(b.cls.cpu()),
                'conf': float(b.conf.cpu()),
                'edited': False
            })
        return dets


class AnnotationEditor:
    """
    Main class for the annotation editor GUI.
    """
    def __init__(self,
                 data_path,
                 model_path,
                 conf_thresh,
                 out_img_dir,
                 out_lbl_dir,
                 width=1920,
                 height=1080,
                 video_or_images="video",
                 start_from_frame=0):
        """
        Initialize video capture, model task, output directories, and pygame UI.

        Args:
            video_path: Path to input video file.
            model_path: Path to YOLO weights file.
            conf_thresh: Confidence threshold for initial detections.
            out_img_dir: Directory to save corrected image frames.
            out_lbl_dir: Directory to save YOLO-format .txt labels.
            width: Width of the display window.
            height: Height of the display window.
            video_or_images: Specifies how to interpret `data_path`:
            - ``video`` treat ``data_path`` as a single video file
            - ``images`` treat ``data_path`` as a directory of image files
            start_from_frame: Frame index to start editing from.
        """
        self.video_or_images = video_or_images
        self.current_frame = start_from_frame
        if video_or_images == "video":
            video_path = data_path
            # Setup video capture
            self.base_name = os.path.splitext(os.path.basename(video_path))[0]
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")
            self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            ret, frame = self.cap.read()
        elif video_or_images == "images":
            self.frames = os.listdir(data_path)
            self.base_name = os.path.basename(data_path)
            self.n_frames = len(self.frames)
            self.data_path = data_path
            frame = cv2.imread(os.path.join(data_path, self.frames[self.current_frame]))
        else:
            raise ValueError("video_or_images must be one of: ['video', 'images']")


        # Initialize YOLO detection task
        self.task = YOLOTask(model_path, conf_thresh)

        # Prepare output directories
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        self.out_img_dir = out_img_dir
        self.out_lbl_dir = out_lbl_dir
        self.width = width
        self.height = height

        # Annotation data and UI state
        self.dets = {}                # cache of detections per frame
        self.mode = 'edit'            # 'edit' or 'draw' mode
        self.selected = None          # index of selected box
        self.tmp_box = None           # temporary box while drawing
        self.drag = False             # flag for dragging/resizing
        self.drag_corner = None       # which corner is dragged or 'move'
        self.orig = None              # original box coords before drag
        self.off = (0, 0)             # offset for moving a box

        # Initialize pygame window
        pygame.init()
        frame = cv2.resize(frame, (width, height))
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('YOLO Annotation Editor')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.running = True

        # For efficient frame caching
        self.last_frame = None
        self.last_index = -1

    def _to_surface(self, frame):
        """
        Convert an OpenCV BGR image to a pygame Surface.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def _draw(self, frame):
        """
        Render the current frame, boxes, and UI elements to the screen.
        """
        surf = self._to_surface(frame)
        self.screen.blit(surf, (0, 0))

        # Draw each detected/edited box
        for idx, det in enumerate(self.dets.get(self.current_frame, [])):
            x1, y1, x2, y2 = det['box']
            # Highlight selected box in red, others in green
            color = (255, 0, 0) if idx == self.selected else (0, 255, 0)
            pygame.draw.rect(self.screen, color, (x1, y1, x2 - x1, y2 - y1), BOX_THICK)
            # Draw corner handles on the selected box
            if idx == self.selected:
                for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                    pygame.draw.circle(self.screen, (255, 0, 0), (cx, cy), CORNER_SIZE // 2)
            # Draw class name above box
            cls_name = self.task.model.names[det['cls']]
            lbl = self.font.render(cls_name, True, color)
            self.screen.blit(lbl, (x1, y1 - FONT_SIZE))

        # Draw temporary new box when in draw mode
        if self.mode == 'draw' and self.tmp_box:
            x1, y1, x2, y2 = self.tmp_box
            pygame.draw.rect(self.screen, (0, 0, 255), (x1, y1, x2 - x1, y2 - y1), 1)

        # Heads-up display: current frame and mode
        hud1 = self.font.render(f'Frame {self.current_frame+1}/{self.n_frames}', True, (255, 255, 0))
        hud2 = self.font.render(f'Mode: {self.mode.upper()}', True, (255, 255, 0))
        self.screen.blit(hud1, (10, 10))
        self.screen.blit(hud2, (10, 35))

        pygame.display.flip()

    def _save(self, frame):
        """
        Save the current frame as an image and write its YOLO-format label file.
        """
        # Construct filenames
        if self.video_or_images == "video":
            fname = f'{self.base_name}_frame_{self.current_frame:06d}.jpg'
            img_path = os.path.join(self.out_img_dir, fname)
            lbl_path = os.path.join(self.out_lbl_dir, fname.replace('.jpg', '.txt'))

            # Save image
            cv2.imwrite(img_path, frame)
        else:
            fname = self.frames[self.current_frame]
            lbl_path = os.path.join(self.out_lbl_dir, fname.replace('.jpg', '.txt'))


        # Save label file: one line per box in 'cls x_center y_center width height' normalized format
        h, w = frame.shape[:2]
        with open(lbl_path, 'w') as f:
            for det in self.dets[self.current_frame]:
                x1, y1, x2, y2 = det['box']
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{det['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def _get_next_frame(self):
        if self.video_or_images == "video":
            ret, frame = self.cap.read()
        else:
            try:
                frame = cv2.imread(os.path.join(self.data_path, self.frames[self.current_frame]))
                ret = True
            except:
                frame = None
                ret = False
        return ret, frame
    
    def _get_frame(self):
        # Load frame if needed
        if self.last_index != self.current_frame:
            # Seek to current frame index
            ret, frame = self._get_next_frame()
            if not ret:
                return False, None
            
            self.last_frame = frame
            self.last_index = self.current_frame
        else:
            ret, frame = True, self.last_frame
        
        frame = cv2.resize(frame, (self.width, self.height))
        return ret, frame

    def start(self):
        """
        Main loop: handle events, update state, draw frames, and manage editing.
        """
        while self.running:
            # Load the current frame
            ret, frame = self._get_frame()
            if not ret:
                break
            # Perform detection once per frame
            if self.current_frame not in self.dets:
                self.dets[self.current_frame] = self.task.detect(frame)

            # Event handling
            for ev in pygame.event.get():
                # Quit events
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                    self.running = False

                # Keyboard controls
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_d:
                        # Next frame
                        self.current_frame = min(self.current_frame + 1, self.n_frames - 1)
                        self.selected = None
                        self.mode = 'edit'
                    elif ev.key == pygame.K_a:
                        # Previous frame
                        self.current_frame = max(self.current_frame - 1, 0)
                        if self.video_or_images == "video":
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                        self.selected = None
                        self.mode = 'edit'
                    elif ev.key == pygame.K_b:
                        # Enter draw-new-box mode
                        self.mode = 'draw'
                        self.selected = None
                    elif ev.key == pygame.K_e:
                        # Enter edit-existing-box mode
                        self.mode = 'edit'
                    elif ev.key == pygame.K_s:
                        # Save current frame and labels
                        self._save(frame)
                    elif ev.key == pygame.K_x and self.selected is not None:
                        # Delete selected box
                        del self.dets[self.current_frame][self.selected]
                        self.selected = None
                    # Change class via number keys (top row & numpad)
                    elif pygame.K_0 <= ev.key <= pygame.K_9 and self.selected is not None:
                        num = ev.key - pygame.K_0
                        if num < len(self.task.model.names):
                            self.dets[self.current_frame][self.selected]['cls'] = num
                    elif pygame.K_KP1 <= ev.key <= pygame.K_KP0 and self.selected is not None:
                        num = (ev.key - pygame.K_KP1 + 1) % 10
                        if num < len(self.task.model.names):
                            self.dets[self.current_frame][self.selected]['cls'] = num

                # Mouse down: start draw or select/drag
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    x, y = ev.pos
                    if self.mode == 'draw':
                        # Begin drawing a new box
                        self.tmp_box = [x, y, x, y]
                    else:
                        # Attempt to start dragging or select a box
                        self.drag = False
                        if self.selected is not None:
                            # Check if clicking on a corner handle
                            det = self.dets[self.current_frame][self.selected]
                            x1, y1, x2, y2 = det['box']
                            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            for i, (cx, cy) in enumerate(corners):
                                if abs(x - cx) <= CORNER_SIZE and abs(y - cy) <= CORNER_SIZE:
                                    self.drag = True
                                    self.drag_corner = i
                                    self.orig = det['box'][:]
                                    break
                        if not self.drag:
                            # Select or move existing box
                            self.selected = None
                            for idx in reversed(range(len(self.dets[self.current_frame]))):
                                det = self.dets[self.current_frame][idx]
                                x1, y1, x2, y2 = det['box']
                                # Check corners first, then interior
                                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                                for i, (cx, cy) in enumerate(corners):
                                    if abs(x - cx) <= CORNER_SIZE and abs(y - cy) <= CORNER_SIZE:
                                        self.selected = idx
                                        self.drag = True
                                        self.drag_corner = i
                                        self.orig = det['box'][:]
                                        break
                                else:
                                    if x1 <= x <= x2 and y1 <= y <= y2:
                                        # Move the entire box
                                        self.selected = idx
                                        self.drag = True
                                        self.drag_corner = 'move'
                                        self.orig = det['box'][:]
                                        self.off = (x - x1, y - y1)
                                if self.drag:
                                    break

                # Mouse moving: update drawing or dragging
                elif ev.type == pygame.MOUSEMOTION:
                    x, y = ev.pos
                    if self.mode == 'draw' and self.tmp_box:
                        # Update the temporary box coordinates
                        self.tmp_box[2], self.tmp_box[3] = x, y
                    elif self.drag and self.selected is not None:
                        bx = self.dets[self.current_frame][self.selected]['box']
                        if self.drag_corner == 'move':
                            # Move entire box by offset
                            w = self.orig[2] - self.orig[0]
                            h = self.orig[3] - self.orig[1]
                            nx, ny = x - self.off[0], y - self.off[1]
                            bx[:] = [nx, ny, nx + w, ny + h]
                        else:
                            # Resize box by dragging corner
                            if self.drag_corner == 0:
                                bx[0], bx[1] = x, y
                            elif self.drag_corner == 1:
                                bx[2], bx[1] = x, y
                            elif self.drag_corner == 2:
                                bx[2], bx[3] = x, y
                            elif self.drag_corner == 3:
                                bx[0], bx[3] = x, y

                # Mouse up: finalize draw or stop drag
                elif ev.type == pygame.MOUSEBUTTONUP:
                    if self.mode == 'draw' and self.tmp_box:
                        # Add the new box to detections
                        x1, y1, x2, y2 = self.tmp_box
                        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                        cls = 0  # default class for new boxes
                        self.dets[self.current_frame].append({
                            'box': box,
                            'cls': cls,
                            'conf': 1.0,
                            'edited': True
                        })
                        # Reset draw mode back to edit
                        self.tmp_box = None
                        self.mode = 'edit'
                    self.drag = False

            # Render current state and cap the loop rate
            self._draw(frame)
            self.clock.tick(FPS)

        # Clean up pygame when done
        pygame.quit()

    @staticmethod
    def video_to_frames(video_path, out_img_dir, prefix="frame"):
        """
        Extracts all frames from a video file and saves them as individual image files.

        Args:
            video_path (str): Path to the input video file.
            out_img_dir (str): Directory where extracted frames will be saved. It will not be created by this function,
                                so ensure it exists before calling.
            prefix (str, optional): Prefix for generated frame filenames. Defaults to "frame".

        Raises:
            IOError: If the video file cannot be opened.

        Returns:
            None: The function writes image files to disk and does not return a value.
        """
        from tqdm import tqdm
        print("Extracting frames from video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in tqdm(range(n_frames)):
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            # Build filename and save
            filename = f"{prefix}_{idx:06d}.jpg"
            out_path = os.path.join(out_img_dir, filename)
            cv2.imwrite(out_path, frame)

        cap.release()
        print(f"Extracted {idx} frames to '{out_img_dir}'")


if __name__ == '__main__':
    # AnnotationEditor.video_to_frames("20_2_24_1.mp4", "output/images", prefix="video1")
    
    editor = AnnotationEditor(
        data_path='videos/20_2_24_1.mp4',
        video_or_images="video",
        model_path='models/best.pt',
        conf_thresh=0.5,
        out_img_dir='images',
        out_lbl_dir='labels',
        start_from_frame=0,
        width=1280,
        height=720
    )
    editor.start()
