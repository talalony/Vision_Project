import argparse
import os
import cv2
import torch
from ultralytics import YOLO

def get_default_device():
    """
    Returns the default torch device: GPU/MPS if available, else CPU.
    """
    return 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLO model on a video and display the detections live."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the YOLO .pt model file."
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Location to save frames into"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save the annotated video with '_annotated' suffix."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width for display (default: 1280)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height for display (default: 720)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load YOLO model
    device = get_default_device()
    print(f"Loading model from '{args.model_path}' on device '{device}'")
    model = YOLO(args.model_path).eval().to(device)

    # Open video capture
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.video_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Get video properties for saving
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer if save_video flag is set
    video_writer = None
    output_video_path = None
    if args.save_video:
        # Generate output filename based on input video name
        input_basename = os.path.basename(args.video_path)
        input_name, input_ext = os.path.splitext(input_basename)
        output_video_path = os.path.join(args.save_dir, f"{input_name}_annotated{input_ext}")

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Alternative: cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (args.width, args.height))
        print(f"Will save annotated video to: {output_video_path}")

    window_name = "YOLO Video Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    print(f"Processing video '{args.video_path}' (press 'q' to quit and 'space' to pause)...")
    if args.save_video:
        print("Note: Annotated video will be saved even if you quit early.")

    play = True
    last_frame = None
    frames_processed = 0

    while True:
        if play or last_frame is None:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached.")
                break

            # Resize
            frame = cv2.resize(frame, (args.width, args.height))

            # Run inference
            results = model(frame, conf=args.conf_thresh, verbose=False)[0]

            # Draw detections
            annotated = results.plot()

            # Save frame to video if writer is active
            if video_writer is not None:
                video_writer.write(annotated)

            frames_processed += 1
            if frames_processed % 30 == 0:  # Progress update every 30 frames
                progress = (frames_processed / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processed {frames_processed}/{total_frames} frames ({progress:.1f}%)")

            last_frame = annotated
        else:
            annotated = last_frame

        # Display
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting video inference.")
            break
        elif key == 32: # Space bar
            play = not play
            print("Paused" if not play else "Resumed")
        elif key == ord('s'):
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            filename = f"frame_{frame_idx:06d}.png"
            save_path = os.path.join(args.save_dir, filename)
            cv2.imwrite(save_path, annotated)
            print(f"Frame saved: {save_path}")

    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Annotated video saved to: {output_video_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()