"""
Conda-based installation script for the YOLO pseudo-labeling project.
Creates a conda environment and installs dependencies (with GPU-enabled PyTorch).
"""
import subprocess
import sys

ENV_NAME = "yolo_env_project"
PYTHON_VERSION = "3.11"
CUDA_VERSION = "11.8"

def run(cmd):
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    # 1. Create conda environment
    try:
        run(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
    except subprocess.CalledProcessError:
        print(f"Environment '{ENV_NAME}' may already exist. Continuing...")

    # 2. Install GPU-enabled PyTorch
    run([
        "conda", "install", "-n", ENV_NAME,
        "-c", "pytorch", "-c", "nvidia",
        "pytorch", "torchvision", "torchaudio",
        f"pytorch-cuda={CUDA_VERSION}",
        "-y"
    ])

    # 3. Install other dependencies via conda
    run([
        "conda", "install", "-n", ENV_NAME,
        "-c", "conda-forge",
        "opencv", "pygame", "tqdm",
        "-y"
    ])

    # 4. Install pip-only packages inside env
    run([
        "conda", "run", "-n", ENV_NAME,
        "pip", "install",
        "ultralytics"
    ])

    print("\nInstallation complete.")
    print(f"To activate the environment, run: conda activate {ENV_NAME}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: 'conda' not found. Please ensure Anaconda or Miniconda is installed and on your PATH.")
        sys.exit(1)
