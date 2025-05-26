"""
Conda-based installation script for the YOLO pseudo-labeling project.
Creates a conda environment and installs dependencies (with GPU-enabled PyTorch).
"""
import argparse
import subprocess
import sys
import os

ENV_NAME = "yolo_env_project"
PYTHON_VERSION = "3.11"
CUDA_VERSION = "11.8"

def run(cmd):
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd)




def conda_setup():
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


def pip_setup():
    # 1. Create virtual environment
    try:
        # Try Windows Python launcher first, then fallback
        if sys.platform == "win32":
            try:
                run(["py", f"-{PYTHON_VERSION}", "-m", "venv", ENV_NAME])
            except subprocess.CalledProcessError:
                run(["python", "-m", "venv", ENV_NAME])
        else:
            run([f"python{PYTHON_VERSION}", "-m", "venv", ENV_NAME])
    except subprocess.CalledProcessError:
        print(f"Virtual environment '{ENV_NAME}' may already exist. Continuing...")

    # Determine pip and python paths
    if sys.platform == "win32":
        pip_exe = os.path.join(ENV_NAME, "Scripts", "pip")
        python_exe = os.path.join(ENV_NAME, "Scripts", "python")
    else:
        pip_exe = os.path.join(ENV_NAME, "bin", "pip")
        python_exe = os.path.join(ENV_NAME, "bin", "python")

    # 2. Upgrade pip
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

    # 3. Install GPU-enabled PyTorch
    run([
        pip_exe, "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/cu{CUDA_VERSION.replace('.', '')}"
    ])

    # 4. Install other dependencies via pip
    run([
        pip_exe, "install",
        "opencv-python", "pygame", "tqdm"
    ])

    # 5. Install additional packages
    run([
        pip_exe, "install",
        "ultralytics", "numpy", "pillow", "pyyaml"
    ])

    print("\nInstallation complete.")
    if sys.platform == "win32":
        print(f"To activate the environment, run: {ENV_NAME}\\Scripts\\activate")
    else:
        print(f"To activate the environment, run: source {ENV_NAME}/bin/activate")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Install YOLO pseudo-labeling project dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
                  python setup.py --conda                 # Install with conda (default)
                  python setup.py --pip                   # Install with pip in virtual env (complete)
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--conda",
        action="store_true",
        help="Install using conda (default)"
    )
    group.add_argument(
        "--pip",
        action="store_true",
        help="Install using pip"
    )


    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    try:
        if args.pip:
            pip_setup()
        else:
            conda_setup()

    except subprocess.CalledProcessError as e:
        print(f"Installation failed with error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        if "conda" in str(e):
            print("Error: 'conda' not found. Please ensure Anaconda or Miniconda is installed and on your PATH.")
            print("Alternatively, try running with --pip flag for pip-based installation.")
        else:
            print(f"Error: {e}")
        sys.exit(1)

