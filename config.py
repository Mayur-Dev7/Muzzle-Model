"""
Centralized Configuration for Cattle Muzzle Mask R-CNN & Biometric Verification
================================================================================
Centralizes model settings and applies portable CPU thread limits. Detects
Apple Silicon and validates that its Python/TensorFlow runtime is native ARM64.
"""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = str(BASE_DIR / "weights" / "crop11dec.h5")
OUTPUT_DIR = str(BASE_DIR / "output")
SAMPLES_DIR = str(BASE_DIR / "samples")

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
CPU_CORES = 2
DEVICE = "cpu"
FORCE_CPU = True
CUDA_VISIBLE_DEVICES = "-1"


def _sysctl_value(name: str):
    """Return a macOS sysctl value, or None when unavailable."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def detect_hardware():
    """Describe the host chip, process architecture, and available CPU cores."""
    system = platform.system()
    machine = (platform.machine() or "unknown").lower()
    arm64_capable = _sysctl_value("hw.optional.arm64") == "1"
    is_apple_silicon = system == "Darwin" and (
        machine in {"arm64", "aarch64"} or arm64_capable
    )
    rosetta_translated = _sysctl_value("sysctl.proc_translated") == "1"

    chip_name = _sysctl_value("machdep.cpu.brand_string") or platform.processor()
    if is_apple_silicon and (not chip_name or chip_name.lower() in {"arm", "arm64", "aarch64"}):
        # Apple Silicon often reports a generic architecture through platform.processor().
        try:
            profiler = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
                check=False,
            )
            for line in profiler.stdout.splitlines():
                label, separator, value = line.strip().partition(":")
                if separator and label.strip().lower() in {"chip", "processor name"} and value.strip():
                    chip_name = value.strip()
                    break
        except (OSError, subprocess.SubprocessError):
            pass
    chip_name = chip_name or ("Apple Silicon (model unavailable)" if is_apple_silicon else machine)

    def _positive_int(value):
        try:
            parsed = int(value)
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    physical_cores = _positive_int(_sysctl_value("hw.physicalcpu"))
    logical_cores = _positive_int(_sysctl_value("hw.logicalcpu")) or os.cpu_count() or 1

    return {
        "system": system,
        "machine": machine,
        "chip_name": chip_name,
        "is_apple_silicon": is_apple_silicon,
        "rosetta_translated": rosetta_translated,
        "physical_cores": physical_cores or "N/A",
        "logical_cores": logical_cores,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }


def get_tensorflow_version():
    """Read TensorFlow package metadata without importing the TensorFlow runtime."""
    for package_name in ("tensorflow", "tensorflow-macos"):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not installed"


def validate_tensorflow_runtime():
    """
    Fail early on Apple Silicon when Python is running through Rosetta or the
    legacy TensorFlow 1.x runtime is installed. Call this before importing TF.
    """
    hardware = detect_hardware()
    if not hardware["is_apple_silicon"]:
        return

    if hardware["machine"] not in {"arm64", "aarch64"} or hardware["rosetta_translated"]:
        raise RuntimeError(
            "Apple Silicon was detected, but this Python process is not running "
            "natively as ARM64 (it appears to be x86_64/Rosetta). Install/use a "
            "native ARM64 Python, create a fresh virtual environment, and reinstall "
            "the repository requirements. Verify with: "
            "python -c \"import platform; print(platform.machine())\" "
            "(expected: arm64)."
        )

    version = get_tensorflow_version()
    try:
        version_parts = tuple(int(part) for part in version.split(".")[:2])
    except (AttributeError, ValueError):
        version_parts = ()

    if not version_parts or version_parts < (2, 13) or version_parts >= (2, 16):
        raise RuntimeError(
            "This Matterport Mask R-CNN codebase requires TensorFlow >=2.13,<2.16 "
            "for its legacy Keras API; the detected TensorFlow package is "
            f"{version!r}. TensorFlow 1.15.x is incompatible with Apple Silicon, "
            "and TensorFlow 2.16+ uses Keras 3, which this codebase has not been "
            "ported to. Create a fresh native ARM64 virtual environment and run: "
            "python -m pip install -r requirements.txt"
        )


def enforce_cpu_lock(cores: int = CPU_CORES):
    """
    Limit CPU-heavy libraries to a bounded number of threads. Linux process
    affinity is applied only when supported; macOS uses library thread limits
    because os.sched_setaffinity is not available there.
    """
    try:
        cores = max(1, min(int(cores), os.cpu_count() or int(cores)))
    except (TypeError, ValueError):
        cores = CPU_CORES

    # Set limits before TensorFlow/OpenMP/BLAS initialize.
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(cores)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Restrict to the first permitted CPUs, not blindly to {0, 1}. In containers
    # the process may only be allowed to use a different CPU range.
    affinity_set = False
    if hasattr(os, "sched_setaffinity"):
        try:
            if hasattr(os, "sched_getaffinity"):
                permitted_cores = sorted(os.sched_getaffinity(0))
            else:
                permitted_cores = list(range(os.cpu_count() or cores))
            selected_cores = set(permitted_cores[:cores])
            if selected_cores:
                os.sched_setaffinity(0, selected_cores)
                affinity_set = True
        except (OSError, ValueError):
            pass

    try:
        import cv2
        cv2.setNumThreads(cores)
    except ImportError:
        pass

    return affinity_set


# Apply portable thread limits before TensorFlow is imported.
enforce_cpu_lock()

# -----------------------------------------------------------------------------
# Video Sampling Configuration
# -----------------------------------------------------------------------------
FRAME_INTERVAL = 10               # Sample every 10th frame
MAX_CANDIDATE_FRAMES = 8          # Stop video sampling after evaluating N candidate frames
BLUR_VARIANCE_THRESHOLD = 20.0    # Minimum Laplacian variance (skip frames with blur < 20.0)

# -----------------------------------------------------------------------------
# Model & Detection Configuration
# -----------------------------------------------------------------------------
DETECTION_CONFIDENCE_THRESHOLD = 0.70   # Minimum Mask R-CNN detection confidence
PRISTINE_CONFIDENCE_THRESHOLD = 0.85    # Early-exit threshold for pristine muzzle detection
IMAGE_MIN_DIM = 800                     # Input mold minimum dimension
IMAGE_MAX_DIM = 1024                    # Input mold maximum dimension (must be divisible by 64: 1024 / 64 = 16)
POST_NMS_ROIS_INFERENCE = 100           # ROIs passed after NMS for inference speedup
CROP_SIZE = (224, 224)                  # Final normalized biometric crop resolution (W, H)
CROP_PADDING_PCT = 0.08                 # 8% bounding box margin expansion

# -----------------------------------------------------------------------------
# Super-Resolution & Enhancement Configuration
# -----------------------------------------------------------------------------
UPSCALE_METHOD = "lanczos4"             # "lanczos4" (sinc 8-point), "bicubic", or "bilinear"
UPSCALE_FACTOR = 2                      # 2x super-resolution (e.g. 224x224 -> 448x448)
APPLY_CLAHE = True                      # Contrast Limited Adaptive Histogram Equalization
APPLY_UNSHARP = True                    # Micro-edge groove unsharp masking
CLAHE_CLIP_LIMIT = 2.0                  # CLAHE clip limit
CLAHE_GRID_SIZE = (8, 8)                # CLAHE local grid tile size

# -----------------------------------------------------------------------------
# Biometric Matching Configuration
# -----------------------------------------------------------------------------
ORB_N_FEATURES = 1000                   # Number of ORB keypoints to extract
ORB_RATIO_TEST = 0.7                    # Lowe's distance ratio test threshold (0.7)
DUPLICATE_DECISION_THRESHOLD = 35       # >= 35 good matches declares SAME CATTLE duplicate

# -----------------------------------------------------------------------------
# Mask R-CNN Inference Config Subclass
# -----------------------------------------------------------------------------
try:
    from mrcnn.config import Config

    class CowMuzzleInferenceConfig(Config):
        """Matterport Mask R-CNN configuration for cattle muzzle detection."""
        NAME = "CowMuzzle"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1             # Background (0) + Cow Muzzle (1)
        DETECTION_MIN_CONFIDENCE = DETECTION_CONFIDENCE_THRESHOLD
        DETECTION_NMS_THRESHOLD = 0.30
        IMAGE_MIN_DIM = IMAGE_MIN_DIM
        IMAGE_MAX_DIM = IMAGE_MAX_DIM
        IMAGE_RESIZE_MODE = "square"
        POST_NMS_ROIS_INFERENCE = POST_NMS_ROIS_INFERENCE

except ImportError:
    CowMuzzleInferenceConfig = None
