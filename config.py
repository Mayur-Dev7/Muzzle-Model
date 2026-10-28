"""
Centralized Configuration for Cattle Muzzle Mask R-CNN & Biometric Verification
================================================================================
Centralizes all hardware constraints, video sampling, Mask R-CNN detection,
super-resolution enhancement, and ORB biometric matching parameters.
"""

import os
import sys
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = str(BASE_DIR / "weights" / "crop11dec.h5")
OUTPUT_DIR = str(BASE_DIR / "output")
SAMPLES_DIR = str(BASE_DIR / "samples")

# -----------------------------------------------------------------------------
# Hardware Configuration (Enforce 2 CPU Cores, Force CPU execution)
# -----------------------------------------------------------------------------
CPU_CORES = 2
DEVICE = "cpu"
FORCE_CPU = True
CUDA_VISIBLE_DEVICES = "-1"

def enforce_cpu_lock(cores: int = CPU_CORES):
    """
    Enforces process affinity to `cores` CPU cores (default: cores 0 and 1)
    and disables GPU for TensorFlow, OpenMP, BLAS, and OpenCV.
    Must be called before TensorFlow is imported or initialized.
    """
    # 1. Environment variables to suppress GPU and lock OpenMP/BLAS threads
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(cores)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # 2. Linux CPU Affinity to cores {0, 1}
    affinity_set = False
    if hasattr(os, "sched_setaffinity"):
        try:
            available_cores = set(range(min(cores, os.cpu_count() or cores)))
            os.sched_setaffinity(0, available_cores)
            affinity_set = True
        except Exception:
            pass

    # 3. OpenCV threading limit
    try:
        import cv2
        cv2.setNumThreads(cores)
    except ImportError:
        pass

    return affinity_set

# Automatically enforce hardware lock upon module import
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
