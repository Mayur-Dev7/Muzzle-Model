# Cattle Muzzle Mask R-CNN & Biometric Verification System

A high-performance standalone Python engine for detecting cattle muzzles, extracting normalized dermatoglyphic crops, and performing biometric identity verification across videos or images.

## Architecture Highlights

- **Hardware Lockdown**: Hard-locked to **2 CPU cores** (`{0, 1}`) via `os.sched_setaffinity`, with GPU explicitly disabled (`CUDA_VISIBLE_DEVICES = "-1"`, `OMP_NUM_THREADS = 2`, `TF_NUM_INTRAOP_THREADS = 2`).
- **Mask R-CNN Detection**: Optimized inference using custom weights (`crop11dec.h5`), square 800x800 input mold, and 100 post-NMS ROIs.
- **Biometric Extraction**: Aspect-ratio-preserving square padding (8% expansion) with normalized (224, 224) resolution.
- **Motion Blur Pre-filtering**: Laplacian variance thresholding (`< 20.0`) to reject motion-blurred frames in sub-millisecond time.
- **Pristine Early Exit**: Automatically triggers early exit when a pristine muzzle crop (`>= 85%` confidence) is acquired, minimizing processing time.
- **Super-Resolution & Dermatoglyphic Enhancement**: 2x Lanczos-4 sinc interpolation, CLAHE contrast equalization, and micro-edge unsharp masking.
- **ORB Biometric Matching**: 1000 keypoint extraction and KNN Hamming matching with Lowe's ratio test (0.7) and duplicate decision threshold (`>= 35` matches).

---

## Directory Structure

```
Muzzle-Model/
├── config.py             # Centralized hyperparameter and hardware configuration
├── cropper.py            # Standalone MuzzleCropper module
├── matcher.py            # Standalone MuzzleMatcher module
├── main.py               # User-friendly CLI with subcommands (info, crop, compare)
├── requirements.txt      # Pinned CPU dependencies
├── .gitignore            # Git ignore rules (weights tracked via Git LFS)
├── .gitattributes       # Git LFS declaration for *.h5 weights
├── mrcnn/                # Matterport Mask R-CNN engine
├── weights/
│   └── crop11dec.h5      # Mask R-CNN pre-trained model weights
├── samples/
│   ├── muzzleVideo.mp4   # Sample testing video 1
│   └── muzzleVideo2.mp4  # Sample testing video 2
└── output/               # Generated crops and comparison diagrams
```

---

## CLI Usage

### 1. Hardware Allocation and Model Status
```bash
python main.py info
```

### 2. Biometric Comparison Between Two Videos
```bash
python main.py compare --video-a samples/muzzleVideo.mp4 --video-b samples/muzzleVideo2.mp4
```

### 3. Cattle Muzzle Cropping from a Video
```bash
python main.py crop --video samples/muzzleVideo.mp4
```

### 4. Cattle Muzzle Cropping from an Image
```bash
python main.py crop --image path/to/cattle.jpg
```

---

## Python API Usage

### MuzzleCropper
```python
from cropper import MuzzleCropper

cropper = MuzzleCropper()

# Crop from image
res = cropper.crop_image("path/to/cow.jpg", output_dir="output")

# Crop from video
video_res = cropper.crop_video("samples/muzzleVideo.mp4", output_dir="output")
```

### MuzzleMatcher
```python
from matcher import MuzzleMatcher

matcher = MuzzleMatcher()

# Compare two cattle videos
result = matcher.compare_videos(
    video_a_path="samples/muzzleVideo.mp4",
    video_b_path="samples/muzzleVideo2.mp4",
    output_dir="output"
)
print("Verdict:", result["verdict"])
print("Matches:", result["good_matches"], "Similarity:", result["similarity_percent"])
```
