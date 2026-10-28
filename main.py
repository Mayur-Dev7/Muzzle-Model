#!/usr/bin/env python3
"""
Cattle Muzzle Mask R-CNN & Biometric Verification CLI
====================================================
Command-line interface for cattle muzzle detection, square biometric cropping,
dermatoglyphic ridge enhancement, and video-to-video identity verification.

Subcommands:
  compare  - Compare two cattle videos to verify if they are the same animal
  crop     - Detect and crop cattle muzzles from an image or video file
  info     - Display hardware resource allocation, 2-core CPU lock, and weights status
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce 2 CPU cores and CPU-only execution before TensorFlow/OpenMP initialize
import config
config.enforce_cpu_lock()

import argparse
from tabulate import tabulate
import cv2


def run_info(args):
    """Prints hardware allocation, CPU 2-core affinity status, and weights status."""
    print("\n" + "=" * 70)
    print(" CATTLE MUZZLE SYSTEM CONFIGURATION & HARDWARE DIAGNOSTICS")
    print("=" * 70)

    # 1. Hardware Allocation & CPU Affinity
    affinity_set = "N/A"
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_set = str(sorted(list(os.sched_getaffinity(0))))
        except Exception as e:
            affinity_set = f"Error: {e}"

    cv2_threads = cv2.getNumThreads() if hasattr(cv2, "getNumThreads") else "N/A"

    hw_table = [
        ["Target Device", config.DEVICE.upper()],
        ["Force CPU Mode", str(config.FORCE_CPU)],
        ["CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")],
        ["Configured CPU Cores", str(config.CPU_CORES)],
        ["Linux CPU Affinity", affinity_set],
        ["Total System CPU Cores", str(os.cpu_count())],
        ["OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "Not set")],
        ["TF_NUM_INTRAOP_THREADS", os.environ.get("TF_NUM_INTRAOP_THREADS", "Not set")],
        ["OpenCV Thread Limit", str(cv2_threads)],
    ]
    print("\n[1] HARDWARE ALLOCATION & CPU LOCK (2 CORES)")
    print(tabulate(hw_table, headers=["Parameter", "Status / Value"], tablefmt="fancy_grid"))

    # 2. Model Weights Status
    weights_path = Path(args.weights if hasattr(args, "weights") and args.weights else config.WEIGHTS_PATH).resolve()
    weights_exists = weights_path.is_file()
    file_size_mb = f"{weights_path.stat().st_size / (1024 * 1024):.1f} MB" if weights_exists else "N/A"
    is_readable = os.access(weights_path, os.R_OK) if weights_exists else False

    weights_table = [
        ["Weights File Path", str(weights_path)],
        ["File Exists", "YES (Ready)" if weights_exists else "NO (Missing)"],
        ["File Size", file_size_mb],
        ["Read Permissions", "YES" if is_readable else "NO"],
        ["Input Mold Resolution", f"{config.IMAGE_MIN_DIM}x{config.IMAGE_MAX_DIM} px"],
        ["Post-NMS ROIs Inference", str(config.POST_NMS_ROIS_INFERENCE)],
    ]
    print("\n[2] MODEL WEIGHTS & INFERENCE GEOMETRY")
    print(tabulate(weights_table, headers=["Attribute", "Value"], tablefmt="fancy_grid"))

    # 3. Biometric Pipeline Parameters
    pipeline_table = [
        ["Video Sampling Interval", f"Every {config.FRAME_INTERVAL}th frame"],
        ["Max Candidate Frames", str(config.MAX_CANDIDATE_FRAMES)],
        ["Laplacian Blur Threshold", f"{config.BLUR_VARIANCE_THRESHOLD} var"],
        ["Detection Confidence Threshold", f"{config.DETECTION_CONFIDENCE_THRESHOLD * 100:.0f}%"],
        ["Pristine Early-Exit Threshold", f"{config.PRISTINE_CONFIDENCE_THRESHOLD * 100:.0f}%"],
        ["Biometric Crop Resolution", f"{config.CROP_SIZE[0]}x{config.CROP_SIZE[1]} px (Square)"],
        ["Crop Margin Padding", f"{config.CROP_PADDING_PCT * 100:.0f}%"],
        ["Super-Resolution Interpolation", f"{config.UPSCALE_METHOD.upper()} ({config.UPSCALE_FACTOR}x)"],
        ["Dermatoglyphic CLAHE", f"Enabled (Clip: {config.CLAHE_CLIP_LIMIT}, Grid: {config.CLAHE_GRID_SIZE})"],
        ["Micro-edge Unsharp Masking", str(config.APPLY_UNSHARP)],
        ["ORB Feature Keypoints", str(config.ORB_N_FEATURES)],
        ["Lowe Ratio Test", str(config.ORB_RATIO_TEST)],
        ["Duplicate Decision Threshold", f">= {config.DUPLICATE_DECISION_THRESHOLD} Good Matches"],
        ["Default Output Directory", config.OUTPUT_DIR],
    ]
    print("\n[3] BIOMETRIC VERIFICATION HYPERPARAMETERS")
    print(tabulate(pipeline_table, headers=["Parameter", "Setting"], tablefmt="fancy_grid"))
    print("\n")


def run_crop(args):
    """Executes cattle muzzle detection and square cropping on an image or video."""
    if not args.image and not args.video:
        print("Error: Must specify either --image <path> or --video <path>.", file=sys.stderr)
        sys.exit(1)

    from cropper import MuzzleCropper

    output_dir = args.output_dir or config.OUTPUT_DIR
    weights_path = args.weights or config.WEIGHTS_PATH
    confidence = args.confidence or config.DETECTION_CONFIDENCE_THRESHOLD

    print(f"\nInitializing 2-core CPU MuzzleCropper with weights: {weights_path}...")
    cropper = MuzzleCropper(weights_path=weights_path, confidence_threshold=confidence)

    if args.image:
        print(f"\nProcessing Image: {args.image}...")
        res = cropper.crop_image(
            args.image,
            output_dir=output_dir,
            padding_pct=config.CROP_PADDING_PCT,
            make_square=True,
            target_size=config.CROP_SIZE,
            save_annotated=True,
        )

        num_det = len(res["detections"])
        print(f"\nDetection Results for {res['source']}: {num_det} muzzle(s) found")

        table_rows = []
        for det in res["detections"]:
            table_rows.append([
                det["muzzle_index"],
                f"{det['confidence'] * 100:.1f}%",
                f"{det['bbox_original']}",
                f"{det['crop_size'][0]}x{det['crop_size'][1]}",
                det.get("saved_crop_path", "In-memory"),
            ])

        if table_rows:
            print(tabulate(
                table_rows,
                headers=["#", "Confidence", "Bounding Box (y1, x1, y2, x2)", "Target Size", "Saved Crop Path"],
                tablefmt="fancy_grid",
            ))
            if res.get("saved_annotated_path"):
                print(f"\nAnnotated preview saved: {res['saved_annotated_path']}")
        else:
            print("No cattle muzzles detected with confidence >= threshold.")

        timings = res["timings"]
        t_rows = [
            ["Image Ingestion", f"{timings.get('t_load_ms', 0):.1f} ms"],
            ["Preprocessing & Mold", f"{timings.get('t_preprocess_ms', 0):.1f} ms"],
            ["Neural Network Inference", f"{timings.get('t_inference_ms', 0):.1f} ms"],
            ["Square Padding & Crop", f"{timings.get('t_crop_ms', 0):.1f} ms"],
            ["Total Pipeline Time", f"{timings.get('t_total_ms', 0):.1f} ms"],
            ["Inference Throughput", f"{timings.get('fps', 0):.1f} FPS"],
        ]
        print("\nLatency Profiling Breakdown:")
        print(tabulate(t_rows, headers=["Stage", "Latency"], tablefmt="fancy_grid"))

    elif args.video:
        print(f"\nProcessing Video: {args.video}...")
        frame_interval = args.frame_interval or config.FRAME_INTERVAL
        max_frames = args.max_frames or config.MAX_CANDIDATE_FRAMES

        res = cropper.crop_video(
            args.video,
            output_dir=output_dir,
            frame_interval=frame_interval,
            max_frames=max_frames,
            blur_threshold=config.BLUR_VARIANCE_THRESHOLD,
            padding_pct=config.CROP_PADDING_PCT,
            make_square=True,
            target_size=config.CROP_SIZE,
        )

        detections = res["detections"]
        print(f"\nVideo Processing Complete: {len(detections)} muzzle crop(s) extracted.")
        v_rows = [
            ["Video File", res["video_path"]],
            ["Total Video Frames", str(res["total_frames"])],
            ["Candidate Frames Evaluated", str(res["evaluated_candidates"])],
            ["Total Muzzles Cropped", str(len(detections))],
            ["Total Processing Time", f"{res['total_time_ms']:.1f} ms"],
        ]
        print(tabulate(v_rows, headers=["Video Metric", "Value"], tablefmt="fancy_grid"))

        if res["crop_paths"]:
            print("\nGenerated Biometric Crops:")
            for p in res["crop_paths"][:10]:
                print(f"  - {p}")
            if len(res["crop_paths"]) > 10:
                print(f"  ... and {len(res['crop_paths']) - 10} more.")


def run_compare(args):
    """Executes end-to-end biometric verification comparing two cattle videos."""
    from matcher import MuzzleMatcher

    output_dir = args.output_dir or config.OUTPUT_DIR
    weights_path = args.weights or config.WEIGHTS_PATH
    threshold = args.threshold or config.DUPLICATE_DECISION_THRESHOLD

    print(f"\nInitializing 2-core CPU MuzzleMatcher with weights: {weights_path}...")
    matcher = MuzzleMatcher(weights_path=weights_path)

    result = matcher.compare_videos(
        video_a_path=args.video_a,
        video_b_path=args.video_b,
        output_dir=output_dir,
        threshold=threshold,
    )

    verdict_display = "SAME CATTLE (DUPLICATE)" if result["is_match"] else "DIFFERENT CATTLE (DISTINCT)"

    print("\n" + "=" * 70)
    print(f" BIOMETRIC MATCH VERDICT: {verdict_display}")
    print("=" * 70)

    summary_rows = [
        ["Match Decision", result["verdict"]],
        ["Biometric Verdict", verdict_display],
        ["Good ORB Matches", f"{result['good_matches']} (Threshold: {result['threshold']})"],
        ["Similarity Score", f"{result['similarity_percent']:.1f}%"],
        ["Video A Keypoints", str(result["kp1_count"])],
        ["Video B Keypoints", str(result["kp2_count"])],
        ["Extraction Latency", f"{result['timings']['t_extract_ms']} ms"],
        ["Enhancement Latency", f"{result['timings']['t_enhance_ms']} ms"],
        ["Biometric Matching Latency", f"{result['timings']['t_match_ms']} ms"],
        ["Total Pipeline Time", f"{result['timings']['t_total_ms']} ms"],
    ]
    print(tabulate(summary_rows, headers=["Verification Metric", "Result"], tablefmt="fancy_grid"))

    print("\nGenerated Artifacts:")
    print(f"  - Video A Pristine Crop:  {result.get('crop_a_path')}")
    print(f"  - Video B Pristine Crop:  {result.get('crop_b_path')}")
    print(f"  - Enhanced Print A:       {result.get('enhanced_print_a_path')}")
    print(f"  - Enhanced Print B:       {result.get('enhanced_print_b_path')}")
    print(f"  - Comparison Diagram:     {result.get('visual_diagram_path')}")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cattle Muzzle Mask R-CNN & Biometric Verification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: info
    parser_info = subparsers.add_parser("info", help="Display hardware allocation and model status")
    parser_info.add_argument("--weights", type=str, default=config.WEIGHTS_PATH, help="Path to model weights")

    # Subcommand: crop
    parser_crop = subparsers.add_parser("crop", help="Detect and crop muzzles from an image or video")
    parser_crop.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser_crop.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser_crop.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR, help="Directory to save crops")
    parser_crop.add_argument("--weights", type=str, default=config.WEIGHTS_PATH, help="Path to model weights")
    parser_crop.add_argument("--confidence", type=float, default=config.DETECTION_CONFIDENCE_THRESHOLD, help="Detection threshold")
    parser_crop.add_argument("--frame-interval", type=int, default=config.FRAME_INTERVAL, help="Video frame sampling interval")
    parser_crop.add_argument("--max-frames", type=int, default=config.MAX_CANDIDATE_FRAMES, help="Max candidate frames to evaluate")

    # Subcommand: compare
    parser_compare = subparsers.add_parser("compare", help="Compare two cattle videos for identity verification")
    parser_compare.add_argument("--video-a", type=str, required=True, help="Path to Video A")
    parser_compare.add_argument("--video-b", type=str, required=True, help="Path to Video B")
    parser_compare.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR, help="Directory to save match diagram")
    parser_compare.add_argument("--weights", type=str, default=config.WEIGHTS_PATH, help="Path to model weights")
    parser_compare.add_argument("--threshold", type=int, default=config.DUPLICATE_DECISION_THRESHOLD, help="Decision threshold")

    args = parser.parse_args()

    if args.command == "info":
        run_info(args)
    elif args.command == "crop":
        run_crop(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
