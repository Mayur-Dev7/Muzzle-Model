"""
Cattle Muzzle Biometric Matcher & Verification Pipeline
======================================================
Provides biometric verification between cattle muzzle prints.
Features:
- Motion blur pre-filtering & smart early-exit for pristine muzzle acquisition
- 2x Lanczos-4 super-resolution edge-preserving upscaling
- CLAHE dermatoglyphic ridge bead enhancement & unsharp masking
- ORB keypoint descriptor extraction & Hamming KNN matching with Lowe's ratio test
- Automated duplicate decision logic (Threshold = 35 matches)
- Side-by-side keypoint visual match comparison diagram generation
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce 2-core CPU configuration
import config
config.enforce_cpu_lock()

import cv2
import numpy as np

from cropper import MuzzleCropper

logger = logging.getLogger("muzzle_matcher")
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


def compute_image_quality(image_bgr: np.ndarray, confidence: float = 1.0) -> Dict[str, Any]:
    """
    Computes objective quality metrics for a cattle muzzle crop:
    - Sharpness via Laplacian variance
    - Contrast via pixel intensity standard deviation
    - Resolution & area
    - Composite quality score (0.0 - 100.0)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std_dev = float(np.std(gray))
    h, w = gray.shape[:2]
    area = h * w

    # Normalized component scores (0 - 100)
    sharpness_score = min(100.0, (laplacian_var / 500.0) * 100.0)
    contrast_score = min(100.0, (std_dev / 64.0) * 100.0)
    res_score = min(100.0, (min(h, w) / 224.0) * 100.0)
    conf_score = float(confidence) * 100.0

    # Composite weighted score
    composite = (
        (sharpness_score * 0.35)
        + (contrast_score * 0.25)
        + (conf_score * 0.25)
        + (res_score * 0.15)
    )

    return {
        "composite": round(composite, 1),
        "sharpness": round(laplacian_var, 1),
        "contrast": round(std_dev, 1),
        "resolution": f"{w}x{h}",
        "area": area,
        "confidence": round(conf_score, 1),
    }


def enhance_muzzle_image(
    image_bgr: np.ndarray,
    method: str = config.UPSCALE_METHOD,
    scale_factor: int = config.UPSCALE_FACTOR,
    unsharp: bool = config.APPLY_UNSHARP,
    clahe: bool = config.APPLY_CLAHE,
    clahe_clip_limit: float = config.CLAHE_CLIP_LIMIT,
    clahe_grid_size: Tuple[int, int] = config.CLAHE_GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upscales cattle muzzle image using edge-preserving interpolation and
    enhances dermatoglyphic ridge beads using CLAHE and unsharp masking.

    Returns:
        (enhanced_gray, upscaled_bgr)
    """
    interp_map = {
        "lanczos4": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
    }
    interp = interp_map.get(method.lower(), cv2.INTER_LANCZOS4)

    h, w = image_bgr.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # 1. Super-resolution / Edge-preserving Upscaling
    upscaled_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Unsharp Masking for Ridge/Groove Edge Sharpening
    if unsharp:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    if clahe:
        clahe_filter = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )
        gray = clahe_filter.apply(gray)

    return gray, upscaled_bgr


def match_muzzle_features(
    gray1: np.ndarray,
    gray2: np.ndarray,
    n_features: int = config.ORB_N_FEATURES,
    ratio_test: float = config.ORB_RATIO_TEST,
    threshold: int = config.DUPLICATE_DECISION_THRESHOLD,
    bgr1: Optional[np.ndarray] = None,
    bgr2: Optional[np.ndarray] = None,
    save_visual_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs ORB feature extraction and KNN Hamming matching with Lowe's ratio test.
    Renders a visual comparison diagram with verdict banner.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    num_kp1 = len(kp1) if kp1 else 0
    num_kp2 = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return {
            "good_matches": 0,
            "threshold": threshold,
            "is_match": False,
            "verdict": "INSUFFICIENT_KEYPOINTS",
            "similarity_percent": 0.0,
            "kp1_count": num_kp1,
            "kp2_count": num_kp2,
            "visual_path": None,
        }

    # KNN matching with k=2
    try:
        raw_matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        raw_matches = []

    # Lowe's ratio test
    good_pairs = []
    for pair in raw_matches:
        if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance:
            good_pairs.append(pair[0])

    good_matches = len(good_pairs)
    is_match = good_matches >= threshold
    verdict = "MATCH_SAME_CATTLE" if is_match else "DIFFERENT_CATTLE"

    # Similarity percentage heuristic
    if is_match:
        # Scale 35 -> 75% up to 65+ -> 99%
        similarity_percent = min(99.5, 75.0 + ((good_matches - threshold) / 30.0) * 24.5)
    else:
        # Scale 0 -> 0% up to 34 -> 68%
        similarity_percent = min(68.0, (good_matches / float(threshold)) * 68.0)

    visual_saved = None
    if save_visual_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_visual_path)), exist_ok=True)
        img_left = bgr1 if bgr1 is not None else cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        img_right = bgr2 if bgr2 is not None else cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

        # Match top 40 best pairs
        top_matches = sorted(good_pairs, key=lambda m: m.distance)[:40]
        match_canvas = cv2.drawMatches(
            img_left, kp1,
            img_right, kp2,
            top_matches,
            None,
            matchColor=(34, 197, 94) if is_match else (0, 165, 255),  # Green or Orange
            singlePointColor=(120, 120, 120),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Build informative top banner
        banner_h = 56
        banner = np.zeros((banner_h, match_canvas.shape[1], 3), dtype=np.uint8)
        banner_color = (34, 197, 94) if is_match else (68, 68, 239)  # Green or Red in BGR

        verdict_text = "VERDICT: SAME CATTLE (DUPLICATE MATCH)" if is_match else "VERDICT: DIFFERENT CATTLE (DISTINCT)"
        stats_text = f"Matches: {good_matches}/{threshold} | Sim: {similarity_percent:.1f}% | KPs: ({num_kp1}, {num_kp2})"

        cv2.putText(
            banner, verdict_text, (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2, cv2.LINE_AA
        )
        cv2.putText(
            banner, stats_text, (16, 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA
        )

        final_composite = np.vstack([banner, match_canvas])
        cv2.imwrite(save_visual_path, final_composite)
        visual_saved = str(Path(save_visual_path).resolve())

    return {
        "good_matches": good_matches,
        "threshold": threshold,
        "is_match": is_match,
        "verdict": verdict,
        "similarity_percent": round(similarity_percent, 1),
        "kp1_count": num_kp1,
        "kp2_count": num_kp2,
        "visual_path": visual_saved,
    }


class MuzzleMatcher:
    """
    Biometric Verification Engine comparing two cattle videos or images.
    Combines motion blur pre-filtering, 2-core CPU Mask R-CNN detection,
    early-exit pristine selection, Lanczos-4 super-resolution, CLAHE enhancement,
    and ORB keypoint matching.
    """

    def __init__(
        self,
        cropper: Optional[MuzzleCropper] = None,
        weights_path: Optional[str] = None,
    ):
        config.enforce_cpu_lock()
        self.cropper = cropper or MuzzleCropper(weights_path=weights_path)

    def extract_pristine_crop(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        video_tag: str = "A",
        frame_interval: int = config.FRAME_INTERVAL,
        max_frames: int = config.MAX_CANDIDATE_FRAMES,
        blur_threshold: float = config.BLUR_VARIANCE_THRESHOLD,
        pristine_threshold: float = config.PRISTINE_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Extracts candidate frames from video, pre-filters motion blur, detects muzzles,
        and returns the single highest-quality pristine muzzle crop.
        Triggers early exit immediately if confidence >= pristine_threshold.
        """
        video_path_obj = Path(video_path).resolve()
        if not video_path_obj.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_idx = 0
        sampled_count = 0
        all_candidate_crops = []

        logger.info(
            f"[MuzzleMatcher] Scanning Video {video_tag} ({video_path_obj.name}) "
            f"for pristine muzzle (Early-exit threshold: {pristine_threshold*100:.0f}%)..."
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # 1. Fast motion blur pre-filter (<0.5ms)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_var = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
                if blur_var < blur_threshold:
                    frame_idx += 1
                    continue

                # 2. Crop cattle muzzle using 2-core CPU Mask R-CNN
                res = self.cropper.crop_image(
                    frame,
                    output_dir=output_dir,
                    padding_pct=config.CROP_PADDING_PCT,
                    make_square=True,
                    target_size=config.CROP_SIZE,
                    save_annotated=False,
                )

                if res["detections"]:
                    for det in res["detections"]:
                        crop_bgr = det["cropped_bgr"]
                        quality = compute_image_quality(crop_bgr, det["confidence"])
                        item = {
                            "video_tag": video_tag,
                            "frame_idx": frame_idx,
                            "confidence": float(det["confidence"]),
                            "crop_bgr": crop_bgr,
                            "crop_path": det.get("saved_crop_path"),
                            "quality": quality,
                            "blur_var": blur_var,
                        }
                        all_candidate_crops.append(item)

                sampled_count += 1

                # 3. Smart Early Exit: pristine muzzle acquired
                if all_candidate_crops:
                    pristine = [c for c in all_candidate_crops if c["confidence"] >= pristine_threshold]
                    if pristine:
                        logger.info(
                            f"[MuzzleMatcher] Video {video_tag}: Acquired pristine muzzle crop at Frame #{frame_idx:04d} "
                            f"(Conf: {pristine[0]['confidence']*100:.1f}%, Blur Var: {blur_var:.1f}). "
                            f"Early exit triggered!"
                        )
                        break

                if max_frames and sampled_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        # Safety Fallback: scan beginning frames if none found
        if not all_candidate_crops:
            logger.warning(f"[MuzzleMatcher] Video {video_tag}: No muzzle found during normal sampling. Running fallback scan...")
            cap_fb = cv2.VideoCapture(str(video_path_obj))
            fb_idx = 0
            while cap_fb.isOpened() and fb_idx < 30:
                ret, frame = cap_fb.read()
                if not ret:
                    break
                if fb_idx % 5 == 0:
                    res = self.cropper.crop_image(
                        frame,
                        output_dir=output_dir,
                        padding_pct=config.CROP_PADDING_PCT,
                        make_square=True,
                        target_size=config.CROP_SIZE,
                        save_annotated=False,
                    )
                    if res["detections"]:
                        for det in res["detections"]:
                            crop_bgr = det["cropped_bgr"]
                            quality = compute_image_quality(crop_bgr, det["confidence"])
                            all_candidate_crops.append({
                                "video_tag": video_tag,
                                "frame_idx": fb_idx,
                                "confidence": float(det["confidence"]),
                                "crop_bgr": crop_bgr,
                                "crop_path": det.get("saved_crop_path"),
                                "quality": quality,
                                "blur_var": 0.0,
                            })
                        break
                fb_idx += 1
            cap_fb.release()

        if not all_candidate_crops:
            raise RuntimeError(f"Failed to detect any cattle muzzle in Video {video_tag}: {video_path}")

        # Rank candidate crops by composite quality score descending
        all_candidate_crops.sort(key=lambda c: c["quality"]["composite"], reverse=True)
        best_crop = all_candidate_crops[0]

        # Save pristine crop to output_dir if specified and not already saved
        if output_dir and not best_crop.get("crop_path"):
            os.makedirs(output_dir, exist_ok=True)
            crop_fname = f"pristine_muzzle_{video_tag}_f{best_crop['frame_idx']:04d}_conf{int(best_crop['confidence']*100)}.jpg"
            saved_crop_path = os.path.join(output_dir, crop_fname)
            cv2.imwrite(saved_crop_path, best_crop["crop_bgr"])
            best_crop["crop_path"] = saved_crop_path

        logger.info(
            f"[MuzzleMatcher] Video {video_tag} pristine crop selected: "
            f"Frame #{best_crop['frame_idx']:04d} | Confidence: {best_crop['confidence']*100:.1f}% | "
            f"Composite Quality: {best_crop['quality']['composite']}/100"
        )
        return best_crop

    def compare_videos(
        self,
        video_a_path: str,
        video_b_path: str,
        output_dir: Optional[str] = None,
        threshold: int = config.DUPLICATE_DECISION_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Performs end-to-end biometric verification between two cattle videos:
        1. Extracts pristine muzzle crops from Video A and Video B
        2. Super-resolves (2x Lanczos-4) and enhances dermatoglyphic ridges (CLAHE + unsharp mask)
        3. Matches biometric prints via ORB + Lowe ratio test
        4. Generates side-by-side visualization diagram with verdict banner
        """
        t_start = time.perf_counter()
        out_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        logger.info("=" * 70)
        logger.info(" STARTING CATTLE MUZZLE BIOMETRIC VERIFICATION")
        logger.info(f" Video A: {video_a_path}")
        logger.info(f" Video B: {video_b_path}")
        logger.info(f" Decision Threshold: {threshold} matches | Hardware: 2 CPU Cores")
        logger.info("=" * 70)

        # 1. Extract pristine crops
        t0 = time.perf_counter()
        crop_a = self.extract_pristine_crop(video_a_path, output_dir=out_dir, video_tag="A")
        crop_b = self.extract_pristine_crop(video_b_path, output_dir=out_dir, video_tag="B")
        t_extract_ms = (time.perf_counter() - t0) * 1000

        # 2. Super-resolution & enhancement
        t0 = time.perf_counter()
        gray_a, upscaled_a = enhance_muzzle_image(crop_a["crop_bgr"])
        gray_b, upscaled_b = enhance_muzzle_image(crop_b["crop_bgr"])

        # Save enhanced prints
        enhanced_path_a = os.path.join(out_dir, "enhanced_print_A.jpg")
        enhanced_path_b = os.path.join(out_dir, "enhanced_print_B.jpg")
        cv2.imwrite(enhanced_path_a, gray_a)
        cv2.imwrite(enhanced_path_b, gray_b)
        t_enhance_ms = (time.perf_counter() - t0) * 1000

        # 3. Match biometric prints
        t0 = time.perf_counter()
        visual_path = os.path.join(out_dir, "comparison_match_diagram.jpg")
        match_res = match_muzzle_features(
            gray_a, gray_b,
            n_features=config.ORB_N_FEATURES,
            ratio_test=config.ORB_RATIO_TEST,
            threshold=threshold,
            bgr1=upscaled_a,
            bgr2=upscaled_b,
            save_visual_path=visual_path,
        )
        t_match_ms = (time.perf_counter() - t0) * 1000
        total_time_ms = (time.perf_counter() - t_start) * 1000

        logger.info("-" * 70)
        logger.info(
            f" VERDICT: {match_res['verdict']} | Good Matches: {match_res['good_matches']}/{threshold} | "
            f"Similarity: {match_res['similarity_percent']:.1f}% | Total Time: {total_time_ms:.1f} ms"
        )
        logger.info(f" Comparison diagram saved to: {visual_path}")
        logger.info("-" * 70)

        return {
            "is_match": match_res["is_match"],
            "verdict": match_res["verdict"],
            "good_matches": match_res["good_matches"],
            "threshold": threshold,
            "similarity_percent": match_res["similarity_percent"],
            "kp1_count": match_res["kp1_count"],
            "kp2_count": match_res["kp2_count"],
            "video_a": str(video_a_path),
            "video_b": str(video_b_path),
            "crop_a_path": crop_a.get("crop_path"),
            "crop_b_path": crop_b.get("crop_path"),
            "enhanced_print_a_path": enhanced_path_a,
            "enhanced_print_b_path": enhanced_path_b,
            "visual_diagram_path": match_res["visual_path"],
            "crop_a_quality": crop_a["quality"],
            "crop_b_quality": crop_b["quality"],
            "timings": {
                "t_extract_ms": round(t_extract_ms, 1),
                "t_enhance_ms": round(t_enhance_ms, 1),
                "t_match_ms": round(t_match_ms, 1),
                "t_total_ms": round(total_time_ms, 1),
            },
        }

    def compare_images(
        self,
        image_a_input: Union[str, Path, np.ndarray],
        image_b_input: Union[str, Path, np.ndarray],
        output_dir: Optional[str] = None,
        threshold: int = config.DUPLICATE_DECISION_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Compares two cattle muzzle images or arrays directly.
        """
        t_start = time.perf_counter()
        out_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        res_a = self.cropper.crop_image(image_a_input, output_dir=out_dir, save_annotated=False)
        res_b = self.cropper.crop_image(image_b_input, output_dir=out_dir, save_annotated=False)

        if not res_a["detections"]:
            raise RuntimeError(f"No cattle muzzle detected in Image A: {image_a_input}")
        if not res_b["detections"]:
            raise RuntimeError(f"No cattle muzzle detected in Image B: {image_b_input}")

        crop_a_bgr = res_a["detections"][0]["cropped_bgr"]
        crop_b_bgr = res_b["detections"][0]["cropped_bgr"]

        gray_a, upscaled_a = enhance_muzzle_image(crop_a_bgr)
        gray_b, upscaled_b = enhance_muzzle_image(crop_b_bgr)

        visual_path = os.path.join(out_dir, "image_comparison_diagram.jpg")
        match_res = match_muzzle_features(
            gray_a, gray_b,
            n_features=config.ORB_N_FEATURES,
            ratio_test=config.ORB_RATIO_TEST,
            threshold=threshold,
            bgr1=upscaled_a,
            bgr2=upscaled_b,
            save_visual_path=visual_path,
        )

        total_time_ms = (time.perf_counter() - t_start) * 1000

        return {
            "is_match": match_res["is_match"],
            "verdict": match_res["verdict"],
            "good_matches": match_res["good_matches"],
            "threshold": threshold,
            "similarity_percent": match_res["similarity_percent"],
            "kp1_count": match_res["kp1_count"],
            "kp2_count": match_res["kp2_count"],
            "visual_diagram_path": match_res["visual_path"],
            "timings": {
                "t_total_ms": round(total_time_ms, 1),
            },
        }
