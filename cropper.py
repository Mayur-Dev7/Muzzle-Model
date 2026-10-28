"""
Cattle Muzzle Cropper Module
============================
Standalone Mask R-CNN cattle muzzle detector and cropper.
Enforces 2 CPU cores affinity, disables GPU execution, and produces
normalized square (224, 224) muzzle crops for biometric identification.
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

# Import centralized configuration & enforce hardware lock BEFORE TensorFlow imports.
# Validate native Apple Silicon Python / supported TensorFlow before importing TF.
import config
config.enforce_cpu_lock()
config.validate_tensorflow_runtime()

import cv2
import numpy as np

# Suppress TF logging and disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
try:
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(config.CPU_CORES)
    tf.config.threading.set_inter_op_parallelism_threads(config.CPU_CORES)
except Exception:
    pass

import mrcnn.model as modellib
from config import CowMuzzleInferenceConfig


logger = logging.getLogger("muzzle_cropper")
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


class MuzzleCropper:
    """
    Mask R-CNN Cattle Muzzle Detection & Cropping Engine.
    Strictly locked to 2 CPU cores and optimized for (224, 224) biometric crops.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD,
    ):
        self.weights_path = Path(weights_path or config.WEIGHTS_PATH).resolve()
        self.confidence_threshold = confidence_threshold

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at: {self.weights_path}. "
                f"Please ensure weights are placed in weights/crop11dec.h5"
            )

        # Enforce CPU affinity & threading at initialization
        config.enforce_cpu_lock()

        # Build inference configuration
        self.mrcnn_config = CowMuzzleInferenceConfig()
        self.mrcnn_config.DETECTION_MIN_CONFIDENCE = self.confidence_threshold
        self.mrcnn_config.IMAGE_MIN_DIM = config.IMAGE_MIN_DIM
        self.mrcnn_config.IMAGE_MAX_DIM = config.IMAGE_MAX_DIM
        self.mrcnn_config.POST_NMS_ROIS_INFERENCE = config.POST_NMS_ROIS_INFERENCE

        t0 = time.perf_counter()
        logger.info(
            f"[MuzzleCropper] Initializing Mask R-CNN (2 CPU Cores, "
            f"{config.IMAGE_MIN_DIM}x{config.IMAGE_MAX_DIM} resolution) from {self.weights_path.name}..."
        )

        self.model = modellib.MaskRCNN(
            mode="inference",
            config=self.mrcnn_config,
            model_dir=str(PROJECT_ROOT),
        )
        self.model.load_weights(str(self.weights_path), by_name=True)
        t_load = (time.perf_counter() - t0) * 1000
        logger.info(f"[MuzzleCropper] Model loaded successfully in {t_load:.1f} ms")

    def crop_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_dir: Optional[str] = None,
        padding_pct: float = config.CROP_PADDING_PCT,
        make_square: bool = True,
        target_size: Tuple[int, int] = config.CROP_SIZE,
        save_annotated: bool = True,
    ) -> Dict[str, Any]:
        """
        Detects and crops cattle muzzle from an image file path or BGR/RGB numpy array.

        Returns a dictionary with:
        - 'source': source name or file stem
        - 'detections': list of detection records
        - 'timings': breakdown of load, preprocess, inference, and crop timings (ms)
        - 'image_bgr': original BGR image
        - 'saved_crop_paths': list of filepaths to saved crops (if output_dir is given)
        - 'saved_annotated_path': filepath to annotated visualization (if output_dir is given)
        """
        timings = {}
        t_total_start = time.perf_counter()

        # Step 1: Ingest image
        t0 = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            src_path = Path(image_input).resolve()
            if not src_path.is_file():
                raise FileNotFoundError(f"Input image not found: {src_path}")
            image_bgr = cv2.imread(str(src_path))
            if image_bgr is None:
                raise ValueError(f"cv2.imread failed to read: {src_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            source_name = src_path.stem
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image_input.copy()
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            source_name = "frame"
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")
        timings["t_load_ms"] = (time.perf_counter() - t0) * 1000

        orig_h, orig_w = image_rgb.shape[:2]

        # Step 2: Preprocess / Mold input
        t0 = time.perf_counter()
        molded_images, image_metas, windows = self.model.mold_inputs([image_rgb])
        molded_shape = molded_images[0].shape
        win = windows[0]
        timings["t_preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Step 3: Neural network forward inference
        t0 = time.perf_counter()
        anchors = self.model.get_anchors(molded_shape)
        anchors = np.broadcast_to(anchors, (self.mrcnn_config.BATCH_SIZE,) + anchors.shape)
        detections, _, _, mrcnn_mask, _, _, _ = self.model.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )
        timings["t_inference_ms"] = (time.perf_counter() - t0) * 1000

        # Step 4: Post-process & Unmold detections
        t0 = time.perf_counter()
        rois, class_ids, scores, masks = self.model.unmold_detections(
            detections[0], mrcnn_mask[0], image_rgb.shape, molded_shape, win
        )
        timings["t_postprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Step 5: Crop and normalize detections
        t0 = time.perf_counter()
        detected_crops = []
        for idx in range(len(rois)):
            score = float(scores[idx])
            if score < self.confidence_threshold:
                continue

            y1, x1, y2, x2 = rois[idx]
            box_h = y2 - y1
            box_w = x2 - x1

            # Padding
            pad_y = int(box_h * padding_pct)
            pad_x = int(box_w * padding_pct)
            cy1 = max(0, y1 - pad_y)
            cx1 = max(0, x1 - pad_x)
            cy2 = min(orig_h, y2 + pad_y)
            cx2 = min(orig_w, x2 + pad_x)

            # Square aspect ratio normalization
            if make_square:
                ch = cy2 - cy1
                cw = cx2 - cx1
                diff = abs(ch - cw)
                if ch < cw:
                    pad_top = diff // 2
                    pad_bot = diff - pad_top
                    cy1 = max(0, cy1 - pad_top)
                    cy2 = min(orig_h, cy2 + pad_bot)
                    ch = cy2 - cy1
                    if ch < cw and cy1 > 0:
                        cy1 = max(0, cy2 - cw)
                    elif ch < cw and cy2 < orig_h:
                        cy2 = min(orig_h, cy1 + cw)
                elif cw < ch:
                    pad_left = diff // 2
                    pad_right = diff - pad_left
                    cx1 = max(0, cx1 - pad_left)
                    cx2 = min(orig_w, cx2 + pad_right)
                    cw = cx2 - cx1
                    if cw < ch and cx1 > 0:
                        cx1 = max(0, cx2 - ch)
                    elif cw < ch and cx2 < orig_w:
                        cx2 = min(orig_w, cx1 + ch)

            # Extract patch
            crop_bgr = image_bgr[cy1:cy2, cx1:cx2].copy()
            crop_rgb = image_rgb[cy1:cy2, cx1:cx2].copy()

            # Target resize to 224x224
            if target_size:
                crop_bgr = cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_AREA)
                crop_rgb = cv2.resize(crop_rgb, target_size, interpolation=cv2.INTER_AREA)

            mask_patch = masks[:, :, idx] if masks is not None and masks.shape[-1] > idx else None

            detected_crops.append({
                "muzzle_index": idx,
                "confidence": score,
                "class_id": int(class_ids[idx]),
                "bbox_original": [int(y1), int(x1), int(y2), int(x2)],
                "bbox_cropped": [int(cy1), int(cx1), int(cy2), int(cx2)],
                "crop_size": list(target_size) if target_size else [int(cy2 - cy1), int(cx2 - cx1)],
                "cropped_bgr": crop_bgr,
                "cropped_rgb": crop_rgb,
                "mask": mask_patch,
            })

        timings["t_crop_ms"] = (time.perf_counter() - t0) * 1000
        timings["t_total_ms"] = (time.perf_counter() - t_total_start) * 1000
        timings["fps"] = (1000.0 / timings["t_total_ms"]) if timings["t_total_ms"] > 0 else 0.0

        result = {
            "source": source_name,
            "original_shape": [orig_h, orig_w, 3],
            "detections": detected_crops,
            "timings": timings,
            "image_bgr": image_bgr,
            "saved_crop_paths": [],
            "saved_annotated_path": None,
        }

        # Save outputs if directory specified
        if output_dir:
            self.save_results(result, output_dir=output_dir, save_annotated=save_annotated)

        return result

    def save_results(
        self,
        processed_result: Dict[str, Any],
        output_dir: str,
        save_annotated: bool = True,
    ) -> List[str]:
        """Saves cropped muzzles and annotated preview image."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = processed_result.get("source", "crop")
        saved_paths = []

        # 1. Save Crops
        for det in processed_result["detections"]:
            idx = det["muzzle_index"]
            conf = int(det["confidence"] * 100)
            crop_filename = f"{base_name}_muzzle_{idx}_conf{conf}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, det["cropped_bgr"])
            det["saved_crop_path"] = crop_path
            saved_paths.append(crop_path)

        processed_result["saved_crop_paths"] = saved_paths

        # 2. Save Annotated Visualization
        if save_annotated and len(processed_result["detections"]) > 0:
            annotated = processed_result["image_bgr"].copy()
            for det in processed_result["detections"]:
                y1, x1, y2, x2 = det["bbox_original"]
                conf_pct = det["confidence"] * 100

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 3)

                # Label banner
                label = f"Muzzle: {conf_pct:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (34, 197, 94), -1)
                cv2.putText(
                    annotated, label, (x1 + 5, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
                )

                # Mask overlay
                mask = det.get("mask")
                if mask is not None:
                    color_mask = np.zeros_like(annotated, dtype=np.uint8)
                    color_mask[mask] = (0, 165, 255)
                    annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.35, 0)

            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated)
            processed_result["saved_annotated_path"] = annotated_path

        return saved_paths

    def crop_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        frame_interval: int = config.FRAME_INTERVAL,
        max_frames: Optional[int] = config.MAX_CANDIDATE_FRAMES,
        blur_threshold: float = config.BLUR_VARIANCE_THRESHOLD,
        padding_pct: float = config.CROP_PADDING_PCT,
        make_square: bool = True,
        target_size: Tuple[int, int] = config.CROP_SIZE,
        early_exit_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Samples video frames, filters motion blur, and extracts cattle muzzle crops.
        """
        video_path_obj = Path(video_path).resolve()
        if not video_path_obj.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"cv2.VideoCapture failed to open: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_s = total_frames / fps if fps > 0 else 0

        logger.info(
            f"[MuzzleCropper] Processing video: {video_path_obj.name} "
            f"({total_frames} frames, {fps:.1f} FPS, {duration_s:.1f}s) | "
            f"Frame interval: {frame_interval}, Max candidates: {max_frames}"
        )

        all_detections = []
        all_crop_paths = []
        frame_idx = 0
        evaluated_candidates = 0
        t_start = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Motion blur pre-filtering
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blur_var < blur_threshold:
                    logger.debug(f"Frame #{frame_idx:04d}: skipped due to motion blur (var: {blur_var:.1f})")
                    frame_idx += 1
                    continue

                frame_source = f"{video_path_obj.stem}_frame_{frame_idx:04d}"
                res = self.crop_image(
                    frame,
                    output_dir=output_dir,
                    padding_pct=padding_pct,
                    make_square=make_square,
                    target_size=target_size,
                    save_annotated=True,
                )
                res["source"] = frame_source

                num_found = len(res["detections"])
                timings = res["timings"]
                logger.info(
                    f"Frame #{frame_idx:04d}: {num_found} muzzle(s) found | "
                    f"Inference: {timings['t_inference_ms']:.1f} ms | "
                    f"Total: {timings['t_total_ms']:.1f} ms"
                )

                for det in res["detections"]:
                    det["frame_idx"] = frame_idx
                    det["blur_var"] = blur_var
                    all_detections.append(det)

                all_crop_paths.extend(res.get("saved_crop_paths", []))
                evaluated_candidates += 1

                # Early-exit on pristine confidence crop
                if early_exit_confidence is not None:
                    pristine = [d for d in res["detections"] if d["confidence"] >= early_exit_confidence]
                    if pristine:
                        logger.info(
                            f"[MuzzleCropper] Pristine crop acquired at Frame #{frame_idx:04d} "
                            f"(Conf: {pristine[0]['confidence']*100:.1f}%). Early-exit triggered!"
                        )
                        break

                if max_frames and evaluated_candidates >= max_frames:
                    logger.info(f"[MuzzleCropper] Reached max candidate frames limit ({max_frames})")
                    break

            frame_idx += 1

        cap.release()
        total_time_ms = (time.perf_counter() - t_start) * 1000

        # Safety Fallback: if no crops found in sampled frames, scan first 30 frames every 5 frames
        if not all_detections:
            logger.warning("[MuzzleCropper] No muzzles found in initial sampling. Executing fallback scan...")
            cap_fb = cv2.VideoCapture(str(video_path_obj))
            fb_idx = 0
            while cap_fb.isOpened() and fb_idx < 30:
                ret, frame = cap_fb.read()
                if not ret:
                    break
                if fb_idx % 5 == 0:
                    res = self.crop_image(
                        frame,
                        output_dir=output_dir,
                        padding_pct=padding_pct,
                        make_square=make_square,
                        target_size=target_size,
                        save_annotated=True,
                    )
                    if res["detections"]:
                        for det in res["detections"]:
                            det["frame_idx"] = fb_idx
                            all_detections.append(det)
                        all_crop_paths.extend(res.get("saved_crop_paths", []))
                        logger.info(f"[MuzzleCropper] Fallback scan found {len(res['detections'])} muzzle(s) at Frame #{fb_idx:04d}")
                        break
                fb_idx += 1
            cap_fb.release()

        return {
            "video_path": str(video_path_obj),
            "total_frames": total_frames,
            "evaluated_candidates": evaluated_candidates,
            "detections": all_detections,
            "crop_paths": all_crop_paths,
            "total_time_ms": total_time_ms,
        }
