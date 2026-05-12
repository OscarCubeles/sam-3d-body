"""
Performance Benchmark for Pose Estimators
Measures inference time and FPS for:
1. YOLO11n-pose
2. YOLO11s-pose
3. SAM3D-Body

Takes 100 measurements per model and stores results in a CSV file.
"""

import torch
import cv2
import numpy as np
import csv
import time
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, List
from ultralytics import YOLO
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# GPU monitoring
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
    GPU_MONITORING_AVAILABLE = True
    nvmlInit()
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    print("Warning: pynvml not found. GPU monitoring disabled. Install with: pip install nvidia-ml-py")

# ==========================================
# CONFIGURATION
# ==========================================

# Choose which model to benchmark
# Options: "yolo11n", "yolo11s", "sam3d"
MODEL_TO_BENCHMARK = "sam3d"

# Number of measurements per model
NUM_MEASUREMENTS = 100

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output CSV file
OUTPUT_CSV = "pose_estimator_performance.csv"

# Model paths
YOLO11N_PATH = r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\yolo11n-pose.pt"
YOLO11S_PATH = r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\yolo11s-pose.pt"
SAM3D_CHECKPOINT = r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\checkpoints\sam-3d-body-dinov3\model.ckpt"
SAM3D_MHR_PATH = r"C:\Users\oscar\Documents\TU\TFM\sam-3d-body\checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt"


# ==========================================
# MODEL LOADERS
# ==========================================

def load_yolo_model(model_path: str) -> Callable:
    """Load a YOLO pose model."""
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    model.to(DEVICE)
    print(f"YOLO model loaded on {DEVICE}")
    return model


def load_sam3d_model() -> Tuple[Any, Dict, SAM3DBodyEstimator]:
    """Load SAM3D-Body model."""
    print("Loading SAM3D-Body model...")
    model, model_cfg = load_sam_3d_body(
        SAM3D_CHECKPOINT,
        device=DEVICE,
        mhr_path=SAM3D_MHR_PATH
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    print("SAM3D-Body model loaded")
    return model, model_cfg, estimator


# ==========================================
# INFERENCE WRAPPERS
# ==========================================

def infer_yolo(model: YOLO, frame: np.ndarray) -> np.ndarray:
    """Run inference on a frame using YOLO."""
    results = model.predict(frame, verbose=False, conf=0.5)
    return results[0] if results else None


def infer_sam3d(estimator: SAM3DBodyEstimator, frame_rgb: np.ndarray) -> Dict:
    """Run inference on a frame using SAM3D-Body."""
    outputs = estimator.process_one_image(frame_rgb)
    return outputs[0] if (outputs and len(outputs) > 0) else None


# ==========================================
# GPU MONITORING
# ==========================================

def get_gpu_stats() -> Dict[str, float]:
    """Get current GPU memory and utilization statistics."""
    if not GPU_MONITORING_AVAILABLE:
        return {"gpu_memory_mb": 0, "gpu_util_percent": 0}
    
    try:
        handle = nvmlDeviceGetHandleByIndex(0)  # First GPU
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_rates = nvmlDeviceGetUtilizationRates(handle)
        
        gpu_memory_mb = mem_info.used / 1024 / 1024  # Convert bytes to MB
        gpu_util_percent = util_rates.gpu
        
        return {
            "gpu_memory_mb": round(gpu_memory_mb, 2),
            "gpu_util_percent": round(gpu_util_percent, 1)
        }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return {"gpu_memory_mb": 0, "gpu_util_percent": 0}


# ==========================================
# BENCHMARK FUNCTION
# ==========================================

def benchmark_model(
    model_name: str,
    inference_func: Callable,
    model: Any,
    num_measurements: int = 100,
) -> List[Dict[str, Any]]:
    """
    Benchmark a model using webcam input.
    
    Args:
        model_name: Name of the model being benchmarked
        inference_func: Function to call for inference
        model: The model object (YOLO or SAM3DBodyEstimator)
        num_measurements: Number of frames to measure
        
    Returns:
        List of measurement dictionaries
    """
    measurements = []
    
    print(f"\n{'='*60}")
    print(f"Starting benchmark for {model_name}")
    print(f"Target measurements: {num_measurements}")
    print(f"{'='*60}\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam")
        return measurements
    
    # Set webcam resolution for consistency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    measurement_count = 0
    
    try:
        while measurement_count < num_measurements:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                continue
            
            frame_count += 1
            
            # Prepare frame based on model type
            if model_name == "sam3d":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_frame = frame_rgb
            else:
                input_frame = frame
            
            # Record time before inference
            start_time = time.perf_counter()
            
            # Run inference
            try:
                if model_name == "sam3d":
                    result = inference_func(model, input_frame)
                else:
                    result = inference_func(model, input_frame)
            except Exception as e:
                print(f"Inference error on frame {frame_count}: {e}")
                continue
            
            # Record time after inference
            end_time = time.perf_counter()
            
            # Get GPU stats
            gpu_stats = get_gpu_stats()
            
            # Calculate metrics
            inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Store measurement
            measurement = {
                "model": model_name,
                "frame_number": measurement_count + 1,
                "inference_time_ms": round(inference_time_ms, 4),
                "fps": round(fps, 2),
                "gpu_memory_mb": gpu_stats["gpu_memory_mb"],
                "gpu_util_percent": gpu_stats["gpu_util_percent"],
            }
            measurements.append(measurement)
            measurement_count += 1
            
            # Print progress
            if measurement_count % 10 == 0:
                print(f"Progress: {measurement_count}/{num_measurements} measurements")
                print(f"  Latest inference time: {inference_time_ms:.4f}ms, FPS: {fps:.2f}")
                print(f"  GPU Memory: {gpu_stats['gpu_memory_mb']:.2f}MB, GPU Util: {gpu_stats['gpu_util_percent']:.1f}%")
            
            # Visualize keypoints on frame
            frame_vis = frame.copy()
            
            if result is not None:
                if model_name == "sam3d":
                    # SAM3D keypoints
                    if "pred_keypoints_2d" in result:
                        keypoints_2d = result["pred_keypoints_2d"]
                        for x, y in keypoints_2d:
                            cv2.circle(frame_vis, (int(x), int(y)), 4, (0, 255, 0), -1)
                
                else:
                    # YOLO keypoints
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy
                        if len(keypoints) > 0:
                            for keypoint_set in keypoints:
                                for x, y in keypoint_set:
                                    if x > 0 and y > 0:  # Valid keypoint
                                        cv2.circle(frame_vis, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # Display frame with measurements
            cv2.putText(
                frame_vis,
                f"{model_name} - Frame {measurement_count}/{num_measurements}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_vis,
                f"Time: {inference_time_ms:.2f}ms | FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_vis,
                f"GPU Memory: {gpu_stats['gpu_memory_mb']:.0f}MB | GPU Util: {gpu_stats['gpu_util_percent']:.1f}%",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            cv2.imshow(f"Benchmarking {model_name}", frame_vis)
            
            # Allow early exit with ESC
            if cv2.waitKey(1) & 0xFF == 27:
                print("\nBenchmark interrupted by user")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary statistics
    if measurements:
        inference_times = [m["inference_time_ms"] for m in measurements]
        fps_values = [m["fps"] for m in measurements]
        gpu_memory = [m["gpu_memory_mb"] for m in measurements]
        gpu_util = [m["gpu_util_percent"] for m in measurements]
        
        print(f"\n{model_name} - Summary Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Successful measurements: {measurement_count}")
        print(f"\n  Inference Time:")
        print(f"    Min: {min(inference_times):.4f}ms")
        print(f"    Max: {max(inference_times):.4f}ms")
        print(f"    Mean: {np.mean(inference_times):.4f}ms")
        print(f"    Std: {np.std(inference_times):.4f}ms")
        print(f"\n  FPS:")
        print(f"    Mean: {np.mean(fps_values):.2f}")
        print(f"    Min: {min(fps_values):.2f}")
        print(f"    Max: {max(fps_values):.2f}")
        print(f"\n  GPU Memory (MB):")
        print(f"    Mean: {np.mean(gpu_memory):.2f}MB")
        print(f"    Min: {min(gpu_memory):.2f}MB")
        print(f"    Max: {max(gpu_memory):.2f}MB")
        print(f"\n  GPU Utilization (%):")
        print(f"    Mean: {np.mean(gpu_util):.1f}%")
        print(f"    Min: {min(gpu_util):.1f}%")
        print(f"    Max: {max(gpu_util):.1f}%")
    
    return measurements


# ==========================================
# CSV STORAGE
# ==========================================

def save_measurements_to_csv(measurements: List[Dict[str, Any]], output_file: str):
    """
    Append measurements to CSV file.
    Creates file with headers if it doesn't exist.
    """
    file_exists = Path(output_file).exists()
    
    try:
        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ["model", "frame_number", "inference_time_ms", "fps", "gpu_memory_mb", "gpu_util_percent"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write headers only if file is new
            if not file_exists:
                writer.writeheader()
                print(f"\nCreated new CSV file: {output_file}")
            
            # Write measurements
            writer.writerows(measurements)
            print(f"Saved {len(measurements)} measurements to {output_file}")
            
    except Exception as e:
        print(f"Error saving to CSV: {e}")


# ==========================================
# MAIN BENCHMARK PIPELINE
# ==========================================

def main():
    """Main benchmark pipeline."""
    
    print("\n" + "="*60)
    print("POSE ESTIMATOR PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Selected model: {MODEL_TO_BENCHMARK}")
    print(f"Measurements per model: {NUM_MEASUREMENTS}")
    print(f"Device: {DEVICE}")
    print(f"Output file: {OUTPUT_CSV}")
    
    all_measurements = []
    
    # Determine which model to benchmark
    if MODEL_TO_BENCHMARK.lower() == "yolo11n":
        print("\nLoading YOLO11n model...")
        model = load_yolo_model(YOLO11N_PATH)
        measurements = benchmark_model(
            "yolo11n",
            infer_yolo,
            model,
            NUM_MEASUREMENTS
        )
        all_measurements.extend(measurements)
    
    elif MODEL_TO_BENCHMARK.lower() == "yolo11s":
        print("\nLoading YOLO11s model...")
        model = load_yolo_model(YOLO11S_PATH)
        measurements = benchmark_model(
            "yolo11s",
            infer_yolo,
            model,
            NUM_MEASUREMENTS
        )
        all_measurements.extend(measurements)
    
    elif MODEL_TO_BENCHMARK.lower() == "sam3d":
        print("\nLoading SAM3D-Body model...")
        _, _, estimator = load_sam3d_model()
        measurements = benchmark_model(
            "sam3d",
            infer_sam3d,
            estimator,
            NUM_MEASUREMENTS
        )
        all_measurements.extend(measurements)
    
    else:
        print(f"Error: Unknown model '{MODEL_TO_BENCHMARK}'")
        print("Available options: yolo11n, yolo11s, sam3d")
        return
    
    # Save results
    if all_measurements:
        save_measurements_to_csv(all_measurements, OUTPUT_CSV)
        print(f"\nBenchmark completed successfully!")
    else:
        print("\nNo measurements collected.")


if __name__ == "__main__":
    main()
