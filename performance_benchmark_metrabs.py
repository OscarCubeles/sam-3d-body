"""
Performance Benchmark for Metrabs Pose Estimator
Measures inference time and FPS for Metrabs model.

Takes 100 measurements using webcam input and stores results in a CSV file.
"""

import torch
import cv2
import numpy as np
import csv
import time
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, List

import settings
from calibration_utils import load_metrabs_calibration
from metrabs_pytorch.inference import metrabsInference

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

# Number of measurements
NUM_MEASUREMENTS = 100

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output CSV file
OUTPUT_CSV = "metrabs_performance.csv"

# Model paths
METRABS_MODEL_DIR = Path(__file__).resolve().parent / "metrabs_eff2l_384px_800k_28ds_pytorch" # might need to change this and use media as conda environment


# ==========================================
# MODEL LOADER
# ==========================================

def load_metrabs_model() -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Metrabs model and camera calibration.
    
    Returns:
        Tuple of (model, intrinsic_matrix, distortion_coeffs, joint_names, joint_edges)
    """
    print(f"Loading Metrabs model from {METRABS_MODEL_DIR}...")
    
    intrinsic_matrix, distortion_coeffs = load_metrabs_calibration()
    
    metrabs_inference_model = metrabsInference.metrabs_inference(str(METRABS_MODEL_DIR))
    model = metrabs_inference_model.load_model()
    model.eval()
    
    joint_names = model.per_skeleton_joint_names[settings.SKELETON]
    joint_edges = model.per_skeleton_joint_edges[settings.SKELETON].cpu().numpy()
    
    print(f"Metrabs model loaded on {DEVICE}")
    print(f"Skeleton: {settings.SKELETON}")
    print(f"Number of joints: {len(joint_names)}")
    
    return model, intrinsic_matrix, distortion_coeffs, joint_names, joint_edges


# ==========================================
# INFERENCE WRAPPER
# ==========================================

def infer_metrabs(
    model: Any,
    frame: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray
) -> Dict[str, Any]:
    """Run inference on a frame using Metrabs.
    
    Args:
        model: Metrabs model instance
        frame: Input image as numpy array (BGR)
        intrinsic_matrix: Camera intrinsic matrix
        distortion_coeffs: Camera distortion coefficients
        
    Returns:
        Prediction dict with '3d_poses', '2d_poses' keys or None if failed
    """
    try:
        with torch.inference_mode(), torch.device("cuda"):
            image_pt = torch.from_numpy(frame).permute(2, 0, 1).cuda(non_blocking=True)
            pred = model.detect_poses(
                image_pt,
                intrinsic_matrix=intrinsic_matrix,
                distortion_coeffs=distortion_coeffs,
                detector_threshold=settings.DETECTOR_THRESHOLD,
                detector_nms_iou_threshold=settings.DETECTOR_NMS_IOU,
                max_detections=settings.MAX_DETECTIONS,
                skeleton=settings.SKELETON,
                num_aug=settings.NUM_AUG,
                antialias_factor=settings.ANTIALIAS_FACTOR,
                internal_batch_size=settings.INTERNAL_BATCH_SIZE,
                average_aug=settings.AVERAGE_AUG,
                suppress_implausible_poses=settings.SUPPRESS_IMPLAUSIBLE_POSES,
                detector_flip_aug=settings.DETECTOR_FLIP_AUG,
            )
        return pred
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


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

def benchmark_metrabs(
    model: Any,
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    joint_names: np.ndarray,
    joint_edges: np.ndarray,
    num_measurements: int = 100,
) -> List[Dict[str, Any]]:
    """
    Benchmark Metrabs using webcam input.
    
    Args:
        model: Metrabs model instance
        intrinsic_matrix: Camera intrinsic matrix
        distortion_coeffs: Camera distortion coefficients
        joint_names: Array of joint names
        joint_edges: Array of joint edges for visualization
        num_measurements: Number of frames to measure
        
    Returns:
        List of measurement dictionaries
    """
    measurements = []
    
    print(f"\n{'='*60}")
    print(f"Starting Metrabs benchmark")
    print(f"Target measurements: {num_measurements}")
    print(f"{'='*60}\n")
    
    # Open webcam
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {settings.CAMERA_INDEX}")
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
            
            # Record time before inference
            start_time = time.perf_counter()
            
            # Run inference
            try:
                result = infer_metrabs(model, frame, intrinsic_matrix, distortion_coeffs)
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
                "model": "metrabs",
                "frame_number": measurement_count + 1,
                "inference_time_ms": round(inference_time_ms, 4),
                "fps": round(fps, 2),
                "gpu_memory_mb": gpu_stats["gpu_memory_mb"],
                "gpu_util_percent": gpu_stats["gpu_util_percent"],
                "num_detections": len(result.get('poses3d', [])) if result else 0,
            }
            measurements.append(measurement)
            measurement_count += 1
            
            # Print progress
            if measurement_count % 10 == 0:
                print(f"Progress: {measurement_count}/{num_measurements} measurements")
                print(f"  Latest inference time: {inference_time_ms:.4f}ms, FPS: {fps:.2f}")
                print(f"  GPU Memory: {gpu_stats['gpu_memory_mb']:.2f}MB, GPU Util: {gpu_stats['gpu_util_percent']:.1f}%")
                if result:
                    print(f"  Detections: {measurement['num_detections']}")
            
            # Visualize keypoints on frame
            frame_vis = frame.copy()
            
            if result is not None and 'poses3d' in result:
                poses3d = result['poses3d']
                poses2d = result.get('poses2d', None)
                
                # Draw keypoints and skeleton
                if poses2d is not None:
                    for pose_idx, pose2d in enumerate(poses2d):
                        # Draw keypoints
                        for joint_idx, (x, y) in enumerate(pose2d):
                            if x > 0 and y > 0:  # Valid keypoint
                                cv2.circle(frame_vis, (int(x), int(y)), 4, (0, 255, 0), -1)
                        
                        # Draw skeleton connections
                        for edge in joint_edges:
                            start_idx, end_idx = int(edge[0]), int(edge[1])
                            if (start_idx < len(pose2d) and end_idx < len(pose2d)):
                                x1, y1 = pose2d[start_idx]
                                x2, y2 = pose2d[end_idx]
                                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                    cv2.line(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Display frame with measurements
            cv2.putText(
                frame_vis,
                f"Metrabs - Frame {measurement_count}/{num_measurements}",
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
            if result:
                cv2.putText(
                    frame_vis,
                    f"Detections: {measurement['num_detections']}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
            
            cv2.imshow("Metrabs Benchmark", frame_vis)
            
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
        detections = [m["num_detections"] for m in measurements]
        
        print(f"\nMetrabs - Summary Statistics:")
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
        print(f"\n  Detections per frame:")
        print(f"    Mean: {np.mean(detections):.1f}")
        print(f"    Min: {min(detections):.0f}")
        print(f"    Max: {max(detections):.0f}")
    
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
            fieldnames = ["model", "frame_number", "inference_time_ms", "fps", "gpu_memory_mb", "gpu_util_percent", "num_detections"]
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
    print("METRABS PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Measurements: {NUM_MEASUREMENTS}")
    print(f"Device: {DEVICE}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Model directory: {METRABS_MODEL_DIR}")
    
    # Check if model directory exists
    if not METRABS_MODEL_DIR.exists():
        print(f"\nError: Model directory not found at {METRABS_MODEL_DIR}")
        print("Make sure the Metrabs model is available at this location.")
        return
    
    # Load model
    print("\nLoading Metrabs model...")
    model, intrinsic_matrix, distortion_coeffs, joint_names, joint_edges = load_metrabs_model()
    
    # Run benchmark
    print("\nStarting benchmark...")
    measurements = benchmark_metrabs(
        model,
        intrinsic_matrix,
        distortion_coeffs,
        joint_names,
        joint_edges,
        NUM_MEASUREMENTS
    )
    
    # Save results
    if measurements:
        save_measurements_to_csv(measurements, OUTPUT_CSV)
        print(f"\nBenchmark completed successfully!")
    else:
        print("\nNo measurements collected.")


if __name__ == "__main__":
    main()
