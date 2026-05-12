"""
Performance Benchmark Visualization
Processes pose_estimator_performance.csv and creates multi-plot visualizations
with a green monochrome palette.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Options: "line", "histogram", "candle"
LEFT_PLOT_TYPE = "candle"

# ==========================================
# COLOR PALETTE
# ==========================================

COLOR_PALETTE = {
    "dark": "#00441B",
    "medium_dark": "#238B45",
    "medium": "#66C2A4",
    "light": "#B2E2E2",
}

MODEL_COLORS = {
    "yolo11n": COLOR_PALETTE["dark"],
    "yolo11s": COLOR_PALETTE["medium_dark"],
    "sam3d": COLOR_PALETTE["medium"],
    "metrabs": COLOR_PALETTE["light"],
}

# Model display names
MODEL_DISPLAY_NAMES = {
    "yolo11n": "Pose YOLO11n",
    "yolo11s": "Pose YOLO11s",
    "sam3d": "SAM3D-Body",
    "metrabs": "METRabs",
}

# ==========================================
# DATA LOADING & PROCESSING
# ==========================================

def load_performance_data(csv_file: str = "pose_estimator_performance.csv") -> pd.DataFrame:
    """Load performance data from CSV file."""
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} measurements from {csv_file}")
    print(f"Models: {df['model'].unique()}")
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per model."""
    stats = df.groupby("model").agg({
        "inference_time_ms": ["mean", "std", "min", "max"],
        "fps": ["mean", "std", "min", "max"],
        "gpu_memory_mb": ["mean", "std", "min", "max"],
        "gpu_util_percent": ["mean", "std", "min", "max"],
    }).round(2)
    
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS BY MODEL")
    print("="*80)
    print(stats)
    return stats


# ==========================================
# PLOTTING HELPER FUNCTIONS
# ==========================================

def plot_gpu_line_chart(ax, df, models, display_names, colors_list):
    """Line chart showing GPU utilization over frames per model."""
    for model, display_name, color in zip(models, display_names, colors_list):
        model_data = df[df["model"] == model].sort_values("frame_number")
        x_frames = range(1, len(model_data) + 1)
        ax.plot(x_frames, model_data["gpu_util_percent"].values, 
                label=display_name, color=color, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel("Frame Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("GPU Utilization (%)", fontsize=12, fontweight="bold")
    ax.set_title("GPU Utilization Over Frames (Per Model)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--", color=COLOR_PALETTE["medium"])
    ax.set_facecolor("#F8F9F8")


def plot_gpu_histogram(ax, df, models, display_names, colors_list):
    """Histogram showing GPU utilization distribution per model."""
    for model, color, name in zip(models, colors_list, display_names):
        model_data = df[df["model"] == model]["gpu_util_percent"]
        ax.hist(model_data, bins=20, alpha=0.6, label=name, color=color)
    
    ax.set_xlabel("GPU Utilization (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("GPU Utilization Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")


def plot_gpu_candlestick(ax, df, models, display_names, colors_list):
    """Candlestick chart showing GPU utilization statistics per model (Min, Q1, Median, Q3, Max)."""
    x_positions = np.arange(len(models))
    width = 0.5
    
    for idx, (model, display_name, color) in enumerate(zip(models, display_names, colors_list)):
        model_data = df[df["model"] == model]["gpu_util_percent"]
        
        # Calculate statistics (Open, High, Low, Close as Q1, Max, Min, Median)
        q1 = model_data.quantile(0.25)
        q3 = model_data.quantile(0.75)
        minimum = model_data.min()
        maximum = model_data.max()
        median = model_data.median()
        
        x_pos = x_positions[idx]
        
        # High-Low line (Min to Max)
        ax.plot([x_pos, x_pos], [minimum, maximum], 
               color=color, linewidth=2.5, alpha=0.9)
        
        # Box (Q1 to Q3)
        box_height = q3 - q1
        rect = plt.Rectangle((x_pos - width/2, q1), width, box_height,
                             facecolor=color, alpha=0.6, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
        
        # Median line
        ax.plot([x_pos - width/2, x_pos + width/2], [median, median], 
               color="red", linewidth=2.5, alpha=1.0)
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("GPU Utilization (%)", fontsize=12, fontweight="bold")
    ax.set_title("GPU Utilization Distribution (Candlestick: Min-Q1-Median-Q3-Max)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    
    # Add legend with custom entries
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.6, label=name) 
                      for color, name in zip(colors_list, display_names)]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.95)


# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_bubble_chart(df: pd.DataFrame, output_file: str = "bubble_chart.png"):
    """
    Bubble plot: X=inference_time_ms, Y=fps, size=gpu_memory_mb, color=model
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    
    # Plot each model
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        ax.scatter(
            model_data["inference_time_ms"],
            model_data["fps"],
            s=model_data["gpu_memory_mb"] / 2,  # Scale for visibility
            alpha=0.6,
            color=MODEL_COLORS.get(model, COLOR_PALETTE["light"]),
            label=display_name,
        )
    
    # Styling
    ax.set_xlabel("Inference Time (ms)", fontsize=14, fontweight="bold")
    ax.set_ylabel("FPS", fontsize=14, fontweight="bold")
    ax.set_title("Pose Estimator Performance: Inference Time vs FPS\n(Bubble size = GPU Memory Usage)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle="--", color=COLOR_PALETTE["medium"])
    ax.legend(fontsize=12, loc="best", framealpha=0.95)
    
    # Set background
    ax.set_facecolor("#F8F9F8")
    fig.patch.set_facecolor("white")
    
    # Add legend for bubble size
    legend_bubbles = [100, 2000, 5000]
    legend_labels = [f"{b:.0f} MB" for b in legend_bubbles]
    for bubble_size, label in zip(legend_bubbles, legend_labels):
        ax.scatter([], [], s=bubble_size/2, c=COLOR_PALETTE["medium_dark"], alpha=0.6, label=label)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=11, loc="best", framealpha=0.95, 
             title="Model / GPU Memory", title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Bubble chart saved to {output_file}")
    plt.close()


def plot_box_plots(df: pd.DataFrame, output_file: str = "boxplots.png"):
    """
    Two subplots: Left = GPU plot (configurable), Right = Grouped bar chart for FPS & Inference Time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor="white")
    fig.suptitle("Performance Analysis: GPU Metrics & Model Comparison", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    models = sorted(df["model"].unique())
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    colors_list = [MODEL_COLORS.get(m, COLOR_PALETTE["light"]) for m in models]
    
    # ===== LEFT PLOT: GPU UTILIZATION (Configurable) =====
    ax = axes[0]
    
    if LEFT_PLOT_TYPE == "line":
        plot_gpu_line_chart(ax, df, models, display_names, colors_list)
    elif LEFT_PLOT_TYPE == "histogram":
        plot_gpu_histogram(ax, df, models, display_names, colors_list)
    elif LEFT_PLOT_TYPE == "candle":
        plot_gpu_candlestick(ax, df, models, display_names, colors_list)
    else:
        raise ValueError(f"Unknown LEFT_PLOT_TYPE: {LEFT_PLOT_TYPE}. Choose 'line', 'histogram', or 'candle'")
    
    # ===== GROUPED BAR CHART FOR FPS & INFERENCE TIME (Right) =====
    ax = axes[1]
    x_pos = np.arange(len(models))
    width = 0.35
    
    avg_times = [df[df["model"] == m]["inference_time_ms"].mean() for m in models]
    avg_fps = [df[df["model"] == m]["fps"].mean() for m in models]
    
    # Plot inference time bars on primary y-axis
    bars1 = ax.bar(x_pos - width/2, avg_times, width, color=colors_list, alpha=0.7, label='Inference Time (ms)')
    
    # Add value labels for inference time
    for bar, val in zip(bars1, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel("Inference Time (ms)", fontsize=12, fontweight="bold", color=COLOR_PALETTE["dark"])
    ax.tick_params(axis='y', labelcolor=COLOR_PALETTE["dark"])
    
    # Create secondary y-axis for FPS
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, avg_fps, width, 
                    color=[COLOR_PALETTE["medium_dark"] for _ in colors_list], 
                    alpha=0.7, label='FPS')
    
    # Add value labels for FPS
    for bar, val in zip(bars2, avg_fps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel("FPS", fontsize=12, fontweight="bold", color=COLOR_PALETTE["medium_dark"])
    ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE["medium_dark"])
    
    # Main axis setup
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Inference Time vs FPS (Per Model)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", color=COLOR_PALETTE["medium"])
    ax.set_facecolor("#F8F9F8")
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Box plots saved to {output_file}")
    plt.close()


def plot_histograms(df: pd.DataFrame, output_file: str = "histograms.png"):
    """
    Histograms showing distribution of inference times, GPU memory, and GPU utilization for each model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.suptitle("Performance Distributions (Histograms)", 
                 fontsize=16, fontweight="bold", y=1.00)
    
    models = sorted(df["model"].unique())
    colors_list = [MODEL_COLORS.get(m, COLOR_PALETTE["light"]) for m in models]
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    
    # Inference Time Distribution
    ax = axes[0, 0]
    for model, color, name in zip(models, colors_list, display_names):
        model_data = df[df["model"] == model]["inference_time_ms"]
        ax.hist(model_data, bins=20, alpha=0.6, label=name, color=color)
    ax.set_xlabel("Inference Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Inference Time Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    
    # GPU Memory Distribution
    ax = axes[0, 1]
    for model, color, name in zip(models, colors_list, display_names):
        model_data = df[df["model"] == model]["gpu_memory_mb"]
        ax.hist(model_data, bins=20, alpha=0.6, label=name, color=color)
    ax.set_xlabel("GPU Memory (MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("GPU Memory Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    
    # GPU Utilization Distribution
    ax = axes[1, 0]
    for model, color, name in zip(models, colors_list, display_names):
        model_data = df[df["model"] == model]["gpu_util_percent"]
        ax.hist(model_data, bins=20, alpha=0.6, label=name, color=color)
    ax.set_xlabel("GPU Utilization (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("GPU Utilization Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    
    # Hide the empty subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Histograms saved to {output_file}")
    plt.close()


def plot_comparison_bars(df: pd.DataFrame, output_file: str = "comparison_bars.png"):
    """
    Bar charts comparing mean values across models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.suptitle("Model Comparison: Average Performance Metrics", 
                 fontsize=16, fontweight="bold", y=1.00)
    
    models = sorted(df["model"].unique())
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    colors_list = [MODEL_COLORS.get(m, COLOR_PALETTE["light"]) for m in models]
    x_pos = np.arange(len(models))
    
    # Average Inference Time
    ax = axes[0, 0]
    avg_times = [df[df["model"] == m]["inference_time_ms"].mean() for m in models]
    bars = ax.bar(x_pos, avg_times, color=colors_list, alpha=0.8, width=0.6)
    ax.set_ylabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Average Inference Time", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    # Add value labels on bars
    for bar, val in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Average FPS
    ax = axes[0, 1]
    avg_fps = [df[df["model"] == m]["fps"].mean() for m in models]
    bars = ax.bar(x_pos, avg_fps, color=colors_list, alpha=0.8, width=0.6)
    ax.set_ylabel("FPS", fontsize=12, fontweight="bold")
    ax.set_title("Average FPS", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    for bar, val in zip(bars, avg_fps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Average GPU Memory
    ax = axes[1, 0]
    avg_mem = [df[df["model"] == m]["gpu_memory_mb"].mean() for m in models]
    bars = ax.bar(x_pos, avg_mem, color=colors_list, alpha=0.8, width=0.6)
    ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
    ax.set_title("Average GPU Memory Usage", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    for bar, val in zip(bars, avg_mem):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Average GPU Utilization
    ax = axes[1, 1]
    avg_util = [df[df["model"] == m]["gpu_util_percent"].mean() for m in models]
    bars = ax.bar(x_pos, avg_util, color=colors_list, alpha=0.8, width=0.6)
    ax.set_ylabel("Utilization (%)", fontsize=12, fontweight="bold")
    ax.set_title("Average GPU Utilization", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_facecolor("#F8F9F8")
    for bar, val in zip(bars, avg_util):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Comparison bars saved to {output_file}")
    plt.close()


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    """Main visualization pipeline."""
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK VISUALIZATION")
    print("="*80)
    print(f"Left plot type: {LEFT_PLOT_TYPE.upper()} (options: 'line', 'histogram', 'candle')")
    print("="*80)
    
    # Load data
    df = load_performance_data()
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_bubble_chart(df)
    plot_box_plots(df)
    plot_histograms(df)
    plot_comparison_bars(df)
    
    print("\n" + "="*80)
    print("✓ All visualizations completed!")
    print("="*80)
    print("\nGenerated files:")
    print("  • bubble_chart.png - Interactive bubble plot (Time vs FPS, bubble size = GPU memory)")
    print(f"  • boxplots.png - Left: {LEFT_PLOT_TYPE.upper()} plot for GPU utilization")
    print("                   Right: Grouped bar chart (Inference Time vs FPS)")
    print("  • histograms.png - Frequency histograms for all metrics")
    print("  • comparison_bars.png - Average metrics comparison bar charts")
    print("\nTo change the left plot, edit LEFT_PLOT_TYPE at the top of the script")
    print("Options: 'line', 'histogram', 'candle'")
    print("="*80)


if __name__ == "__main__":
    main()
